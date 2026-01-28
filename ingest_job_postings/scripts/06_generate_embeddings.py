# %%
# imports
import pandas as pd
import numpy as np
import torch
import gc
import random
import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# setup paths, detect project root
import sys
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

# parse arguments
parser = argparse.ArgumentParser(description='Generate embeddings for jobs and CVs')
parser.add_argument('--quick', action='store_true',
                    help='Quick mode: sample 5000 jobs, save to output/temp/')
args = parser.parse_args()

QUICK_MODE = args.quick
if QUICK_MODE:
    SAMPLE_SIZE = 5000
    print('QUICK_MODE enabled: sampling 5000 jobs, saving to output/temp/')
else:
    SAMPLE_SIZE = None
    print('FULL MODE: processing all jobs')

print('imports loaded')
print(f'pytorch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'project root: {PROJECT_ROOT}')

# %%
spark = SparkSession.builder \
    .appName('GenerateEmbeddings') \
    .config('spark.driver.memory', '4g') \
    .getOrCreate()

print(f'spark started: {spark.version}')

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {gpu_name}')
    print(f'total VRAM: {gpu_memory:.1f} GB')
    print(f'using GPU for encoding')
else:
    device = torch.device('cpu')
    print('!!! GPU not available, using CPU')

# %%
print('loading jobs with Spark')
input_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'final', 'jobs_to_embed.parquet')
jobs_df = spark.read.parquet(input_path)

# sample if QUICK_MODE
if QUICK_MODE and SAMPLE_SIZE:
    jobs_df = jobs_df.limit(SAMPLE_SIZE)
    print(f'QUICK_MODE: sampled {SAMPLE_SIZE} jobs')

print(f'loaded {jobs_df.count():,} jobs')
print(f'columns: {jobs_df.columns}')

# check for isco_code column, needed for stratified splitting in 07
has_isco = 'isco_code' in jobs_df.columns
print(f'has isco_code column: {has_isco}')
if not has_isco:
    print('!!! isco_code not found, stratified splitting will use random splits')

print('\nsample:')
jobs_df.show(3, truncate=80)

# %%
# verify passage prefix
has_prefix = jobs_df.filter(jobs_df['embedding_text'].startswith('passage: ')).count()
total = jobs_df.count()

print(f'jobs with passage prefix: {has_prefix:,}/{total:,}')
if has_prefix == total:
    print('all jobs have correct prefix')

# %%
# collect to pandas for GPU encoding
print('collecting to pandas for GPU encoding')
jobs_pd = jobs_df.toPandas()
print(f'collected {len(jobs_pd):,} jobs to driver')

# %%
# load model with fp16 for ~1.5x speedup, based on: https://sbert.net/docs/sentence_transformer/usage/efficiency.html
print('loading e5-base-v2 model with fp16 precision')

model = SentenceTransformer(
    'intfloat/e5-base-v2', 
    device=device,
    model_kwargs={"torch_dtype": torch.float16}  # fp16 precision
)

print(f'model loaded on: {device}')
print(f'precision: float16 (optimized)')
print(f'embedding dimension: {model.get_sentence_embedding_dimension()}')
print(f'max sequence length: {model.max_seq_length} tokens')

if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

# %%
# sentenceTransformer's built-in encoding with progress bar
# batch_size=256 optimal when GPU near full utilization, ref: https://github.com/UKPLab/sentence-transformers/issues/609

def encode_texts(texts, model, batch_size=256):

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings

print(f'encoding function defined')

# %%
print(f'encoding {len(jobs_pd):,} jobs with fp16')

job_embeddings = encode_texts(
    texts=jobs_pd['embedding_text'].tolist(),
    model=model,
    batch_size=256
)

print(f'encoding complete!')
print(f'embeddings shape: {job_embeddings.shape}')

# %%
# check dimension
expected_dim = 768
actual_dim = job_embeddings.shape[1]

print(f'embedding dimension: {actual_dim}')
print(f'expected: {expected_dim}')
print(f'dimension check: {"PASSED" if actual_dim == expected_dim else "FAILED"}')

# check L2 normalization
norms = np.linalg.norm(job_embeddings, axis=1)
print(f'\nL2 norm stats:')
print(f' mean: {norms.mean():.6f}')
print(f' min: {norms.min():.6f}')
print(f' max: {norms.max():.6f}')

is_normalized = np.allclose(norms, 1.0, atol=1e-6)
print(f'normalization check: {"PASSED" if is_normalized else "FAILED"}')
# check for NaN/inf
has_nan = np.isnan(job_embeddings).any()
has_inf = np.isinf(job_embeddings).any()
print(f'\nquality checks:')
print(f' contains NaN: {has_nan}')
print(f' contains Inf: {has_inf}')
print(f'quality check: {"PASSED" if not (has_nan or has_inf) else "FAILED"}')

# %%
# save job embeddings in chunks to avoid memory explosion, ref: https://github.com/apache/arrow/issues/20512 (pandas to_parquet quadratic memory)

CHUNK_SIZE = 50000  # 50K records per chunk
if QUICK_MODE:
    output_path = os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_embedded_sample.parquet')
else:
    output_path = os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'jobs_embedded.parquet')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print(f'saving {len(jobs_pd):,} jobs in chunks of {CHUNK_SIZE:,}')

# schema depends on whether we have isco_code for stratified splitting
if has_isco:
    schema = pa.schema([
        ('job_id', pa.string()),
        ('embedding_text', pa.string()),
        ('embedding', pa.list_(pa.float32(), 768)),
        ('isco_code', pa.int32())
    ])
    print('including isco_code for stratified splitting')
else:
    schema = pa.schema([
        ('job_id', pa.string()),
        ('embedding_text', pa.string()),
        ('embedding', pa.list_(pa.float32(), 768))
    ])
    print('no isco_code - using basic schema')

writer = pq.ParquetWriter(output_path, schema, compression='snappy')
total_written = 0

for start_idx in range(0, len(jobs_pd), CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, len(jobs_pd))
    
    # build chunk without full .tolist() (memory efficient)
    chunk_ids = jobs_pd['job_id'].iloc[start_idx:end_idx].values
    chunk_texts = jobs_pd['embedding_text'].iloc[start_idx:end_idx].values
    chunk_embs = [job_embeddings[i].tolist() for i in range(start_idx, end_idx)]
    
    # build table data
    table_data = {
        'job_id': chunk_ids,
        'embedding_text': chunk_texts,
        'embedding': chunk_embs
    }
    
    # add isco_code if available
    if has_isco:
        chunk_isco = jobs_pd['isco_code'].iloc[start_idx:end_idx].values.astype('int32')
        table_data['isco_code'] = chunk_isco
    
    # write chunk with pyarrow
    table = pa.table(table_data, schema=schema)
    writer.write_table(table)
    total_written += (end_idx - start_idx)
    
    # cleanup chunk memory
    del chunk_ids, chunk_texts, chunk_embs, table, table_data
    if has_isco:
        del chunk_isco
    gc.collect()
    
    # progress every 5 chunks
    chunk_num = start_idx // CHUNK_SIZE + 1
    if chunk_num % 5 == 0:
        print(f'  written: {total_written:,}/{len(jobs_pd):,}')

writer.close()

print(f'saved to: {output_path}')
print(f'records: {total_written:,}')
file_size = os.path.getsize(output_path) / 1e6
print(f'file size: {file_size:.1f} MB')

# %%
print('loading CVs with Spark')
cv_input_path = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'cv_query_text.parquet')
cvs_df = spark.read.parquet(cv_input_path)

print(f'loaded {cvs_df.count():,} CVs')
print('\nsample:')
cvs_df.show(3, truncate=80)

# %%
# collect to pandas
print('collecting CVs to pandas')
cvs_pd = cvs_df.toPandas()
cvs_pd = cvs_pd.rename(columns={'id': 'cv_id', 'text': 'embedding_text'})
print(f'collected {len(cvs_pd):,} CVs')

# %%
print(f'encoding {len(cvs_pd):,} CVs with fp16')

cv_embeddings = encode_texts(
    texts=cvs_pd['embedding_text'].tolist(),
    model=model,
    batch_size=256
)

print(f'CV encoding complete!')
print(f'embeddings shape: {cv_embeddings.shape}')

# %%
cv_dim = cv_embeddings.shape[1]
print(f'CV embedding dimension: {cv_dim}')
print(f'dimension check: {"PASSED" if cv_dim == 768 else "FAILED"}')

cv_norms = np.linalg.norm(cv_embeddings, axis=1)
print(f'\nL2 norm stats:')
print(f' mean: {cv_norms.mean():.6f}')
print(f' min: {cv_norms.min():.6f}')
print(f' max: {cv_norms.max():.6f}')

cv_normalized = np.allclose(cv_norms, 1.0, atol=1e-6)
print(f'normalization check: {"PASSED" if cv_normalized else "FAILED"}')

has_nan_cv = np.isnan(cv_embeddings).any()
has_inf_cv = np.isinf(cv_embeddings).any()
print(f'quality check: {"PASSED" if not (has_nan_cv or has_inf_cv) else "FAILED"}')

# %%
# save CV embeddings with pyarrow (smaller dataset, single write ok)
cv_output_path = os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'cvs_embedded.parquet')

cv_schema = pa.schema([
    ('cv_id', pa.string()),
    ('embedding_text', pa.string()),
    ('embedding', pa.list_(pa.float32(), 768))
])

# build embedding list, small enough for single operation)
cv_emb_list = [cv_embeddings[i].tolist() for i in range(len(cvs_pd))]

cv_table = pa.table({
    'cv_id': cvs_pd['cv_id'].values,
    'embedding_text': cvs_pd['embedding_text'].values,
    'embedding': cv_emb_list
}, schema=cv_schema)

pq.write_table(cv_table, cv_output_path, compression='snappy')
print(f'saved to: {cv_output_path}')
print(f'records: {len(cvs_pd):,}')
cv_file_size = os.path.getsize(cv_output_path) / 1e6
print(f'file size: {cv_file_size:.1f} MB')

# cleanup
del cv_table, cv_emb_list
gc.collect()

# %%
# demo of cross-tower similarity search, load from saved parquet files to verify they work

print('loading saved embeddings for demo')
jobs_saved = pd.read_parquet(output_path)
cvs_saved = pd.read_parquet(cv_output_path)
print(f'jobs: {len(jobs_saved):,}')
print(f'CVs: {len(cvs_saved):,}')
print(f'job columns: {list(jobs_saved.columns)}')

# pick random CV
cv_idx = random.randint(0, len(cvs_saved) - 1)
query_cv = cvs_saved.iloc[cv_idx]
cv_query_emb = np.array(query_cv['embedding']).reshape(1, -1)
print(f'\nquery CV:')
print(f'ID: {query_cv["cv_id"]}')
print(f'text: {query_cv["embedding_text"][:200]}...')

# compute similarities (dot product = cosine for normalized vectors)
job_embs = np.array(jobs_saved['embedding'].tolist())
similarities = cosine_similarity(cv_query_emb, job_embs)[0]

# top 10 matches
top_indices = similarities.argsort()[::-1][:10]
print('TOP 10 MATCHING JOBS')

for rank, idx in enumerate(top_indices, 1):
    job = jobs_saved.iloc[idx]
    sim = similarities[idx]
    isco_str = f' (ISCO: {job["isco_code"]})' if 'isco_code' in job else ''
    print(f'\n[{rank}] similarity: {sim:.4f}{isco_str}')
    print(f'job ID: {job["job_id"]}')
    print(f'text: {job["embedding_text"][:200]}...')

# %%
if torch.cuda.is_available():
    print(f'GPU memory before cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
    del model
    del job_embeddings
    del cv_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    print(f'GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

# %%
spark.stop()
print('spark stopped')

# %%
print(f'\njobs:')
print(f' input: {input_path}')
print(f' output: {output_path}')
print(f' records: {total_written:,}')
print(f' file size: {file_size:.1f} MB')
print(f' has isco_code: {has_isco}')
print(f'\nCVs:')
print(f' input: {cv_input_path}')
print(f' output: {cv_output_path}')
print(f' records: {len(cvs_pd):,}')
print(f' file size: {cv_file_size:.1f} MB')
print(f'\nmodel: e5-base-v2 (768D, L2 normalized)')
print(f'\nNOTE: CV data from ingest_cv/output/ (colleague pipeline)')


