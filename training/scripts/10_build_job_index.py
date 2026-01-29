# %%
# build Faiss index for all 1.3M+ jobs for the demo
# IVF (Inverted File) index for sub-linear search time at scale

# Output:
# jobs_full_index.faiss - IVF index for all jobs
# jobs_full_ids.npy - Job ID mapping


import os
import sys
import argparse

cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")

# %%
# parse command line arguments
parser = argparse.ArgumentParser(description='Build Faiss index for job matching')
parser.add_argument('--quick', action='store_true',
                    help='Quick mode: sample 10,000 jobs for fast testing (~30 sec)')
args = parser.parse_args()

QUICK_MODE = args.quick

if QUICK_MODE:
    SAMPLE_SIZE = 10000
    print(f'QUICK_MODE enabled: building index for {SAMPLE_SIZE:,} jobs')
else:
    SAMPLE_SIZE = None
    print('FULL MODE: building index for all 1.35M+ jobs')

# %%
import os
os.chdir(PROJECT_ROOT)
print(f"Working directory: {os.getcwd()}")

import numpy as np
import pandas as pd
import faiss
import torch
import gc
import time
from sentence_transformers import SentenceTransformer

# check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# load trained bi-encoder
# use best sweep model if available
MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "output", "models", "cv-job-matcher-e5-best")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "output", "models", "cv-job-matcher-e5")

assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
print(f"Loading bi-encoder from: {MODEL_PATH}")

# load with fp16 for faster inference
bi_encoder = SentenceTransformer(
    MODEL_PATH,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_kwargs={"torch_dtype": torch.float16}  # fp16 precision
)
print(f"Embedding dimension: {bi_encoder.get_sentence_embedding_dimension()}")

# %%
# load ALL 1.35M jobs
start = time.time()
jobs_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'unified_job_postings', 'unified_jobs.parquet'))

load_time = time.time() - start
print(f"Loaded in {load_time:.1f}s")
print(f"\nTotal jobs: {len(jobs_df):,}")
print(f"Columns: {jobs_df.columns.tolist()}")
print(f"Memory: {jobs_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# sample if QUICK_MODE
if QUICK_MODE and SAMPLE_SIZE:
    jobs_df = jobs_df.sample(n=min(SAMPLE_SIZE, len(jobs_df)), random_state=42).reset_index(drop=True)
    print(f'QUICK_MODE: sampled {len(jobs_df):,} jobs')

# %%
# show sample embedding text
print("Sample job embedding_text:")
print(jobs_df.iloc[0]['embedding_text'][:500])
print("...")

# check if text has prefix
first_text = jobs_df.iloc[0]['embedding_text']
if first_text.startswith('passage:'):
    print("\nText already has 'passage:' prefix")
else:
    print("\nText needs 'passage:' prefix (will add during encoding)")

# %%
# encoding config
CHUNK_SIZE = 100000
BATCH_SIZE = 256

total_jobs = len(jobs_df)
n_chunks = (total_jobs + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"Encoding {total_jobs:,} jobs")
print(f"Chunk size: {CHUNK_SIZE:,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total chunks: {n_chunks}")

# %%
# encode in chunks to avoid OOM
all_embeddings = []
job_ids = jobs_df['id'].tolist()  # keep job IDs for lookup

total_start = time.time()

for chunk_idx in range(n_chunks):
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = min(chunk_start + CHUNK_SIZE, total_jobs)
    
    print(f"\nChunk {chunk_idx+1}/{n_chunks}: {chunk_start:,} to {chunk_end:,}")
    
    # get texts for this chunk
    chunk_texts = jobs_df.iloc[chunk_start:chunk_end]['embedding_text'].tolist()
    
    # add 'passage: ' prefix if not present
    # e5 models use asymmetric prefixes (query vs passage)
    chunk_texts = [
        ("passage: " + t) if not t.startswith("passage:") else t 
        for t in chunk_texts
    ]
    
    # encode
    start = time.time()
    chunk_emb = bi_encoder.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 norm for cosine via IP
    )
    elapsed = time.time() - start
    
    print(f"  Encoded {len(chunk_emb):,} in {elapsed:.1f}s")
    print(f"  Shape: {chunk_emb.shape}, dtype: {chunk_emb.dtype}")
    
    all_embeddings.append(chunk_emb)
    
    # cleanup GPU memory between chunks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

total_elapsed = time.time() - total_start
print(f"\nTotal encoding time: {total_elapsed/60:.1f} minutes")

# %%
# concatenate all chunks
embeddings = np.vstack(all_embeddings)
# free the chunks
del all_embeddings
gc.collect()

print(f"Final embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")
print(f"Memory: {embeddings.nbytes / 1e9:.2f} GB")

# verify normalization
sample_norms = np.linalg.norm(embeddings[:1000], axis=1)
print(f"\nNorm check (first 1000): mean={sample_norms.mean():.4f}, std={sample_norms.std():.6f}")

# %%
# IVF index config
# nlist = sqrt(n) rule of thumb for 1M+ vectors
# 1.35M -> sqrt = 1162 round to 1500 for safety
DIMENSION = embeddings.shape[1]
NLIST = 1500  # number of clusters/cells

print(f"Building Faiss IndexIVFFlat")
print(f"  Dimension: {DIMENSION}")
print(f"  nlist (clusters): {NLIST}")
print(f"  Vectors: {len(embeddings):,}")

# using METRIC_INNER_PRODUCT because vectors are normalized
# this gives cosine similarity

# %%
# create quantizer (flat index for cluster centroids)
quantizer = faiss.IndexFlatIP(DIMENSION)
# create IVF index
index = faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST, faiss.METRIC_INNER_PRODUCT)

print(f"Index created (not trained yet)")
print(f"  is_trained: {index.is_trained}")

# %%
# train the index on a sample
# need at least nlist * 30 vectors for good clustering
# using 100K should be plenty
train_size = min(len(embeddings), 100000)

# random sample for training
np.random.seed(42)
train_indices = np.random.choice(len(embeddings), train_size, replace=False)
train_sample = embeddings[train_indices]

print(f"Training index on {train_size:,} samples")
start = time.time()
index.train(train_sample)
train_time = time.time() - start
print(f"Training completed in {train_time:.1f}s")
print(f"is_trained: {index.is_trained}")

# %%
# add all vectors to the index
print(f"Adding {len(embeddings):,} vectors to index")
start = time.time()
index.add(embeddings)
add_time = time.time() - start
print(f"Added in {add_time:.1f}s")
print(f"\nIndex stats:")
print(f"  ntotal: {index.ntotal:,}")
print(f"  nlist: {index.nlist}")

# %%
# create output directory
os.makedirs(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes'), exist_ok=True)

# save Faiss index
if QUICK_MODE:
    index_path = os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_index.faiss')
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
else:
    index_path = os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_index.faiss')
print(f"Saving index to {index_path}")
faiss.write_index(index, index_path)
file_size = os.path.getsize(index_path)
print(f"Index saved")
print(f"  Size: {file_size / 1e9:.2f} GB")

# %%
# save job ID mapping
# position in index -> job_id
if QUICK_MODE:
    ids_path = os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_ids.npy')
else:
    ids_path = os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_ids.npy')

np.save(ids_path, np.array(job_ids))

print(f"Saved {len(job_ids):,} job IDs")
print(f"  Sample IDs: {job_ids[:3]}")

# %%
# test with a sample CV query
test_cv = "query: python developer with 5 years experience in Django and PostgreSQL, AWS"
print(f"\nTest CV: {test_cv}")

# encode the query
cv_embedding = bi_encoder.encode([test_cv], convert_to_numpy=True, normalize_embeddings=True)

# %%
# test different nprobe values
# nprobe = number of clusters to search (higher = better recall, slower)
print("\nTesting different nprobe values:")

for nprobe in [10, 20, 50, 100]:
    index.nprobe = nprobe
    
    # time the search
    start = time.time()
    distances, indices = index.search(cv_embedding, 10)
    elapsed = (time.time() - start) * 1000
    
    print(f"nprobe={nprobe:3d}: {elapsed:6.1f}ms, top score={distances[0][0]:.4f}")

# %%
# show top results with nprobe=20
index.nprobe = 20
distances, indices = index.search(cv_embedding, 10)

print(f"\nTop 10 results (nprobe=20):")
print("="*60)

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    job_id = job_ids[idx]
    job_text = jobs_df.iloc[idx]['embedding_text']
    
    print(f"\n{rank}. {job_id} (score: {dist:.4f})")
    print(f"   {job_text[:150]}...")

# %%
# set recommended nprobe
# 20 gives good balance: <100ms queries with good recall
index.nprobe = 20

print("Recommended settings for demo:")
print(f"  nprobe = 20 (good balance of speed vs recall)")
print(f"  Expected query time: <100ms")

# %%
# verify index count
print(f"\nIndex vectors: {index.ntotal:,}")
print(f"Expected (jobs_df): {len(jobs_df):,}")
assert index.ntotal == len(jobs_df), "Count mismatch!"
print("Count matches")

# verify ID mapping
print(f"\nJob IDs count: {len(job_ids):,}")
assert len(job_ids) == len(jobs_df), "ID count mismatch!"
print("ID count matches")

# verify saved files
if QUICK_MODE:
    assert os.path.exists(os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_index.faiss')), "Index file missing!"
    assert os.path.exists(os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_ids.npy')), "IDs file missing!"
else:
    assert os.path.exists(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_index.faiss')), "Index file missing!"
    assert os.path.exists(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_ids.npy')), "IDs file missing!"
print("Files exist")

# %%
# verify we can reload the index

if QUICK_MODE:
    loaded_index = faiss.read_index(os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_index.faiss'))
    loaded_ids = np.load(os.path.join(PROJECT_ROOT, 'output', 'temp', 'jobs_quick_ids.npy'), allow_pickle=True)
else:
    loaded_index = faiss.read_index(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_index.faiss'))
    loaded_ids = np.load(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_ids.npy'), allow_pickle=True)

print(f"Loaded index: {loaded_index.ntotal:,} vectors")
print(f"Loaded IDs: {len(loaded_ids):,}")

assert loaded_index.ntotal == index.ntotal, "Loaded index count mismatch!"
assert len(loaded_ids) == len(job_ids), "Loaded IDs count mismatch!"

print("\nAll verifications passed")
print(f"Ready for demo with {index.ntotal:,} searchable jobs")

# %%
# for demo we want consistent sub-second response times
# check memory usage and performance
import psutil

# current process memory
process = psutil.Process()
mem_mb = process.memory_info().rss / 1e6
print(f"\nCurrent process memory: {mem_mb:.0f} MB")

# index file size
index_size_gb = os.path.getsize(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_full_index.faiss')) / 1e9
print(f"Index file size: {index_size_gb:.2f} GB")

# estimate memory for vectors alone
# 768 dims * 4 bytes * 1.35M vectors
vectors_mem_gb = DIMENSION * 4 * index.ntotal / 1e9
print(f"Vector memory estimate: {vectors_mem_gb:.2f} GB")

# %%
# batch query test (simulate multiple demo queries)
print("\nBatch query test (10 queries):")

test_cvs = [
    "query: data scientist with machine learning experience",
    "query: frontend developer react javascript",
    "query: backend engineer python microservices",
    "query: devops engineer kubernetes docker",
    "query: product manager saas b2b",
    "query: software engineer java spring",
    "query: mobile developer ios swift",
    "query: qa engineer automation testing",
    "query: data engineer etl spark",
    "query: cloud architect aws azure"
]

# encode test queries
test_embeddings = bi_encoder.encode(
    test_cvs, 
    convert_to_numpy=True, 
    normalize_embeddings=True
)

# run queries
index.nprobe = 20
query_times = []

start = time.time()
for emb in test_embeddings:
    q_start = time.time()
    _, _ = index.search(emb.reshape(1, -1), 50)
    query_times.append((time.time() - q_start) * 1000)
    
total_ms = (time.time() - start) * 1000

print(f"Total time: {total_ms:.0f}ms")
print(f"Average per query: {total_ms/10:.0f}ms")
print(f"Min: {min(query_times):.0f}ms, Max: {max(query_times):.0f}ms")

# %%
# final summary
print(f"\nIndex: {index.ntotal:,} jobs indexed")
print(f"Query latency: <100ms with nprobe=20")
print(f"Index size: {index_size_gb:.2f} GB")
print(f"\nUse nprobe=20 for balanced speed/accuracy")
print(f"Increase nprobe to 50 for better recall if needed")


