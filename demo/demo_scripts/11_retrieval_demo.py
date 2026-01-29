# %%
import os
import sys
from pathlib import Path

# find project root
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Working directory: {os.getcwd()}")

# %%

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

# check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# check that trained model exists
# use best sweep model if available, otherwise use initial training model
MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "output", "models", "cv-job-matcher-e5-best")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "output", "models", "cv-job-matcher-e5")
    
assert os.path.exists(f"{MODEL_PATH}/config.json"), \
    f"Trained model not found at {MODEL_PATH} - run notebook 09 first!"
print(f"Using model from: {MODEL_PATH}")

# load trained bi-encoder
# load with fp16 for faster inference
bi_encoder = SentenceTransformer(
    MODEL_PATH,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_kwargs={"torch_dtype": torch.float16}  # fp16 precision
)
print(f"Bi-encoder loaded, embedding dim: {bi_encoder.get_sentence_embedding_dimension()}")

# load cross-encoder for reranking
device = "cuda" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)
print(f"Cross-encoder loaded on {device}")

# %%
# load job data with embeddings
jobs_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'jobs_embedded.parquet'))
print(f"Jobs loaded: {len(jobs_df)}")
print(f"Columns: {jobs_df.columns.tolist()}")

# load CV data
CV_DATA_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'cv_query_text.parquet')
cvs_df = pd.read_parquet(CV_DATA_PATH)
print(f"CVs loaded: {len(cvs_df)}")
print(f"Columns: {cvs_df.columns.tolist()}")

# show sample
print("\nSample Job")
print(jobs_df.iloc[0]['embedding_text'][:300])
print("\nSample CV")
print(cvs_df.iloc[0]['text'][:300])

# %%
# create lookup dictionaries for fast ID -> index mapping
job_id_to_idx = {job_id: idx for idx, job_id in enumerate(jobs_df['job_id'])}
cv_id_to_idx = {cv_id: idx for idx, cv_id in enumerate(cvs_df['id'])}

print(f"Created lookups: {len(job_id_to_idx)} jobs, {len(cv_id_to_idx)} CVs")

# %%
# extract job embeddings from dataframe

# embeddings stored as lists in parquet, need to convert to numpy array
job_embeddings = np.array(jobs_df['embedding'].tolist(), dtype=np.float32)
print(f"Job embeddings shape: {job_embeddings.shape}")

# check if already normalized
norms = np.linalg.norm(job_embeddings, axis=1)
print(f"Embedding norms - mean: {norms.mean():.4f}, min: {norms.min():.4f}, max: {norms.max():.4f}")

# normalize for cosine similarity via inner product
if norms.mean() < 0.99 or norms.mean() > 1.01:
    faiss.normalize_L2(job_embeddings)
    norms = np.linalg.norm(job_embeddings, axis=1)
    print(f"After normalization - mean: {norms.mean():.4f}")
else:
    print("Embeddings already normalized")

# %%
# create Faiss index
# using IndexFlatIP (inner product) which equals cosine similarity for normalized vectors
dimension = job_embeddings.shape[1]
jobs_index = faiss.IndexFlatIP(dimension)
jobs_index.add(job_embeddings)

print(f"Jobs Faiss index built:")
print(f" Vectors: {jobs_index.ntotal}")
print(f" Dimension: {dimension}")

# save index
os.makedirs(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes'), exist_ok=True)
faiss.write_index(jobs_index, os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_index.faiss'))
print(f"Index saved to: output/indexes/jobs_index.faiss")
print(f"File size: {os.path.getsize(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'jobs_index.faiss')) / 1e6:.1f} MB")

# %%
# encode CVs with trained bi-encoder

# !!! fix prefix case, cv data has "Query: " e5 needs lowercase "query: "
cv_texts = []
for text in cvs_df['text']:
    # fix uppercase Query to lowercase query
    if text.startswith("Query: "):
        text = "query: " + text[7:]
    elif not text.startswith("query: "):
        text = "query: " + text
    cv_texts.append(text)

print(f"Sample CV text (first 200 chars): {cv_texts[0][:200]}")

# encode in batches
cv_embeddings = bi_encoder.encode(
    cv_texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"\nCV embeddings shape: {cv_embeddings.shape}")

# %%
# embeddings already normalized, convert to float32 for faiss
cv_embeddings = cv_embeddings.astype('float32')

# create index
cvs_index = faiss.IndexFlatIP(dimension)
cvs_index.add(cv_embeddings)
print(f"CVs Faiss index built:")
print(f" Vectors: {cvs_index.ntotal}")
print(f" Dimension: {dimension}")
# save index
faiss.write_index(cvs_index, os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'cvs_index.faiss'))
print(f"Index saved to: output/indexes/cvs_index.faiss")
print(f"File size: {os.path.getsize(os.path.join(PROJECT_ROOT, 'training', 'output', 'indexes', 'cvs_index.faiss')) / 1e6:.1f} MB")
# also save CV embeddings for later use
cv_embeddings_df = pd.DataFrame({
    'cv_id': cvs_df['id'].tolist(),
    'cv_text': cv_texts,
    'embedding': list(cv_embeddings)
})
cv_embeddings_df.to_parquet(os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'cvs_embedded.parquet'))
print(f"CV embeddings saved to: output/embeddings/cvs_embedded.parquet")

# %%
print("CV -> Jobs Retrieval")

# pick a CV from test set (unseen during training)
CV_SPLITS_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output')
test_cv_ids = pd.read_parquet(os.path.join(CV_SPLITS_PATH, 'test_set_cv_ids.parquet'))
sample_cv_id = test_cv_ids.iloc[0]['anchor']
# get CV text
sample_cv_idx = cv_id_to_idx[sample_cv_id]
sample_cv_text = cv_texts[sample_cv_idx]

print(f"\nSample CV ID: {sample_cv_id}")
print(f"\nCV text (first 400 chars):")
print(sample_cv_text[:400])

# %%
# get CV embedding (already computed)
cv_emb = cv_embeddings[sample_cv_idx:sample_cv_idx+1]
# search top-50 jobs
k = 50
scores, indices = jobs_index.search(cv_emb, k)

print(f"Top {k} jobs by bi-encoder similarity:")

top_jobs = []
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    job_id = jobs_df.iloc[idx]['job_id']
    job_text = jobs_df.iloc[idx]['embedding_text']
    top_jobs.append({
        'rank': rank,
        'job_id': job_id,
        'bi_score': score,
        'job_text': job_text,
        'idx': idx
    })
    if rank <= 5:
        print(f"\n{rank}. {job_id} (score: {score:.4f})")
        print(f"   {job_text[:150]}...")

print(f"showing top 5 of {k}")

# %%
# !!! cross-encoder does NOT use prefixes
cv_text_plain = sample_cv_text.replace("query: ", "").replace("Query: ", "")

# create pairs for cross-encoder
pairs = []
for job in top_jobs:
    job_text_plain = job['job_text'].replace("passage: ", "")
    pairs.append((cv_text_plain, job_text_plain))

# score pairs
cross_scores = cross_encoder.predict(pairs, batch_size=128, show_progress_bar=True)

# add cross-encoder scores to results
for job, cross_score in zip(top_jobs, cross_scores):
    job['cross_score'] = float(cross_score)

print(f"Cross-encoder scores computed for {len(top_jobs)} candidates")

# %%
# sort by cross-encoder score
reranked = sorted(top_jobs, key=lambda x: x['cross_score'], reverse=True)
print("Top 10 jobs AFTER cross-encoder reranking:")

for new_rank, job in enumerate(reranked[:10], 1):
    old_rank = job['rank']
    change = old_rank - new_rank
    change_str = f"+{change}" if change > 0 else str(change)
    
    print(f"\n{new_rank}. {job['job_id']} (was rank {old_rank}, {change_str})")
    print(f"  Bi-encoder: {job['bi_score']:.4f} | Cross-encoder: {job['cross_score']:.4f}")
    print(f"  {job['job_text'][:150]}...")

print(f"BEST MATCH: {reranked[0]['job_id']}")
print(f"Cross-encoder score: {reranked[0]['cross_score']:.4f}")

# %%
print("Job -> CVs Retrieval")

# pick a random job
np.random.seed(42)
sample_job_idx = np.random.randint(len(jobs_df))
sample_job = jobs_df.iloc[sample_job_idx]
sample_job_id = sample_job['job_id']
sample_job_text = sample_job['embedding_text']

print(f"\nSample Job ID: {sample_job_id}")
print(f"\nJob text (first 400 chars):")
print(sample_job_text[:400])

# %%
# get job embedding
job_emb = job_embeddings[sample_job_idx:sample_job_idx+1]
# search top CVs
k_cvs = 10
scores, indices = cvs_index.search(job_emb, k_cvs)
print(f"Top {k_cvs} CVs for this job:")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    cv_id = cvs_df.iloc[idx]['id']
    cv_text = cv_texts[idx]
    print(f"\n{rank}. CV {cv_id} (score: {score:.4f})")
    print(f"   {cv_text[:200]}")

# %%
# load validation pairs (CV -> Job ground truth)
val_pairs = pd.read_parquet(os.path.join(PROJECT_ROOT, 'training', 'output', 'training_data', 'validation_pairs.parquet'))
print(f"\nEvaluating on {len(val_pairs)} validation pairs")
print(f"Columns: {val_pairs.columns.tolist()}")
print(f"Sample pair: CV {val_pairs.iloc[0]['anchor']} -> Job {val_pairs.iloc[0]['match']}")

# %%
# compute recall@k
hits_at_1 = 0
hits_at_5 = 0
hits_at_10 = 0
hits_at_50 = 0
total = 0

for _, row in tqdm(val_pairs.iterrows(), total=len(val_pairs), desc="Computing recall"):
    cv_id = row['anchor']
    true_job_id = row['match']
    
    # get CV index
    cv_idx = cv_id_to_idx.get(cv_id)
    if cv_idx is None:
        continue
    
    # get CV embedding
    cv_emb = cv_embeddings[cv_idx:cv_idx+1]
    
    # search
    _, indices = jobs_index.search(cv_emb, 50)
    retrieved_job_ids = [jobs_df.iloc[idx]['job_id'] for idx in indices[0]]
    
    # check if true job in retrieved
    if true_job_id in retrieved_job_ids[:1]:
        hits_at_1 += 1
    if true_job_id in retrieved_job_ids[:5]:
        hits_at_5 += 1
    if true_job_id in retrieved_job_ids[:10]:
        hits_at_10 += 1
    if true_job_id in retrieved_job_ids[:50]:
        hits_at_50 += 1
    
    total += 1

print(f"Results on {total} validation pairs:")

# %%
# display metrics
print("\nRecall Metrics:")
recall_1 = hits_at_1 / total if total > 0 else 0
recall_5 = hits_at_5 / total if total > 0 else 0
recall_10 = hits_at_10 / total if total > 0 else 0
recall_50 = hits_at_50 / total if total > 0 else 0

print(f"\nRecall@1:  {recall_1:.4f} ({hits_at_1}/{total})")
print(f"Recall@5:  {recall_5:.4f} ({hits_at_5}/{total})")
print(f"Recall@10: {recall_10:.4f} ({hits_at_10}/{total})")
print(f"Recall@50: {recall_50:.4f} ({hits_at_50}/{total})")

print("Interpretation:")
print("- Recall@50 > 0.3 bi-encoder retrieves relevant job")
print("- Recall@10 > 0.2 good ranking quality")
print("- Recall@1 shows exact top match accuracy")

# %%
def match_cv_to_jobs(cv_text, top_k=10, rerank=True):
    # match a CV to jobs.
    # Args:
    #     cv_text: CV text (will add prefix if not present)
    #     top_k: Number of matches to return
    #     rerank: Whether to rerank with cross-encoder
    # returns list of dicts with job_id, score, job_text, ensure lowercase prefix

    if cv_text.startswith("Query: "):
        cv_text = "query: " + cv_text[7:]
    elif not cv_text.startswith("query: "):
        cv_text = "query: " + cv_text
    
    # encode CV
    cv_emb = bi_encoder.encode([cv_text], convert_to_numpy=True, normalize_embeddings=True)
    cv_emb = cv_emb.astype('float32')
    
    # retrieve candidates
    n_candidates = 50 if rerank else top_k
    scores, indices = jobs_index.search(cv_emb, n_candidates)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        job_id = jobs_df.iloc[idx]['job_id']
        job_text = jobs_df.iloc[idx]['embedding_text']
        results.append({
            'job_id': job_id,
            'bi_score': float(score),
            'job_text': job_text
        })
    
    if rerank:
        # cross-encoder reranking (no prefixes)
        cv_plain = cv_text.replace("query: ", "")
        pairs = [(cv_plain, r['job_text'].replace("passage: ", "")) for r in results]
        cross_scores = cross_encoder.predict(pairs, batch_size=128)
        
        for r, cs in zip(results, cross_scores):
            r['cross_score'] = float(cs)
        
        results = sorted(results, key=lambda x: x['cross_score'], reverse=True)
    
    return results[:top_k]

print("match_cv_to_jobs() function defined")
print("\nUsage: match_cv_to_jobs('your CV text here', top_k=10, rerank=True)")

# %%
# test the function
print("TEST")

test_cv = "python developer with 5 years experience in Django and PostgreSQL. Machine learning knowledge. AWS and Docker."
print(f"\nTest CV: {test_cv}")

matches = match_cv_to_jobs(test_cv, top_k=5, rerank=True)

print(f"\nTop 5 matches:")
for i, m in enumerate(matches, 1):
    score = m.get('cross_score', m['bi_score'])
    print(f"\n{i}. {m['job_id']} (score: {score:.4f})")
    print(f"   {m['job_text'][:200]}")

# %%
# try another CV
test_cv2 = "marketing manager with 10 years experience in digital marketing, social media, and brand strategy"
print(f"\nTest CV: {test_cv2}")

matches2 = match_cv_to_jobs(test_cv2, top_k=5, rerank=True)

print(f"\nTop 5 matches:")
for i, m in enumerate(matches2, 1):
    score = m.get('cross_score', m['bi_score'])
    print(f"\n{i}. {m['job_id']} (score: {score:.4f})")
    print(f"   {m['job_text'][:200]}...")


