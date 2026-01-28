# %% [markdown]
# ## Section 1: Setup and Data Loading

# %%
import os
import sys

# Find project root
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")

# %%
# Setup working directory
import os
os.chdir(PROJECT_ROOT)
print(f"Working directory: {os.getcwd()}")

CV_DATA_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'cv_query_text.parquet')
JOBS_EMB_PATH = os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'jobs_embedded.parquet')

print(f"CV data exists: {os.path.exists(CV_DATA_PATH)}")
print(f"Jobs data exists: {os.path.exists(JOBS_EMB_PATH)}")

# %%
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import CrossEncoder, SentenceTransformer
import faiss
import os
from tqdm import tqdm

print("Imports successful")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# %%
#load CV query texts (7299 CVs with Query: prefix)
cv_df = pd.read_parquet(CV_DATA_PATH)
print(f"Loaded CV data: {cv_df.shape}")
print(f"Columns: {cv_df.columns.tolist()}")
print(f"First CV ID: {cv_df.iloc[0]['id']}")
print(f"First CV text (first 400 chars): {cv_df.iloc[0]['text'][:400]}")

# %%
#load job embeddings (165K jobs with passage: prefix)
jobs_df = pd.read_parquet(JOBS_EMB_PATH)
print(f"Loaded jobs data: {jobs_df.shape}")
print(f"Columns: {jobs_df.columns.tolist()}")
print(f"First job ID: {jobs_df.iloc[0]['job_id']}")
print(f"First job text (first 400 chars): {jobs_df.iloc[0]['embedding_text'][:400]}")
print(f"Embedding shape: {jobs_df.iloc[0]['embedding'].shape if hasattr(jobs_df.iloc[0]['embedding'], 'shape') else 'N/A'}")

# %%
# load train/val/test split IDs
CV_SPLITS_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output')
train_ids = pd.read_parquet(os.path.join(CV_SPLITS_PATH, 'training_set_cv_ids.parquet'))
val_ids = pd.read_parquet(os.path.join(CV_SPLITS_PATH, 'validation_set_cv_ids.parquet'))
test_ids = pd.read_parquet(os.path.join(CV_SPLITS_PATH, 'test_set_cv_ids.parquet'))

print(f"Training set: {len(train_ids)} CVs")
print(f"Validation set: {len(val_ids)} CVs")
print(f"Test set: {len(test_ids)} CVs")
print(f"Total: {len(train_ids) + len(val_ids) + len(test_ids)} CVs")
print()
print(f"Sample training IDs: {train_ids['anchor'].iloc[:5].tolist()}")
print(f"Sample validation IDs: {val_ids['anchor'].iloc[:5].tolist()}")

# %%
# !!! fix CV prefix from "Query: " to "query: " (lowercase)
# This must happen BEFORE any encoding for e5-base-v2
cv_df['text_fixed'] = cv_df['text'].apply(lambda x: x.replace("Query: ", "query: "))

# Verify fix
print(f"Original prefix: {cv_df.iloc[0]['text'][:50]}")
print(f"Fixed prefix: {cv_df.iloc[0]['text_fixed'][:50]}")
assert cv_df.iloc[0]['text_fixed'].startswith("query: "), "ERROR: Prefix fix failed!"
print("Prefix fix VERIFIED")
print()

# %%
# Extract job embeddings and build Faiss index

job_embeddings = np.array(jobs_df['embedding'].tolist()).astype('float32')
print(f"Job embeddings shape: {job_embeddings.shape}")

# Normalize embeddings for cosine similarity (IndexFlatIP)
faiss.normalize_L2(job_embeddings)
print("Embeddings normalized")

# Build index
dimension = job_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(job_embeddings)
print(f"Index built with {index.ntotal} jobs")

# %%
# Load e5-base-v2 model to encode CVs
bi_encoder = SentenceTransformer('intfloat/e5-base-v2')
print("e5-base-v2 model for CV encoding loaded")

# %%
def generate_candidate_pairs(cv_ids_df, cv_data, top_k=50):
    # generate candidate pairs for given CV IDs
    #returns list of (cv_id, job_id) tuples

    print(f"Generating candidates for {len(cv_ids_df)} CVs (top-{top_k} per CV)")
    
    # filter CVs to only those in the split
    cv_subset = cv_data[cv_data['id'].isin(cv_ids_df['anchor'])].copy()
    print(f"Filtered to {len(cv_subset)} CVs")
    
    # encode CV texts (with fixed lowercase prefix)
    cv_texts = cv_subset['text_fixed'].tolist()
    print(f"encoding {len(cv_texts)} CV texts")
    cv_embeddings = bi_encoder.encode(cv_texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"CV embeddings shape: {cv_embeddings.shape}")
    
    # normalize for cosine similarity
    cv_embeddings = cv_embeddings.astype('float32')
    faiss.normalize_L2(cv_embeddings)
    print("CV embeddings normalized")
    
    # search for top-k jobs per CV
    print(f"searching for top-{top_k} jobs per CV")
    scores, indices = index.search(cv_embeddings, top_k)
    print(f"Search complete: {scores.shape}")
    
    # build candidate pairs
    pairs = []
    for i, cv_id in enumerate(cv_subset['id']):
        for j in range(top_k):
            job_idx = indices[i][j]
            job_id = jobs_df.iloc[job_idx]['job_id']
            pairs.append((cv_id, job_id))
    
    print(f"Generated {len(pairs)} candidate pairs")
    return pairs

print("Function defined")

# %%
# generate training candidates
print("TRAINING SET")
training_candidates = generate_candidate_pairs(train_ids, cv_df, top_k=50)
print(f"Total training candidates: {len(training_candidates)}")
print(f"Expected: {len(train_ids)} CVs * 50 = {len(train_ids) * 50}")

# %%
# generate validation candidates
print("validation set")
validation_candidates = generate_candidate_pairs(val_ids, cv_df, top_k=50)
print(f"Total validation candidates: {len(validation_candidates)}")
print(f"Expected: {len(val_ids)} CVs * 50 = {len(val_ids) * 50}")

# %%
# Load cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')
print("Cross-encoder (ms-marco-MiniLM-L12-v2) loaded")

# %%
def score_pairs_with_cross_encoder(candidates, cv_data, job_data, batch_size=128):
    #score candidate pairs with cross-encoder
    # returns DataFrame with cv_id, job_id, score
    print(f"Scoring {len(candidates)} pairs with cross-encoder")
    
    # build lookup dicts for fast access
    cv_text_map = dict(zip(cv_data['id'], cv_data['text_fixed']))
    job_text_map = dict(zip(job_data['job_id'], job_data['embedding_text']))
    
    # prepare text pairs for cross-encoder
    # !!!! Cross-encoder does NOT use prefixes, so we strip them
    text_pairs = []
    for cv_id, job_id in candidates:
        cv_text = cv_text_map[cv_id]
        job_text = job_text_map[job_id]
        
        # strip prefixes
        cv_text_clean = cv_text.replace("query: ", "")
        job_text_clean = job_text.replace("passage: ", "")
        
        text_pairs.append([cv_text_clean, job_text_clean])
    
    print(f"Prepared {len(text_pairs)} text pairs")
    print(f"Sample pair:")
    print(f" CV (first 300 chars): {text_pairs[0][0][:300]}")
    print(f" Job (first 300 chars): {text_pairs[0][1][:300]}")
    
    # score in batches
    print(f"Scoring in batches of {batch_size}")
    scores = cross_encoder.predict(text_pairs, batch_size=batch_size, show_progress_bar=True)
    print(f"Scored {len(scores)} pairs")
    
    # build results DataFrame
    results = pd.DataFrame({
        'cv_id': [pair[0] for pair in candidates],
        'job_id': [pair[1] for pair in candidates],
        'score': scores
    })
    
    print(f"Results shape: {results.shape}")
    print(f"Score stats: min={results['score'].min():.4f}, max={results['score'].max():.4f}, mean={results['score'].mean():.4f}")
    
    return results

print("Function defined")

# %%
# score training pairs
print("SCORING TRAINING PAIRS")
training_scores_df = score_pairs_with_cross_encoder(training_candidates, cv_df, jobs_df, batch_size=128)
print(f"Training scores complete: {training_scores_df.shape}")

# %%
# score validation pairs
print("SCORING VALIDATION PAIRS")
validation_scores_df = score_pairs_with_cross_encoder(validation_candidates, cv_df, jobs_df, batch_size=128)
print(f"Validation scores complete: {validation_scores_df.shape}")

# %%
def select_best_matches(scores_df):
    # for each CV, select the job with highest score
    # returns DataFrame with anchor (CV), match (Job)

    print(f"Selecting best matches from {len(scores_df)} scored pairs")
    
    #group by CV and select argmax
    best_matches = scores_df.loc[scores_df.groupby('cv_id')['score'].idxmax()]
    print(f"Selected {len(best_matches)} best matches")
    
    #rrename columns to match expected format
    result = best_matches[['cv_id', 'job_id', 'score']].copy()
    result.columns = ['anchor', 'match', 'score']
    
    print(f"Result shape: {result.shape}")
    print(f"Best match score stats: min={result['score'].min():.4f}, max={result['score'].max():.4f}, mean={result['score'].mean():.4f}")
    
    return result

print("Function defined")

# %%
#select best training matches
print("TRAINING BEST MATCHES")
training_pairs = select_best_matches(training_scores_df)
print(f"Training pairs: {len(training_pairs)}")
print(f"Expected: {len(train_ids)} (one per CV)")
print(f"Sample pairs:")
print(training_pairs.head(10))

# %%
#best validation matches
print("VALIDATION BEST MATCHES")
validation_pairs = select_best_matches(validation_scores_df)
print(f"Validation pairs: {len(validation_pairs)}")
print(f"Expected: {len(val_ids)} (one per CV)")
print(f"Sample pairs:")
print(validation_pairs.head(10))

# %%
#output directory
output_dir = os.path.join(PROJECT_ROOT, 'training', 'output', 'training_data')
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# %%
# save training pairs (ID only)
training_pairs_to_save = training_pairs[['anchor', 'match']].copy()
training_pairs_to_save.to_parquet(f'{output_dir}/training_pairs.parquet', index=False)
training_pairs_to_save.to_csv(f'{output_dir}/training_pairs.csv', index=False)
print(f"Saved training pairs: {len(training_pairs_to_save)} rows")
print(f" - {output_dir}/training_pairs.parquet")
print(f" - {output_dir}/training_pairs.csv")

# %%
# save validation pairs (ID only)
validation_pairs_to_save = validation_pairs[['anchor', 'match']].copy()
validation_pairs_to_save.to_parquet(f'{output_dir}/validation_pairs.parquet', index=False)
validation_pairs_to_save.to_csv(f'{output_dir}/validation_pairs.csv', index=False)
print(f"Saved validation pairs: {len(validation_pairs_to_save)} rows")
print(f" - {output_dir}/validation_pairs.parquet")
print(f" - {output_dir}/validation_pairs.csv")

# %%
# save training text data for bi-encoder

#build CV text lookup
cv_text_map = dict(zip(cv_df['id'], cv_df['text_fixed']))
job_text_map = dict(zip(jobs_df['job_id'], jobs_df['embedding_text']))
#add text to training pairs
training_dataset = training_pairs.copy()
training_dataset['anchor_text'] = training_dataset['anchor'].map(cv_text_map)
training_dataset['positive_text'] = training_dataset['match'].map(job_text_map)
# save
training_dataset[['anchor_text', 'positive_text']].to_parquet(f'{output_dir}/training_dataset.parquet', index=False)
print(f"Saved training dataset: {len(training_dataset)} rows")
print(f" - {output_dir}/training_dataset.parquet")
print(f" Columns: {training_dataset.columns.tolist()}")

# %%
#save validation text data for bi-encoder

#add text to validation pairs
validation_dataset = validation_pairs.copy()
validation_dataset['anchor_text'] = validation_dataset['anchor'].map(cv_text_map)
validation_dataset['positive_text'] = validation_dataset['match'].map(job_text_map)
# save
validation_dataset[['anchor_text', 'positive_text']].to_parquet(f'{output_dir}/validation_dataset.parquet', index=False)
print(f"Saved validation dataset: {len(validation_dataset)} rows")
print(f" - {output_dir}/validation_dataset.parquet")
print(f" Columns: {validation_dataset.columns.tolist()}")

# %%
# Print file sizes
import os
print("OUTPUT FILES")

for filename in os.listdir(output_dir):
    filepath = os.path.join(output_dir, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"{filename}: {size_mb:.2f} MB")

print("\nTraining data preparation complete")

# %%
# verification Cell 1: Prefix Check
# verify CV prefix standardization
sample_cv = training_dataset.iloc[0]['anchor_text']
print(f"Sample CV text (first 200 chars): {sample_cv[:200]}")
assert sample_cv.startswith("query: "), "ERROR: CV text should start with lowercase 'query: '"
print("✓ CV prefix check PASSED")

# verify job prefix
sample_job = training_dataset.iloc[0]['positive_text']
print(f"Sample job text (first 200 chars): {sample_job[:200]}")
assert sample_job.startswith("passage: "), "ERROR: Job text should start with 'passage: '"
print("✓ Job prefix check PASSED")

# %%
#verification Cell 2: ID Format Check
print("VERIFICATION 2: ID FORMAT CHECK")

import re

# check CV IDs (should be A followed by number)
cv_id_pattern = r'^A\d+$'
invalid_cv_ids = training_pairs[~training_pairs['anchor'].str.match(cv_id_pattern)]
print(f"Invalid CV IDs: {len(invalid_cv_ids)}")
assert len(invalid_cv_ids) == 0, f"Found invalid CV IDs: {invalid_cv_ids['anchor'].tolist()[:5]}"
print("✓ CV ID format check PASSED")

# check Job IDs (should be B followed by number)
job_id_pattern = r'^B\d+$'
invalid_job_ids = training_pairs[~training_pairs['match'].str.match(job_id_pattern)]
print(f"Invalid Job IDs: {len(invalid_job_ids)}")
assert len(invalid_job_ids) == 0, f"Found invalid Job IDs: {invalid_job_ids['match'].tolist()[:5]}"
print("✓ Job ID format check PASSED")

# %%
# verification Cell 3: Split Integrity
# verify no overlap between train/val/test
train_cvs = set(training_pairs['anchor'].values)
val_cvs = set(validation_pairs['anchor'].values)
test_cvs = set(pd.read_parquet(os.path.join(CV_SPLITS_PATH, 'test_set_cv_ids.parquet'))['anchor'].values)

train_val_overlap = train_cvs & val_cvs
train_test_overlap = train_cvs & test_cvs
val_test_overlap = val_cvs & test_cvs

print(f"Train set size: {len(train_cvs)}")
print(f"Val set size: {len(val_cvs)}")
print(f"Test set size: {len(test_cvs)}")
print()
print(f"Train-Val overlap: {len(train_val_overlap)}")
print(f"Train-Test overlap: {len(train_test_overlap)}")
print(f"Val-Test overlap: {len(val_test_overlap)}")

assert len(train_val_overlap) == 0, "ERROR: Train and validation sets overlap!"
assert len(train_test_overlap) == 0, "ERROR: Train and test sets overlap!"
assert len(val_test_overlap) == 0, "ERROR: Validation and test sets overlap!"
print("✓ Split integrity check PASSED")

# %%
# verification Cell 4: Score Statistics

# show cross-encoder score distribution for best matches
print("Training pairs score stats:")
print(f" Min: {training_pairs['score'].min():.4f}")
print(f" Max: {training_pairs['score'].max():.4f}")
print(f" Mean: {training_pairs['score'].mean():.4f}")
print(f" Median: {training_pairs['score'].median():.4f}")
print()

print("Validation pairs score stats:")
print(f" Min: {validation_pairs['score'].min():.4f}")
print(f" Max: {validation_pairs['score'].max():.4f}")
print(f" Mean: {validation_pairs['score'].mean():.4f}")
print(f" Median: {validation_pairs['score'].median():.4f}")

print("\nHigh scores indicate good CV-job matches")
print("✓ Score statistics computed")


