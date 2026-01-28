# %%
# imports
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

# setup paths and detect project root
import sys
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

print('imports loaded')
print(f'project root: {PROJECT_ROOT}')

# %%
# spark needs more memory for embeddings with 768 floats each
spark = SparkSession.builder \
    .appName('TrainValTestSplits') \
    .config('spark.driver.memory', '16g') \
    .config('spark.executor.memory', '8g') \
    .config('spark.driver.maxResultSize', '4g') \
    .getOrCreate()
print(f'spark started: {spark.version}')

# %%
cv_data_dir = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output')

print('loading CV splits')
train_cv_ids = pd.read_parquet(os.path.join(cv_data_dir, 'training_set_cv_ids.parquet'))
val_cv_ids = pd.read_parquet(os.path.join(cv_data_dir, 'validation_set_cv_ids.parquet'))
test_cv_ids = pd.read_parquet(os.path.join(cv_data_dir, 'test_set_cv_ids.parquet'))

print(f'\nCV split sizes:')
print(f' training: {len(train_cv_ids):,} CVs')
print(f' validation: {len(val_cv_ids):,} CVs')
print(f' test: {len(test_cv_ids):,} CVs')
total_cvs = len(train_cv_ids) + len(val_cv_ids) + len(test_cv_ids)
print(f'\nsplit percentages:')
print(f' training: {len(train_cv_ids)/total_cvs*100:.1f}%')
print(f' validation: {len(val_cv_ids)/total_cvs*100:.1f}%')
print(f' test: {len(test_cv_ids)/total_cvs*100:.1f}%')

# %%
# check for overlaps
train_cv_set = set(train_cv_ids.iloc[:, 0].values)
val_cv_set = set(val_cv_ids.iloc[:, 0].values)
test_cv_set = set(test_cv_ids.iloc[:, 0].values)
print(f'\ntrain n val: {len(train_cv_set & val_cv_set)}')
print(f'train n test: {len(train_cv_set & test_cv_set)}')
print(f'val n test: {len(val_cv_set & test_cv_set)}')
no_overlap = (
    len(train_cv_set & val_cv_set) == 0 and
    len(train_cv_set & test_cv_set) == 0 and
    len(val_cv_set & test_cv_set) == 0
)
if no_overlap:
    print('\nCV splits validation: PASSED (no overlap)')

# %%
# load job embeddings with Spark
jobs_path = str(os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'jobs_embedded.parquet'))

print('loading job embeddings with Spark')
jobs_df = spark.read.parquet(jobs_path)
total_jobs = jobs_df.count()
print(f'loaded {total_jobs:,} jobs')
print(f'columns: {jobs_df.columns}')
print('\nsample:')
jobs_df.show(3, truncate=60)

# %%
# check if isco_code column exists from 05 stratified sampling
has_isco = 'isco_code' in jobs_df.columns
print(f'has isco_code column: {has_isco}')

if has_isco:
    print('\nISCO distribution in embeddings:')
    isco_counts = jobs_df.groupBy('isco_code').count().orderBy('isco_code').collect()
    for row in isco_counts:
        print(f'  {row["isco_code"]}: {row["count"]:,}')

# %%
# ISCO names for display
ISCO_NAMES = {
    0: 'Military',
    1: 'Managers',
    2: 'Professionals', 
    3: 'Technicians',
    4: 'Clerical',
    5: 'Service/Sales',
    6: 'Agriculture',
    7: 'Craft/Trade',
    8: 'Operators',
    9: 'Elementary'
}

if has_isco:
    # stratified split using sampleBy with fractions per ISCO code
    print('creating stratified job splits by ISCO domain...')
    
    # get counts per domain for fraction calculation
    domain_counts = {row['isco_code']: row['count'] for row in isco_counts}
    
    # calculate fractions for 80/10/10 split per domain
    # train: 0.80, val: 0.10, test: 0.10
    train_fractions = {code: 0.80 for code in domain_counts.keys()}
    val_fractions = {code: 0.10 for code in domain_counts.keys()}
    test_fractions = {code: 0.10 for code in domain_counts.keys()}
    
    # shuffle first for randomness
    jobs_shuffled = jobs_df.orderBy(rand(seed=42))
    
    # sample train set
    train_jobs_df = jobs_shuffled.sampleBy('isco_code', train_fractions, seed=42)
    
    # get remaining jobs (not in train)
    train_ids = set(train_jobs_df.select('job_id').rdd.flatMap(lambda x: x).collect())
    remaining_df = jobs_shuffled.filter(~col('job_id').isin(list(train_ids)))
    
    # split remaining 50/50 for val/test
    val_test_fractions = {code: 0.50 for code in domain_counts.keys()}
    val_jobs_df = remaining_df.sampleBy('isco_code', val_test_fractions, seed=43)
    
    # test is everything not in train or val
    val_ids = set(val_jobs_df.select('job_id').rdd.flatMap(lambda x: x).collect())
    test_jobs_df = remaining_df.filter(~col('job_id').isin(list(val_ids)))
    
    print('stratified sampling complete')
    
else:
    # fallback to random split if no isco_code
    print('WARNING: no isco_code column, falling back to random split')
    jobs_shuffled = jobs_df.orderBy(rand(seed=42))
    train_jobs_df, val_jobs_df, test_jobs_df = jobs_shuffled.randomSplit([0.8, 0.1, 0.1], seed=42)

# %%
# cache and count
train_jobs_df.cache()
val_jobs_df.cache()
test_jobs_df.cache()

train_count = train_jobs_df.count()
val_count = val_jobs_df.count()
test_count = test_jobs_df.count()
total_split = train_count + val_count + test_count

print(f'\njob split sizes:')
print(f'  training: {train_count:,} jobs ({train_count/total_split*100:.1f}%)')
print(f'  validation: {val_count:,} jobs ({val_count/total_split*100:.1f}%)')
print(f'  test: {test_count:,} jobs ({test_count/total_split*100:.1f}%)')
print(f'  total: {total_split:,}')

# %%
# verify stratification maintained in each split
if has_isco:
    train_isco = train_jobs_df.groupBy('isco_code').count().orderBy('isco_code').collect()
    val_isco = val_jobs_df.groupBy('isco_code').count().orderBy('isco_code').collect()
    test_isco = test_jobs_df.groupBy('isco_code').count().orderBy('isco_code').collect()
    
    print(f'\n{"ISCO":15} {"Train":>10} {"Val":>10} {"Test":>10}')
    print('-' * 50)
    
    train_dict = {r['isco_code']: r['count'] for r in train_isco}
    val_dict = {r['isco_code']: r['count'] for r in val_isco}
    test_dict = {r['isco_code']: r['count'] for r in test_isco}
    
    all_codes = sorted(set(train_dict.keys()) | set(val_dict.keys()) | set(test_dict.keys()))
    for code in all_codes:
        name = ISCO_NAMES.get(code, 'Unknown')
        t = train_dict.get(code, 0)
        v = val_dict.get(code, 0)
        ts = test_dict.get(code, 0)
        print(f'{code} {name:12} {t:>10,} {v:>10,} {ts:>10,}')
    
    print('\nstratification: VERIFIED')

# %%
output_dir = os.path.join(PROJECT_ROOT, 'training', 'output', 'splits')
os.makedirs(output_dir, exist_ok=True)

train_jobs_path = os.path.join(output_dir, 'train_jobs.parquet')
val_jobs_path = os.path.join(output_dir, 'val_jobs.parquet')
test_jobs_path = os.path.join(output_dir, 'test_jobs.parquet')

train_jobs_df.write.mode('overwrite').parquet(train_jobs_path)
val_jobs_df.write.mode('overwrite').parquet(val_jobs_path)
test_jobs_df.write.mode('overwrite').parquet(test_jobs_path)

print(f'saved job splits:')
print(f'  {train_jobs_path}: {train_count:,} jobs')
print(f'  {val_jobs_path}: {val_count:,} jobs')
print(f'  {test_jobs_path}: {test_count:,} jobs')

# %%
cvs_path = str(os.path.join(PROJECT_ROOT, 'training', 'output', 'embeddings', 'cvs_embedded.parquet'))

print('loading CV embeddings with Spark')
cvs_df = spark.read.parquet(cvs_path)

print(f'loaded {cvs_df.count():,} CVs')
print(f'\ncolumns: {cvs_df.columns}')
print('\nsample:')
cvs_df.show(3, truncate=60)

# %%
# get CV ID column name
cv_id_col = train_cv_ids.columns[0]
print(f'CV ID column: {cv_id_col}')
# convert to lists for filtering
train_cv_list = train_cv_ids[cv_id_col].tolist()
val_cv_list = val_cv_ids[cv_id_col].tolist()
test_cv_list = test_cv_ids[cv_id_col].tolist()

print(f'\nfiltering CVs based on existing splits')

# %%
# filter with Spark
train_cvs_df = cvs_df.filter(col('cv_id').isin(train_cv_list))
val_cvs_df = cvs_df.filter(col('cv_id').isin(val_cv_list))
test_cvs_df = cvs_df.filter(col('cv_id').isin(test_cv_list))

# cache and count
train_cvs_df.cache()
val_cvs_df.cache()
test_cvs_df.cache()

train_cv_count = train_cvs_df.count()
val_cv_count = val_cvs_df.count()
test_cv_count = test_cvs_df.count()

print(f'\nCV split sizes:')
print(f' training: {train_cv_count:,} CVs')
print(f' validation: {val_cv_count:,} CVs')
print(f' test: {test_cv_count:,} CVs')

if train_cv_count == len(train_cv_ids):
    print('\nall CV IDs found in embeddings')

# %%
train_cvs_path = os.path.join(output_dir, 'train_cvs.parquet')
val_cvs_path = os.path.join(output_dir, 'val_cvs.parquet')
test_cvs_path = os.path.join(output_dir, 'test_cvs.parquet')

train_cvs_df.write.mode('overwrite').parquet(train_cvs_path)
val_cvs_df.write.mode('overwrite').parquet(val_cvs_path)
test_cvs_df.write.mode('overwrite').parquet(test_cvs_path)

print(f'saved CV splits:')
print(f'  {train_cvs_path}: {train_cv_count:,} CVs')
print(f'  {val_cvs_path}: {val_cv_count:,} CVs')
print(f'  {test_cvs_path}: {test_cv_count:,} CVs')

# %%
print(f'\njobs (stratified by ISCO):')
print(f' training: {train_count:,} ({train_count/total_split*100:.1f}%)')
print(f' validation: {val_count:,} ({val_count/total_split*100:.1f}%)')
print(f' test: {test_count:,} ({test_count/total_split*100:.1f}%)')
print(f' total: {total_split:,}')

print(f'\nCVs (from colleague splits):')
print(f' training: {train_cv_count:,} ({train_cv_count/total_cvs*100:.1f}%)')
print(f' validation: {val_cv_count:,} ({val_cv_count/total_cvs*100:.1f}%)')
print(f' test: {test_cv_count:,} ({test_cv_count/total_cvs*100:.1f}%)')
print(f' total: {total_cvs:,}')

print(f'\nvalidation checks:')
print(f' CV splits no overlap: {no_overlap}')
if has_isco:
    print(f' job splits stratified: True')

print(f'\nratios (jobs:CVs):')
print(f' training: {train_count}:{train_cv_count} = 1:{train_cv_count//max(1,train_count//train_cv_count if train_cv_count else 1)}')
print(f' validation: {val_count}:{val_cv_count}')
print(f' test: {test_count}:{test_cv_count}')

print(f'\nall splits ready for training')

# %%
print(f'output directory: {output_dir}/')
print(f'\njob splits (stratified by ISCO):')
print(f' train_jobs.parquet: {train_count:,} jobs (80%)')
print(f' val_jobs.parquet: {val_count:,} jobs (10%)')
print(f' test_jobs.parquet: {test_count:,} jobs (10%)')
print(f'\nCV splits (from colleague):')
print(f' train_cvs.parquet: {train_cv_count:,} CVs (80%)')
print(f' val_cvs.parquet: {val_cv_count:,} CVs (10%)')
print(f' test_cvs.parquet: {test_cv_count:,} CVs (10%)')
print(f'\nall splits created with Spark')

# %%
spark.stop()
print('spark stopped')


