# %%
# imports
import os
import re
import pandas as pd
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, monotonically_increasing_id, length, udf, when, lower
from pyspark.sql.types import IntegerType, StringType

from confluent_kafka import Producer

# setup paths - detect project root
import sys
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

# kafka config
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka-broker:29092')

# sampling config
TARGET_SAMPLE_SIZE = 150000  # 150K jobs for balanced training with 7K CVs
MIN_PER_DOMAIN = 1000        # minimum samples per ISCO domain

print('imports loaded')
print(f'project root: {PROJECT_ROOT}')
print(f'kafka broker: {KAFKA_BROKER}')
print(f'target sample size: {TARGET_SAMPLE_SIZE:,}')

# %%
spark = SparkSession.builder \
    .appName('FinalEmbeddingOutput') \
    .config('spark.driver.memory', '4g') \
    .getOrCreate()

print(f'spark started: {spark.version}')

# %%
input_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'unified_job_postings', 'unified_jobs.parquet')
jobs_df = spark.read.parquet(input_path)
total_jobs = jobs_df.count()

print(f'loaded {total_jobs:,} jobs')
print(f'columns: {jobs_df.columns}')
print('sample:')
jobs_df.select('id', 'job_title', 'embedding_text').show(3, truncate=60)

# %%
# Fast ID assignment using monotonically_increasing_id()---> ~100x faster than zipWithIndex because it doesn't require shuffle
# IDs will have gaps (B0, B1, B8589934592...) should not be a problem for embeddings

def assign_sequential_ids_fast(df, prefix='B'):
    if 'id' in df.columns:
        df = df.withColumnRenamed('id', 'original_uuid')
    
    # monotonically_increasing_id, distributed and fast
    df = df.withColumn(
        'job_id',
        concat(lit(prefix), monotonically_increasing_id().cast('string'))
    )
    
    return df

print('fast ID assignment function defined (uses monotonically_increasing_id)')

# %%
print('assigning B IDs (fast method)')
jobs_with_ids = assign_sequential_ids_fast(jobs_df, prefix='B')

# count is now fast since we're not using RDD
job_count = jobs_with_ids.count()
print(f'assigned IDs to {job_count:,} jobs')

print('sample with IDs:')
jobs_with_ids.select('job_id', 'original_uuid', 'job_title').show(10, truncate=60)

# %%
# ISCO-08 classification patterns based on International Standard Classification of Occupations
#ref: https://www.ilo.org/public/english/bureau/stat/isco/isco08/

ISCO_PATTERNS = {
    1: ['manager', 'director', 'chief', 'head of', 'president', 'vp ', 'vice president', 
        'ceo', 'cfo', 'cto', 'coo', 'executive', 'supervisor', 'lead', 'principal'],
    2: ['engineer', 'developer', 'scientist', 'doctor', 'nurse', 'teacher', 'professor',
        'architect', 'designer', 'analyst', 'consultant', 'specialist', 'attorney', 
        'lawyer', 'accountant', 'pharmacist', 'therapist', 'physician'],
    3: ['technician', 'associate', 'assistant', 'coordinator', 'representative',
        'administrator', 'officer', 'agent', 'inspector', 'controller'],
    4: ['clerk', 'secretary', 'receptionist', 'cashier', 'teller', 'bookkeeper',
        'data entry', 'typist', 'filing'],
    5: ['sales', 'retail', 'customer service', 'waiter', 'waitress', 'bartender',
        'cook', 'chef', 'server', 'barista', 'store', 'shop'],
    6: ['farmer', 'agricultural', 'fisherman', 'forestry', 'gardener', 'rancher'],
    7: ['electrician', 'mechanic', 'plumber', 'carpenter', 'welder', 'machinist',
        'installer', 'repair', 'maintenance', 'hvac'],
    8: ['driver', 'operator', 'machine', 'assembler', 'production', 'manufacturing',
        'warehouse', 'forklift', 'truck'],
    9: ['cleaner', 'janitor', 'helper', 'laborer', 'packer', 'loader', 'dishwasher',
        'housekeeper', 'security guard'],
    0: ['military', 'army', 'navy', 'marine', 'air force', 'soldier']
}

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

def classify_isco(job_title):
    #classify job title into ISCO-08 major group
    if not job_title:
        return 2  # default to Professionals
    
    title_lower = job_title.lower()
    
    # check each ISCO group's patterns
    for isco_code, patterns in ISCO_PATTERNS.items():
        for pattern in patterns:
            if pattern in title_lower:
                return isco_code
    
    # default to Professionals (most common in job datasets)
    return 2

# register as Spark UDF
classify_isco_udf = udf(classify_isco, IntegerType())

print('ISCO classification function defined, 10 ISCO major groups configured')

# %%
# apply ISCO classification to all jobs
print('classifying jobs by ISCO domain')

jobs_with_isco = jobs_with_ids.withColumn(
    'isco_code',
    classify_isco_udf(col('job_title'))
)

# show distribution
print('ISCO distribution (before sampling):')
isco_counts = jobs_with_isco.groupBy('isco_code').count().orderBy('isco_code').collect()

total = sum(row['count'] for row in isco_counts)
for row in isco_counts:
    code = row['isco_code']
    count = row['count']
    pct = count / total * 100
    name = ISCO_NAMES.get(code, 'Unknown')
    print(f'  {code} - {name:15s}: {count:>10,} ({pct:5.1f}%)')

print(f'\ntotal jobs: {total:,}')

# %%
# calculate stratified sampling fractions, proportional sampling with minimum per domain

print(f'calculating stratified sampling fractions, target: {TARGET_SAMPLE_SIZE:,} jobs, minimum per domain: {MIN_PER_DOMAIN:,}')

# get counts per domain
domain_counts = {row['isco_code']: row['count'] for row in isco_counts}

# calculate how many to sample from each domain, proportional to size, but ensure minimum
sample_per_domain = {}
remaining_budget = TARGET_SAMPLE_SIZE

# first pass: allocate minimum to small domains
for code, count in domain_counts.items():
    if count < MIN_PER_DOMAIN:
        # take all if domain is smaller than minimum
        sample_per_domain[code] = count
        remaining_budget -= count
    elif count * (TARGET_SAMPLE_SIZE / total) < MIN_PER_DOMAIN:
        # domain is small, give it minimum
        sample_per_domain[code] = MIN_PER_DOMAIN
        remaining_budget -= MIN_PER_DOMAIN

# second pass: proportional allocation for remaining
remaining_domains = {k: v for k, v in domain_counts.items() if k not in sample_per_domain}
remaining_total = sum(remaining_domains.values())

for code, count in remaining_domains.items():
    proportion = count / remaining_total
    sample_per_domain[code] = int(proportion * remaining_budget)

# calculate fractions for Spark sampleBy
fractions = {}
for code, sample_count in sample_per_domain.items():
    domain_total = domain_counts[code]
    fraction = min(1.0, sample_count / domain_total * 1.1)  # 10% buffer for randomness
    fractions[code] = fraction

print('\nsampling fractions per domain:')
for code in sorted(fractions.keys()):
    name = ISCO_NAMES.get(code, 'Unknown')
    frac = fractions[code]
    target = sample_per_domain[code]
    print(f'  {code} - {name:15s}: {frac:.3f} (target ~{target:,})')

# %%
# perform stratified sampling using Spark's sampleBy
print('performing stratified sampling')

sampled_jobs = jobs_with_isco.sampleBy('isco_code', fractions, seed=42)

# cache and count
sampled_jobs.cache()
sampled_count = sampled_jobs.count()

print(f'sampled {sampled_count:,} jobs (target: {TARGET_SAMPLE_SIZE:,}, sampling ratio: {sampled_count/total*100:.1f}% of original')

# show new distribution
print('\nISCO distribution (after sampling):')
sampled_isco = sampled_jobs.groupBy('isco_code').count().orderBy('isco_code').collect()

for row in sampled_isco:
    code = row['isco_code']
    count = row['count']
    pct = count / sampled_count * 100
    name = ISCO_NAMES.get(code, 'Unknown')
    orig = domain_counts.get(code, 0)
    print(f'  {code} - {name:15s}: {count:>8,} ({pct:5.1f}%) [from {orig:,}]')

# %%
# add passage prefix to sampled jobs
sampled_with_prefix = sampled_jobs.withColumn(
    'embedding_text',
    concat(lit('passage: '), col('embedding_text'))
)

print('added passage prefix to sampled jobs')
print('sample:')
sampled_with_prefix.select('job_id', 'isco_code', 'embedding_text').show(5, truncate=70)

# %%
# select final columns for embedding
final_output = sampled_with_prefix.select('job_id', 'embedding_text', 'isco_code')

print('final output columns:')
print(final_output.columns)
print(f'total records: {sampled_count:,}')
print('\nsample:')
final_output.show(10, truncate=70)

# %%
# save final output
output_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'final', 'jobs_to_embed.parquet')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# save with isco_code for later stratified splitting
final_output.coalesce(1).write.mode('overwrite').parquet(output_path)

print(f'saved {sampled_count:,} jobs to: {output_path}')

# %%
# save full UUID to B ID mapping (for all 1.3M jobs, not just sampled)
mapping_df = jobs_with_ids.select('original_uuid', 'job_id')
mapping_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'final', 'uuid_to_bid_mapping.parquet')
mapping_df.coalesce(1).write.mode('overwrite').parquet(mapping_path)

print(f'saved full UUID mapping ({total_jobs:,} jobs) to: {mapping_path}')

# %%
# save ISCO distribution as JSON for reference
isco_dist = {
    'original_total': total,
    'sampled_total': sampled_count,
    'target_size': TARGET_SAMPLE_SIZE,
    'domains': {}
}

for row in sampled_isco:
    code = row['isco_code']
    count = row['count']
    isco_dist['domains'][str(code)] = {
        'name': ISCO_NAMES.get(code, 'Unknown'),
        'original_count': domain_counts.get(code, 0),
        'sampled_count': count,
        'percentage': round(count / sampled_count * 100, 1)
    }

isco_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'final', 'isco_distribution.json')
with open(isco_path, 'w') as f:
    json.dump(isco_dist, f, indent=2)

print(f'saved ISCO distribution to: {isco_path}')

# %%
kafka_config = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'final-embedding-producer'
}

try:
    producer = Producer(kafka_config)
    print('kafka producer initialized')
    kafka_available = True
except Exception as e:
    print(f'kafka not available: {e}')
    print('skipping kafka publishing')
    kafka_available = False

# %%
if kafka_available:
    print('publishing to kafka topic: jobs_to_embed')
    
    # collect to pandas for iteration
    pdf = final_output.select('job_id', 'embedding_text').toPandas()
    
    published = 0
    for idx, row in pdf.iterrows():
        msg = json.dumps({
            'job_id': row['job_id'],
            'embedding_text': row['embedding_text']
        })
        
        producer.produce('jobs_to_embed', value=msg.encode('utf-8'))
        published += 1
        
        if published % 10000 == 0:
            producer.flush()
            print(f' published {published:,} jobs...')
    
    producer.flush()
    print(f'\npublished {published:,} jobs to kafka')
else:
    published = 0
    print('kafka publishing skipped')

# %%
from pyspark.sql.functions import avg, min as spark_min, max as spark_max

length_stats = final_output.agg(
    avg(length('embedding_text')).alias('avg_len'),
    spark_min(length('embedding_text')).alias('min_len'),
    spark_max(length('embedding_text')).alias('max_len')
).collect()[0]

print(f'original jobs: {total_jobs:,}')
print(f'sampled jobs: {sampled_count:,}')
print(f'sampling ratio: {sampled_count/total_jobs*100:.1f}%')
print(f'\nISCO domains: {len(sampled_isco)}')
print(f'\nembedding text lengths:')
print(f' average: {length_stats["avg_len"]:.0f} chars')
print(f' min: {length_stats["min_len"]} chars')
print(f' max: {length_stats["max_len"]} chars')

# %%
print(f'input: {input_path}')
print(f'\nstratified sampling:')
print(f' original: {total_jobs:,} jobs')
print(f' sampled: {sampled_count:,} jobs')
print(f' ratio: 1:{total_jobs//sampled_count}')
print(f'\noutputs:')
print(f' parquet: {output_path}')
print(f' mapping: {mapping_path}')
print(f' isco: {isco_path}')
if kafka_available:
    print(f' kafka: jobs_to_embed topic ({published:,} messages)')
print(f'\nbalance with CVs:')
print(f'  7K CVs : {sampled_count:,} jobs = 1:{sampled_count//7000}')

# %%
spark.stop()
print('spark stopped')


