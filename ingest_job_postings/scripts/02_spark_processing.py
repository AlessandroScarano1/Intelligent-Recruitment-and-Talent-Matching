# %%
# imports
import os
import sys
import time

# pyspark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, explode, split, trim, lit, udf, broadcast, 
    when, lower, regexp_extract
)
from pyspark.sql.types import StringType

print('Imports loaded')

# %%
# setup paths
cwd = os.getcwd()

# detect if we're in notebooks/ or project root
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'raw_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output')
RAW_JOBS_DIR = os.path.join(OUTPUT_DIR, 'raw_job_postings')

print(f'Project root: {PROJECT_ROOT}')
print(f'Raw data: {RAW_DATA_DIR}')
print(f'Output: {OUTPUT_DIR}')

# check data size
import subprocess
try:
    result = subprocess.run(
        ['du', '-sh', RAW_JOBS_DIR], 
        capture_output=True, 
        text=True
    )
    data_size = result.stdout.split()[0]
    print(f'\nData size: {data_size}')
except:
    print('\nCould not determine data size')

# %%
# create spark session
print('Creating Spark session')

spark = SparkSession.builder \
    .appName('JobPostingPipeline') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '16') \
    .config('spark.sql.adaptive.enabled', 'true') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')

print(f'Spark version: {spark.version}')
print('Spark session created')

# %%
# load job postings from parquet
print(f'Loading data from {RAW_JOBS_DIR}')

jobs_df = spark.read.parquet(RAW_JOBS_DIR)
total_count = jobs_df.count()

print(f'\nLoaded {total_count:,} job postings')
print('\nRecords by source:')
jobs_df.groupBy('source').count().show()

# %%
# load ground-truth skills
skills_path = os.path.join(RAW_DATA_DIR, 'job_skills.csv')
print(f'Loading ground-truth skills from {skills_path}')

skills_df = spark.read.csv(skills_path, header=True, inferSchema=True)
skills_count = skills_df.count()

print(f'\nLoaded {skills_count:,} skill records')
print('\nSample:')
skills_df.show(3, truncate=80)

# %%
# separate linkedin from other sources
linkedin_jobs = jobs_df.filter(col('source') == 'linkedin')
other_jobs = jobs_df.filter(col('source') != 'linkedin')

print(f'LinkedIn jobs: {linkedin_jobs.count():,}')
print(f'Other jobs (Indeed/Glassdoor): {other_jobs.count():,}')

# %%
# join linkedin with skills
print('Joining LinkedIn jobs with skills')
start_time = time.time()

linkedin_with_skills = linkedin_jobs.join(
    broadcast(skills_df),
    linkedin_jobs['job_link'] == skills_df['job_link'],
    'left'
).drop(skills_df['job_link']).withColumnRenamed('job_skills', 'skills')

elapsed = time.time() - start_time
print(f'Join completed in {elapsed:.1f}s')

with_skills = linkedin_with_skills.filter(col('skills').isNotNull()).count()
linkedin_count = linkedin_with_skills.count()
pct = 100 * with_skills / linkedin_count if linkedin_count > 0 else 0
print(f'LinkedIn jobs with skills: {with_skills:,} / {linkedin_count:,} ({pct:.1f}%)')

# %%
# add empty skills to other jobs
other_jobs_with_skills = other_jobs.withColumn('skills', lit(None).cast('string'))

# combine all
all_jobs = linkedin_with_skills.union(other_jobs_with_skills)
print(f'\nTotal jobs: {all_jobs.count():,}')

# %%
# extract seniority from job titles using regex
# this gives us 5 levels instead of just 2 from job_level column
# order matters: check intern/junior/senior BEFORE manager (Sr. Manager = senior)

print('Extracting seniority from job titles')

all_jobs = all_jobs.withColumn(
    'seniority',
    when(col('job_title').rlike(r'(?i)\b(intern|internship)\b'), 'intern')
    .when(col('job_title').rlike(r'(?i)\b(jr\.?|junior|entry|graduate|trainee)\b'), 'junior')
    .when(col('job_title').rlike(r'(?i)\b(sr\.?|senior|ii|iii|iv)\b'), 'senior')
    .when(col('job_title').rlike(r'(?i)\b(principal|staff|head of|director|vp|chief|lead)\b'), 'lead/principal')
    .when(col('job_title').rlike(r'(?i)\bassociate\b'), 'junior')
    .otherwise('mid')
)

# register as temp view for SQL queries
all_jobs.createOrReplaceTempView('jobs')

print('\nSeniority distribution (extracted from job titles):')
all_jobs.groupBy('seniority').count().orderBy(col('count').desc()).show()

# %%
# sample with skills
print('\nSample LinkedIn jobs WITH skills:')
all_jobs.filter(
    (col('source') == 'linkedin') & (col('skills').isNotNull())
).select(
    'job_title', 'company', 'skills'
).show(5, truncate=70)

# %%
# build skill dictionary from all skills
print('Building skill dictionary')
start_time = time.time()

skill_counts = all_jobs \
    .filter(col('skills').isNotNull()) \
    .select(explode(split(col('skills'), ',')).alias('skill')) \
    .select(trim(col('skill')).alias('skill')) \
    .filter(col('skill') != '') \
    .groupBy('skill') \
    .agg(count('*').alias('count')) \
    .orderBy(col('count').desc())

skill_counts.cache()
unique_skills = skill_counts.count()
elapsed = time.time() - start_time

print(f'\nFound {unique_skills:,} unique skills in {elapsed:.1f}s')
print('\nTop 25 skills:')
skill_counts.show(25, truncate=False)

# %%
# inline embedding string builder function
# this creates natural language descriptions for embedding models

def build_embedding_string(title, company, location, skills, seniority, job_type):
    # build a natural language string from job posting fields
    # template:
    # "Role of {title} at {company} in {location}. Required skills: {skills}.
    # Experience level: {seniority}. Work type: {job_type}."
    parts = []
    
    # title and company
    if title and company and location:
        parts.append(f'Role of {title} at {company} in {location}.')
    elif title and company:
        parts.append(f'Role of {title} at {company}.')
    elif title:
        parts.append(f'Role of {title}.')
    
    # skills - take first 15 to keep string manageable
    if skills:
        skill_list = [s.strip() for s in skills.split(',')][:15]
        if skill_list:
            parts.append(f'Required skills: {", ".join(skill_list)}.')
    
    # seniority with mapped experience levels
    seniority_map = {
        'intern': 'Internship level',
        'junior': 'Junior level, 1-2 years experience',
        'mid': 'Mid-level, 3-5 years experience',
        'senior': 'Senior level, 5+ years experience',
        'lead/principal': 'Principal level, 8+ years experience'
    }
    if seniority and seniority in seniority_map:
        parts.append(f'Experience level: {seniority_map[seniority]}.')
    
    # job type
    if job_type and job_type not in ['nan', 'None', '']:
        parts.append(f'Work type: {job_type}.')
    
    return ' '.join(parts) if parts else ''

# register as spark UDF
build_embedding_udf = udf(build_embedding_string, StringType())
print('Embedding UDF defined (with seniority mapping)')

# %%
# generate embedding strings for all jobs
print('Building embedding strings')
start_time = time.time()

jobs_with_embeddings = all_jobs.withColumn(
    'embedding_text',
    build_embedding_udf(
        col('job_title'), 
        col('company'), 
        col('job_location'),
        col('skills'), 
        col('seniority'), 
        col('job_type')
    )
)

final_count = jobs_with_embeddings.count()
elapsed = time.time() - start_time

print(f'\nGenerated embeddings for {final_count:,} jobs in {elapsed:.1f}s')

# %%
# show sample embedding strings
print('\nSample embedding strings:')
jobs_with_embeddings.filter(
    col('skills').isNotNull()
).select(
    'job_title', 'embedding_text'
).show(5, truncate=100)

# %%
# define output paths
# LinkedIn-specific output (processed with skills JOIN)
linkedin_path = os.path.join(OUTPUT_DIR, 'processed', 'linkedin')
skill_dict_path = os.path.join(OUTPUT_DIR, 'skill_dictionary')

os.makedirs(linkedin_path, exist_ok=True)
os.makedirs(skill_dict_path, exist_ok=True)

print('Output paths:')
print(f' linkedin: {linkedin_path}')
print(f' skills:  {skill_dict_path}')


# %%
# save LinkedIn jobs with skills and seniority
output_file = os.path.join(linkedin_path, 'linkedin_jobs_with_skills')
print(f'Saving LinkedIn jobs to {output_file}')
jobs_with_embeddings.write.mode('overwrite').parquet(output_file)
print('Saved')

# save skill dictionary
skill_output = os.path.join(skill_dict_path, 'all_skills')
print(f'\nSaving skill dictionary to {skill_output}')
skill_counts.write.mode('overwrite').parquet(skill_output)
print('Saved')


# %%
# show seniority stats
print('Seniority Levels (extracted from job titles):')
spark.sql("""
    SELECT seniority, COUNT(*) as job_count,
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM jobs), 1) as percentage
    FROM jobs
    GROUP BY seniority ORDER BY job_count DESC
""").show(truncate=50)

# %%
# show job level stats
print('Job Levels:')
spark.sql("""
    SELECT job_level, COUNT(*) as job_count
    FROM jobs WHERE job_level IS NOT NULL AND job_level != '' AND job_level != 'nan'
    GROUP BY job_level ORDER BY job_count DESC
""").show(truncate=50)

# %%
# summary
print('SPARK PROCESSING COMPLETE')
print(f'\nTotal jobs: {total_count:,}')
print(f'Jobs with skills: {with_skills:,}')
print(f'Unique skills: {unique_skills:,}')
print(f'\nSeniority extracted from job titles (5 levels):')
print(f' - lead/principal, senior, mid, junior, intern')
print(f'\nOutputs:')
print(f' - {linkedin_path}/linkedin_jobs_with_skills/')
print(f' - {skill_dict_path}/all_skills/')


# %%
# verify output
print('\nVerifying output')
verify_df = spark.read.parquet(os.path.join(linkedin_path, 'linkedin_jobs_with_skills'))
print(f'Verified: {verify_df.count():,} records')
print('\nSample with skills and seniority:')
verify_df.filter(
    col('skills').isNotNull()
).select(
    'job_title', 'seniority', 'skills'
).show(5, truncate=50)


# %%
# stop spark
spark.stop()
print('\nSpark session stopped')


