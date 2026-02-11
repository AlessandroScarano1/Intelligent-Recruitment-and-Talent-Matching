# %%
# imports
import os
import sys
import time
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, concat_ws, lit
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

import spacy
from spacy.matcher import PhraseMatcher

from confluent_kafka import Producer

# setup paths - detect project root
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

# kafka config
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka-broker:29092')

print('imports loaded')
print(f'spaCy version: {spacy.__version__}')
print(f'project root: {PROJECT_ROOT}')
print(f'kafka broker: {KAFKA_BROKER}')

# %%
# start spark session
spark = SparkSession.builder \
    .appName('NLPExtractionWithSpaCy') \
    .config('spark.driver.memory', '4g') \
    .config('spark.executor.memory', '4g') \
    .getOrCreate()

print(f'spark started: {spark.version}')
print(f'spark app name: {spark.sparkContext.appName}')

# %%
# load jobs from processed linkedin output
linkedin_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'processed', 'linkedin', 'linkedin_jobs_with_skills')
jobs_df = spark.read.parquet(linkedin_path)

total_jobs = jobs_df.count()
print(f'loaded {total_jobs:,} jobs')

# show source distribution
print('\nby source:')
jobs_df.groupBy('source').count().show()

# %%
# filter to indeed/glassdoor jobs needing extraction
indeed_glassdoor_df = jobs_df.filter(col('source').isin(['indeed', 'glassdoor']))

ig_count = indeed_glassdoor_df.count()
print(f'indeed/glassdoor jobs: {ig_count:,}')

# %%
# load skill dictionary
skill_dict_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'skill_dictionary', 'all_skills')
skill_df = spark.read.parquet(skill_dict_path)

# !! using min 100 occurrences to filter out noise
# the raw dictionary has 3.3M entries including stopwords and generic words
skill_df_filtered = skill_df.filter(col('count') >= 100)

skill_count = skill_df_filtered.count()
print(f'loaded {skill_count:,} skills (min 100 occurrences)')

# define stopwords to filter out
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
             'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
             'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
             'might', 'must', 'shall', 'can', 'need', 'our', 'we', 'you', 'your', 'they',
             'their', 'its', 'it', 'that', 'this', 'which', 'what', 'who', 'whom', 'any',
             'all', 'some', 'more', 'most', 'other', 'each', 'few', 'many', 'such', 'no',
             'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'}

# generic resume words, appear in almost every job description but aren't skills
GENERIC_RESUME_WORDS = {'experience', 'years', 'strong', 'team', 'work', 'working', 'position',
                        'ability', 'able', 'excellent', 'good', 'great', 'skills', 'knowledge',
                        'understanding', 'familiarity', 'proficiency', 'expertise', 'demonstrated',
                        'proven', 'equivalent', 'gain', 'various', 'different', 'multiple', 'new',
                        'within', 'across', 'using', 'including', 'related', 'required', 'preferred',
                        'minimum', 'ideal', 'desirable', 'essential', 'degree', 'bachelor', 'master',
                        'diploma', 'intern', 'internship', 'job', 'role', 'opportunity', 'employment',
                        'company', 'organization', 'business', 'environment', 'requirements', 
                        'responsibilities', 'status', 'seeking', 'level', 'current', 'support'}

# ultra-generic single words that are NOT skills when alone
#keep multi-word versions like 'data analysis' but filter single 'data'
# also filter job titles when they appear as single words
ULTRA_GENERIC_SINGLE = {'data', 'information', 'process', 'systems', 'solutions',
                        'pull', 'building', 'tools', 'supporting', 'identification',
                        'preparation', 'methodology', 'investigations', 'engagement',
                        'developer', 'engineer', 'analyst', 'scientist', 'manager',
                        'coordinator', 'director', 'supervisor', 'administrator',
                        'assistant', 'associate', 'officer', 'executive', 'remote',
                        'exposure', 'issues', 'structure', 'structures', 'oversight',
                        'computer', 'lead', 'maintenance'}

def is_quality_skill(skill):
    #filter out stopwords, generic resume words, and ultra-generic single words
    skill_lower = skill.lower().strip()
    
    # length check
    if len(skill_lower) < 3:
        return False
    
    # stopwords
    if skill_lower in STOPWORDS:
        return False
    
    # generic resume words  
    if skill_lower in GENERIC_RESUME_WORDS:
        return False
    
    # for single words only, filter ultra-generic terms
    # keep multi-word phrases like 'data analysis', 'project management', 'remote work'
    if ' ' not in skill_lower and skill_lower in ULTRA_GENERIC_SINGLE:
        return False
        
    return True

# collect to driver and filter
print('\ncollecting skills to driver')
raw_skills = [row['skill'] for row in skill_df_filtered.select('skill').collect()]
skills_list = [s.lower() for s in raw_skills if is_quality_skill(s)]

print(f'after filtering stopwords/generic words: {len(skills_list):,} skills')
print(f'removed {len(raw_skills) - len(skills_list):,} noisy entries')

# %%
# initialize spacy blank model (just tokenizer, no heavy NLP)
nlp = spacy.blank('en')
print(f'spacy model loaded: {nlp.lang}')

# create PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')  # case-insensitive matching

# convert skills to patterns
print('building phrase patterns')
patterns = [nlp.make_doc(skill) for skill in skills_list]
matcher.add('SKILLS', patterns)

print(f'added {len(patterns):,} patterns to matcher')
print('phrasematcher ready')

# %%
# skill extraction using PhraseMatcher
def extract_skills_spacy(text):
    #extract skills using spacy PhraseMatcher
    # FAST trie-based lookup, not regex loops
    
    if not text or not isinstance(text, str):
        return []
    
    doc = nlp(text.lower())
    matches = matcher(doc)
    
    # extract unique matched skills
    found_skills = list(set([doc[start:end].text for _, start, end in matches]))
    
    return found_skills

print('spacy skill extractor defined')

# %%
# test extractor with multiple examples
test_texts = [
    'Senior Python Developer needed. 5+ years with Django, AWS, PostgreSQL. Remote.',
    'Data Scientist with Machine Learning, TensorFlow, PyTorch experience.',
    'Full Stack Engineer: React, Node.js, MongoDB, Docker, Kubernetes.'
]

print('testing skill extraction:')
print('-' * 60)
for text in test_texts:
    skills = extract_skills_spacy(text)
    print(f'text: {text}')
    print(f'skills: {skills}')
    print('-' * 60)

# %%
# broadcast skills and matcher to executors
# this makes them available on all worker nodes
broadcast_skills = spark.sparkContext.broadcast(skills_list)

print(f'broadcasted {len(skills_list):,} skills to executors')

# %%
# create UDF for skill extraction
# this allows spark to call our python function in distributed way
skill_extraction_udf = udf(extract_skills_spacy, ArrayType(StringType()))

print('spark udf created for skill extraction')

# %%
# apply skill extraction to all indeed/glassdoor jobs
print(f'extracting skills from {ig_count:,} jobs using spark distributed processing')

start_time = time.time()

# add skills column using UDF
extracted_df = indeed_glassdoor_df.withColumn(
    'extracted_skills',
    skill_extraction_udf(col('description'))
)

# convert array to comma-separated string
extracted_df = extracted_df.withColumn(
    'skills',
    concat_ws(', ', col('extracted_skills'))
)

# trigger execution by counting
result_count = extracted_df.count()

elapsed = time.time() - start_time
print(f'extraction complete, processed {result_count:,} jobs in {elapsed:.1f}s')

# %%
# show sample results
print('sample extractions:')
extracted_df.select('job_title', 'company', 'skills').show(10, truncate=60)

# %%
# extraction stats
from pyspark.sql.functions import size, when

# count skills extracted
stats_df = extracted_df.withColumn(
    'skill_count',
    size(col('extracted_skills'))
).withColumn(
    'has_skills',
    when(col('skill_count') > 0, 1).otherwise(0)
)

has_skills = stats_df.filter(col('has_skills') == 1).count()
total = stats_df.count()

print(f'extraction statistics:')
print(f' total jobs: {total:,}')
print(f' with skills: {has_skills:,} ({has_skills/total*100:.1f}%)')
print(f' without skills: {total - has_skills:,} ({(total-has_skills)/total*100:.1f}%)')

# %%
# kafka configuration
kafka_config = {
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'nlp-extraction-producer'
}

producer = Producer(kafka_config)
print('kafka producer initialized')

# %%
# publish to kafka
print('publishing to kafka topic: extracted_jobs')

# collect to driver (small dataset - 771 jobs)
results_pd = extracted_df.toPandas()

published_count = 0
for idx, row in results_pd.iterrows():
    # create message
    msg = {
        'id': row['id'],
        'job_title': row['job_title'],
        'company': row['company'],
        'source': row['source'],
        'skills': row['skills'],
        'extracted_skill_count': len(row['extracted_skills']),
        'timestamp': time.time()
    }
    
    # publish to kafka
    producer.produce(
        'extracted_jobs',
        key=row['id'].encode('utf-8'),
        value=json.dumps(msg).encode('utf-8')
    )
    
    published_count += 1
    
    # flush every 100 messages
    if published_count % 100 == 0:
        producer.flush()
        print(f'  published {published_count}/{len(results_pd)}...')

# final flush
producer.flush()
print(f'published {published_count:,} messages to kafka, kafka publishing complete')

# %%
# save extracted jobs to parquet
output_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'processed', 'indeed_glassdoor', 'indeed_glassdoor_extracted')

print(f'saving to {output_path}')
extracted_df.write.mode('overwrite').parquet(output_path)

print(f'saved {result_count:,} jobs to parquet')

# %%
print('NLP EXTRACTION COMPLETE')
print(f'processed: {result_count:,} indeed/glassdoor jobs')
print(f'extraction time: {elapsed:.1f}s')
print(f'technology used:')
print(f' - spark for distributed processing')
print(f' - spacy PhraseMatcher for fast skill matching')
print(f' - kafka for publishing results')
print(f'\noutput:')
print(f' - parquet: {output_path}')
print(f' - kafka: extracted_jobs topic')

# %%
# cleanup
spark.stop()
print('spark stopped')


