import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, concat, lit, length, avg,
    min as spark_min, max as spark_max, coalesce
)
from pyspark.sql.types import StringType

print('Imports loaded')

# setup paths
cwd = os.getcwd()

if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

SKILLS_AGG_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'skills_aggregated')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'cv_query_text.parquet')

print(f'Project root: {PROJECT_ROOT}')
print(f'Input: {SKILLS_AGG_DIR}')
print(f'Output: {OUTPUT_PATH}')

# create spark session
print('Creating Spark session')

spark = SparkSession.builder \
    .appName('CVEmbeddingStrings') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '8') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# load script 03 output
print('Loading skills_aggregated data')
df = spark.read.parquet(SKILLS_AGG_DIR)
total_count = df.count()
print(f'Loaded {total_count} records')
print(f'Columns: {df.columns}')
df.show(3, truncate=80)

# define embedding string builder function
def build_cv_embedding_text(title, years, level, skills, education_summary, company):
    # start with title
    text = f'I am a {title}'

    # add years if available and > 0
    if years is not None and years > 0:
        text += f' with {years:.0f} years of experience'

    # add level if available
    if level is not None and level != '':
        text += f', {level}'

    text += '.'

    # add skills, limit to first 15
    if skills is not None and skills != '':
        skill_list = skills.split(', ')[:15]
        text += f' My skills include: {", ".join(skill_list)}.'

    # add education summary if available
    if education_summary is not None and education_summary != '':
        text += f' I studied {education_summary}.'

    # add company if not the fallback value
    if company is not None and company != 'various organizations':
        text += f' I worked as {title} at {company}.'

    return text

# register as UDF
build_cv_embedding_udf = udf(build_cv_embedding_text, StringType())

# test UDF with sample inputs
print('Testing embedding string builder:')

test_cases = [
    ('Software Engineer', 5.0, 'mid-level', 'Python, Java, SQL', "Bachelor's in Computer Science from MIT", 'Google'),
    ('Professional', None, 'entry level', 'Excel, Word', None, 'various organizations'),
    ('Data Analyst', 0.0, None, None, "Master's in Data Science", 'Acme Corp'),
    ('DevOps Engineer', 12.0, 'expert level', 'AWS, Docker, Kubernetes, Terraform, Jenkins, Ansible, Linux, Git, CI/CD, Monitoring, Python, Bash, Helm, Prometheus, Grafana, CloudWatch', "Bachelor's", 'Amazon'),
    ('Manager', 8.0, 'senior level', 'Project Management, Agile', "MBA from Harvard", 'various organizations'),
]

for args in test_cases:
    result = build_cv_embedding_text(*args)
    print(f'  Input: title={args[0]}, years={args[1]}, level={args[2]}')
    print(f'  Output: {result}')
    print()

# apply UDF to dataframe
print('Building embedding strings')
df = df.withColumn('embedding_text',
    build_cv_embedding_udf(
        col('title'),
        col('total_years'),
        col('level'),
        col('skills'),
        col('education_summary'),
        col('company')
    )
)

# add query prefix (lowercase, with space after colon)
df = df.withColumn('text', concat(lit('query: '), col('embedding_text')))

print('Sample embedding strings:')
df.select('id', 'text').show(5, truncate=200)

# validate output
print('Validating output')

null_count = df.filter(col('text').isNull()).count()
print(f'Null text values: {null_count}')

not_query = df.filter(~col('text').startswith('query: ')).count()
print(f'Text values not starting with "query: ": {not_query}')

# length statistics
length_stats = df.withColumn('text_len', length(col('text'))).agg(
    avg('text_len').alias('avg_len'),
    spark_min('text_len').alias('min_len'),
    spark_max('text_len').alias('max_len')
).collect()[0]

print(f'Text length - avg: {length_stats["avg_len"]:.0f}, min: {length_stats["min_len"]}, max: {length_stats["max_len"]}')

# samples by seniority level
print('Samples by seniority level:')
for level in ['entry level', 'mid-level', 'senior level', 'expert level']:
    sample = df.filter(col('level') == level).select('text').limit(1).collect()
    if sample:
        print(f'  [{level}] {sample[0]["text"][:180]}')

# education inclusion rate
edu_count = df.filter(col('text').contains('I studied')).count()
print(f'Embedding strings with education: {edu_count} / {total_count}')

# select final columns and save
print('Saving output')
output_df = df.select('id', 'text')
output_df.coalesce(1).write.mode('overwrite').parquet(OUTPUT_PATH)
print(f'Saved to {OUTPUT_PATH}')

# verify output
print('Verifying saved output')
verify_df = spark.read.parquet(OUTPUT_PATH)
print(f'Records: {verify_df.count()}')
print(f'Columns: {verify_df.columns}')
verify_df.show(5, truncate=200)

print(f'Total records processed: {total_count}')
spark.stop()
print('Done')
