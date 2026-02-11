import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, when, lit, coalesce, explode,
    to_date, months_between, current_date, round as spark_round,
    sum as spark_sum
)
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType
)

print('Imports loaded')

# setup paths
cwd = os.getcwd()

if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

RAW_CVS_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'raw_cvs')
NORMALIZED_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'normalized')
NORMALIZED_WITH_CV_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'normalized_with_cv')

print(f'Project root: {PROJECT_ROOT}')
print(f'Input: {RAW_CVS_DIR}')
print(f'Output: {NORMALIZED_DIR}')

# create spark session
print('Creating Spark session')

spark = SparkSession.builder \
    .appName('CVNormalization') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '8') \
    .config('spark.sql.legacy.timeParserPolicy', 'LEGACY') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# load phase 2 output
print('Loading raw CV data')
raw_df = spark.read.parquet(RAW_CVS_DIR)
total_count = raw_df.count()
print(f'Loaded {total_count} records')
raw_df.show(3, truncate=80)

# define explicit JSON schema for the fields we need
# education.degree is a struct with level/field, institution is a struct with name
tech_skill_schema = ArrayType(StructType([
    StructField('name', StringType(), True)
]))

cv_schema = StructType([
    StructField('experience', ArrayType(StructType([
        StructField('company', StringType(), True),
        StructField('title', StringType(), True),
        StructField('level', StringType(), True),
        StructField('dates', StructType([
            StructField('start', StringType(), True),
            StructField('end', StringType(), True)
        ]), True),
        StructField('technical_environment', StructType([
            StructField('technologies', ArrayType(StringType()), True)
        ]), True)
    ])), True),
    StructField('skills', StructType([
        StructField('technical', StructType([
            StructField('programming_languages', tech_skill_schema, True),
            StructField('frameworks', tech_skill_schema, True),
            StructField('databases', tech_skill_schema, True),
            StructField('cloud', tech_skill_schema, True),
            StructField('tools', tech_skill_schema, True),
            StructField('web_technologies', tech_skill_schema, True),
            StructField('testing', tech_skill_schema, True),
            StructField('automation', tech_skill_schema, True),
            StructField('project_management', tech_skill_schema, True),
            StructField('networking_skills', tech_skill_schema, True),
            StructField('software', tech_skill_schema, True),
            StructField('software_tools', tech_skill_schema, True),
            StructField('operating_systems', tech_skill_schema, True),
            StructField('other', tech_skill_schema, True),
        ]), True)
    ]), True),
    StructField('education', ArrayType(StructType([
        StructField('degree', StructType([
            StructField('level', StringType(), True),
            StructField('field', StringType(), True)
        ]), True),
        StructField('institution', StructType([
            StructField('name', StringType(), True)
        ]), True)
    ])), True)
])

# parse raw_data JSON
print('Parsing JSON with explicit schema')
df = raw_df.withColumn('cv', from_json(col('raw_data'), cv_schema))

# extract fields from experience[0]
df = df.withColumn('title', col('cv.experience')[0]['title'])
df = df.withColumn('company', col('cv.experience')[0]['company'])
df = df.withColumn('level', col('cv.experience')[0]['level'])

print('Extracted fields from experience[0]:')
df.select('id', 'title', 'company', 'level').show(5, truncate=60)

# clean null placeholders for title, company, level
# replace null, empty string, "Unknown", "Not Provided" with None
for field in ['title', 'company', 'level']:
    df = df.withColumn(field,
        when(
            col(field).isNull() |
            (col(field) == '') |
            (col(field) == 'Unknown') |
            (col(field) == 'Not Provided'),
            lit(None)
        ).otherwise(col(field))
    )

# for company, also replace "Fresher"
df = df.withColumn('company',
    when(col('company') == 'Fresher', lit(None)).otherwise(col('company'))
)

# apply fallback defaults
df = df.withColumn('title', coalesce(col('title'), lit('Professional')))
df = df.withColumn('company', coalesce(col('company'), lit('various organizations')))

print('After cleaning and fallbacks:')
df.select('id', 'title', 'company', 'level').show(5, truncate=60)

# compute total years of experience from date ranges
print('Computing total years of experience')

# explode experience array
exp_df = df.select(
    col('id'),
    explode(col('cv.experience')).alias('exp')
)

# parse start dates - handle both yyyy-MM and yyyy-MM-dd formats
# filter out placeholders like "Unknown", "Not Provided" before parsing
exp_df = exp_df.withColumn('start_clean',
    when(
        col('exp.dates.start').isin('Unknown', 'Not Provided', ''),
        lit(None)
    ).otherwise(col('exp.dates.start'))
)
exp_df = exp_df.withColumn('start_date',
    coalesce(
        to_date(col('start_clean'), 'yyyy-MM-dd'),
        to_date(col('start_clean'), 'yyyy-MM')
    )
)

# parse end dates - "Present" means current date
exp_df = exp_df.withColumn('end_clean',
    when(
        col('exp.dates.end').isin('Unknown', 'Not Provided', ''),
        lit(None)
    ).otherwise(col('exp.dates.end'))
)
exp_df = exp_df.withColumn('end_date',
    when(col('end_clean') == 'Present', current_date())
    .otherwise(
        coalesce(
            to_date(col('end_clean'), 'yyyy-MM-dd'),
            to_date(col('end_clean'), 'yyyy-MM')
        )
    )
)

# compute years for each experience entry
exp_df = exp_df.withColumn('years_exp',
    months_between(col('end_date'), col('start_date')) / 12
)

# filter out negatives and nulls
exp_df = exp_df.filter(
    col('years_exp').isNotNull() & (col('years_exp') >= 0)
)

# sum per CV
total_years_df = exp_df.groupBy('id').agg(
    spark_sum('years_exp').alias('total_years')
)

# join back
df = df.join(total_years_df, on='id', how='left')

# handle null total_years (no valid dates) - default to 0, round to 1 decimal
df = df.withColumn('total_years',
    spark_round(
        when(col('total_years').isNull(), lit(0.0)).otherwise(col('total_years')),
        1
    )
)

print('Total years statistics:')
df.select('total_years').summary('min', 'max', 'mean').show()

# map seniority level
# first normalize existing level values, then infer from years if null
print('Mapping seniority levels')

df = df.withColumn('level',
    when(col('level').isNotNull(),
        # normalize existing values
        when(col('level').isin('entry', 'entry-level', 'intern', 'junior'), lit('entry level'))
        .when(col('level').isin('mid', 'mid-senior'), lit('mid-level'))
        .when(col('level') == 'senior', lit('senior level'))
        .when(col('level').isin('expert', 'principal'), lit('expert level'))
        .otherwise(col('level'))  # keep manager, professional etc as-is
    ).otherwise(
        # infer from total_years
        when(col('total_years') < 2, lit('entry level'))
        .when(col('total_years') < 6, lit('mid-level'))
        .when(col('total_years') < 11, lit('senior level'))
        .otherwise(lit('expert level'))
    )
)

print('Seniority distribution:')
df.groupBy('level').count().orderBy(col('count').desc()).show()

# save normalized output (without cv column) for script 03 join target
print('Saving normalized output')
normalized_df = df.select('id', 'source', 'title', 'company', 'level', 'total_years')
normalized_df.write.mode('overwrite').parquet(NORMALIZED_DIR)
print(f'Saved to {NORMALIZED_DIR}')

# save full dataframe with cv column for script 03 (needs cv.skills.technical and cv.education)
print('Saving normalized with cv column')
df_with_cv = df.select('id', 'source', 'title', 'company', 'level', 'total_years', 'cv')
df_with_cv.write.mode('overwrite').parquet(NORMALIZED_WITH_CV_DIR)
print(f'Saved to {NORMALIZED_WITH_CV_DIR}')

# verify output
print('Verifying output')
verify_df = spark.read.parquet(NORMALIZED_DIR)
print(f'Normalized records: {verify_df.count()}')
verify_df.show(5, truncate=60)

spark.stop()
print('Done')
