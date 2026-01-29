import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, collect_set, concat_ws, coalesce, lit,
    when, concat, size, split
)

print('Imports loaded')

# setup paths
cwd = os.getcwd()

if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

NORMALIZED_WITH_CV_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'normalized_with_cv')
NORMALIZED_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'normalized')
SKILLS_AGG_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'skills_aggregated')

print(f'Project root: {PROJECT_ROOT}')
print(f'Input: {NORMALIZED_WITH_CV_DIR}')
print(f'Output: {SKILLS_AGG_DIR}')

# create spark session
print('Creating Spark session')

spark = SparkSession.builder \
    .appName('CVSkillAggregation') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '8') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# load script 02 output with parsed cv column
print('Loading normalized data with cv column')
df = spark.read.parquet(NORMALIZED_WITH_CV_DIR)
total_count = df.count()
print(f'Loaded {total_count} records')
print(f'Columns: {df.columns}')

# aggregate skills from all technical subcategories
print('Aggregating skills from technical subcategories')

TECH_SUBCATEGORIES = [
    'programming_languages',
    'frameworks',
    'databases',
    'cloud',
    'tools',
    'web_technologies',
    'testing',
    'automation',
    'project_management',
    'networking_skills',
    'software',
    'software_tools',
    'operating_systems',
    'other',
]

# helper to extract skills from one subcategory
def extract_subcategory_skills(dataframe, subcategory):
    path = f'cv.skills.technical.{subcategory}'
    return dataframe.select(
        col('id'),
        explode(col(path)).alias('skill_entry')
    ).select(
        col('id'),
        col('skill_entry.name').alias('skill')
    ).filter(
        col('skill').isNotNull() &
        (col('skill') != '') &
        (~col('skill').isin('Unknown', 'Not Provided'))
    )

# collect skills from all subcategories
skill_dfs = []
for subcat in TECH_SUBCATEGORIES:
    try:
        subcat_df = extract_subcategory_skills(df, subcat)
        skill_dfs.append(subcat_df)
    except Exception as e:
        print(f'Warning: could not extract {subcat}: {e}')

# union all skill dataframes
all_skills = skill_dfs[0]
for sdf in skill_dfs[1:]:
    all_skills = all_skills.union(sdf)

print(f'Total skill entries from technical subcategories: {all_skills.count()}')

# also extract skills from experience.technical_environment.technologies
print('Extracting skills from experience.technical_environment.technologies')

# explode experience array first, then technologies
exp_df = df.select(
    col('id'),
    explode(col('cv.experience')).alias('exp')
)

tech_env_skills = exp_df.select(
    col('id'),
    explode(col('exp.technical_environment.technologies')).alias('skill')
).filter(
    col('skill').isNotNull() &
    (col('skill') != '') &
    (~col('skill').isin('Unknown', 'Not Provided'))
)

tech_env_count = tech_env_skills.count()
print(f'Skills from technical_environment: {tech_env_count}')

# union with subcategory skills
all_skills = all_skills.union(tech_env_skills)
print(f'Total skill entries (all sources): {all_skills.count()}')

# deduplicate and aggregate per CV
print('Deduplicating and aggregating per CV')

skills_agg = all_skills.groupBy('id').agg(
    concat_ws(', ', collect_set(col('skill'))).alias('skills')
)

print('Sample skills strings:')
skills_agg.show(5, truncate=100)

# extract education summary from cv.education[0]
print('Extracting education summary')

edu_df = df.select(
    col('id'),
    col('cv.education')[0]['degree']['level'].alias('degree_level'),
    col('cv.education')[0]['degree']['field'].alias('degree_field'),
    col('cv.education')[0]['institution']['name'].alias('institution_name')
)

# clean placeholders in education fields
for field in ['degree_level', 'degree_field', 'institution_name']:
    edu_df = edu_df.withColumn(field,
        when(
            col(field).isNull() |
            (col(field) == '') |
            (col(field) == 'Unknown') |
            (col(field) == 'Not Provided'),
            lit(None)
        ).otherwise(col(field))
    )

# build education_summary string
# "degree_level in degree_field from institution_name"
edu_df = edu_df.withColumn('degree_part',
    when(
        col('degree_level').isNotNull() & col('degree_field').isNotNull(),
        concat(col('degree_level'), lit(' in '), col('degree_field'))
    ).when(
        col('degree_level').isNotNull(),
        col('degree_level')
    ).when(
        col('degree_field').isNotNull(),
        concat(lit('degree in '), col('degree_field'))
    ).otherwise(lit(None))
)

edu_df = edu_df.withColumn('education_summary',
    when(
        col('degree_part').isNotNull() & col('institution_name').isNotNull(),
        concat(col('degree_part'), lit(' from '), col('institution_name'))
    ).when(
        col('degree_part').isNotNull(),
        col('degree_part')
    ).when(
        col('institution_name').isNotNull(),
        concat(lit('graduate from '), col('institution_name'))
    ).otherwise(lit(None))
)

edu_summary = edu_df.select('id', 'education_summary')

print('Sample education summaries:')
edu_summary.filter(col('education_summary').isNotNull()).show(10, truncate=100)

non_null_edu = edu_summary.filter(col('education_summary').isNotNull()).count()
print(f'CVs with education summary: {non_null_edu} / {total_count}')

# join back to normalized data (without cv column)
print('Loading normalized data for final join')
normalized_df = spark.read.parquet(NORMALIZED_DIR)
print(f'Normalized records: {normalized_df.count()}')

# left join with skills
result_df = normalized_df.join(skills_agg, on='id', how='left')

# left join with education
result_df = result_df.join(edu_summary, on='id', how='left')

# fallback for skills (not for education, let script 04 handle null education)
result_df = result_df.withColumn('skills',
    coalesce(col('skills'), lit('various technical skills'))
)

# print skill statistics
print('Skill statistics:')
with_skills = result_df.filter(col('skills') != 'various technical skills').count()
without_skills = result_df.filter(col('skills') == 'various technical skills').count()
print(f'CVs with extracted skills: {with_skills}')
print(f'CVs with fallback skills: {without_skills}')

# average number of skills per CV
result_df_with_count = result_df.withColumn('skill_count',
    size(split(col('skills'), ', '))
)
print('Skills per CV:')
result_df_with_count.select('skill_count').summary('min', 'max', 'mean').show()

# save output
print('Saving skills_aggregated output')
output_df = result_df.select('id', 'source', 'title', 'company', 'level', 'total_years', 'skills', 'education_summary')
output_df.write.mode('overwrite').parquet(SKILLS_AGG_DIR)
print(f'Saved to {SKILLS_AGG_DIR}')

# verify output
print('Verifying output')
verify_df = spark.read.parquet(SKILLS_AGG_DIR)
print(f'Records: {verify_df.count()}')
print(f'Columns: {verify_df.columns}')
verify_df.show(5, truncate=80)

spark.stop()
print('Done')
