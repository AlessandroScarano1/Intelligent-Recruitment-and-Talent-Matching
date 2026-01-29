import os

from pyspark.sql import SparkSession

print('Imports loaded')

# setup paths
cwd = os.getcwd()

if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

INPUT_PATH = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'cv_query_text.parquet')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output')

print(f'Project root: {PROJECT_ROOT}')
print(f'Input: {INPUT_PATH}')
print(f'Output dir: {OUTPUT_DIR}')

# create spark session
print('Creating Spark session')

spark = SparkSession.builder \
    .appName('CVSplits') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '8') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
print(f'Spark version: {spark.version}')

# load input
print('Loading cv_query_text data')
df = spark.read.parquet(INPUT_PATH)
total_count = df.count()
print(f'Loaded {total_count} records')
print(f'Columns: {df.columns}')

# cache before split for deterministic results
print('Caching dataframe before split')
df.cache()
df.count()

# split into train/val/test
print('Splitting into train/val/test with seed=42')
train_df, val_df, test_df = df.randomSplit([0.8, 0.1, 0.1], seed=42)

train_count = train_df.count()
val_count = val_df.count()
test_count = test_df.count()

print(f'Train: {train_count} ({train_count/total_count*100:.1f}%)')
print(f'Validation: {val_count} ({val_count/total_count*100:.1f}%)')
print(f'Test: {test_count} ({test_count/total_count*100:.1f}%)')

# select id column and rename to anchor
train_ids = train_df.select('id').withColumnRenamed('id', 'anchor')
val_ids = val_df.select('id').withColumnRenamed('id', 'anchor')
test_ids = test_df.select('id').withColumnRenamed('id', 'anchor')

# write split files
train_path = os.path.join(OUTPUT_DIR, 'training_set_cv_ids.parquet')
val_path = os.path.join(OUTPUT_DIR, 'validation_set_cv_ids.parquet')
test_path = os.path.join(OUTPUT_DIR, 'test_set_cv_ids.parquet')

print('Writing split files')
train_ids.coalesce(1).write.mode('overwrite').parquet(train_path)
print(f'Saved: {train_path}')

val_ids.coalesce(1).write.mode('overwrite').parquet(val_path)
print(f'Saved: {val_path}')

test_ids.coalesce(1).write.mode('overwrite').parquet(test_path)
print(f'Saved: {test_path}')

# verify schema of written files
print('Verifying written files')

train_verify = spark.read.parquet(train_path)
print(f'Train schema: columns={train_verify.columns}, count={train_verify.count()}')

val_verify = spark.read.parquet(val_path)
print(f'Validation schema: columns={val_verify.columns}, count={val_verify.count()}')

test_verify = spark.read.parquet(test_path)
print(f'Test schema: columns={test_verify.columns}, count={test_verify.count()}')

# cleanup
spark.stop()
print('Done')
