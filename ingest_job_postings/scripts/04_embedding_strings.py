# %%
# imports
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, when, trim
from pyspark.sql.types import StringType
import re

# setup paths - detect project root
import sys
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

print("Imports loaded")
print(f"Project root: {PROJECT_ROOT}")

# %%
# start spark
spark = SparkSession.builder \
    .appName("BuildEmbeddingStrings") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session started")
print(f"Version: {spark.version}")

# %%
# load linkedin jobs, already have skills from JOIN in notebook 02
linkedin_path = os.path.join(PROJECT_ROOT, "ingest_job_postings", "output", "processed", "linkedin", "linkedin_jobs_with_skills")
linkedin_df = spark.read.parquet(linkedin_path)
linkedin_count = linkedin_df.count()
print(f"LinkedIn jobs: {linkedin_count:,}")
print("\nColumns:")
print(linkedin_df.columns)
print("\nSample:")
linkedin_df.select("id", "job_title", "company", "skills").show(3, truncate=50)

# %%
# load indeed/glassdoor jobs, already have NLP extracted fields from notebook 03
indeed_glassdoor_path = os.path.join(PROJECT_ROOT, "ingest_job_postings", "output", "processed", "indeed_glassdoor", "indeed_glassdoor_extracted")
indeed_glassdoor_df = spark.read.parquet(indeed_glassdoor_path)
ig_count = indeed_glassdoor_df.count()
print(f"Indeed/Glassdoor jobs: {ig_count:,}")
print("\nColumns:")
print(indeed_glassdoor_df.columns)
print("\nSample:")
indeed_glassdoor_df.select("id", "job_title", "company", "skills").show(3, truncate=50)

# %%
# extract seniority from job titles
# simple regex patterns for common seniority indicators
def extract_seniority_from_title(title):
    # extract seniority level from job title, returns: intern, junior, mid, senior, lead, principal, or None
    
    if not title or not str(title).strip():
        return None
    
    title_lower = str(title).lower()
    
    # patterns in priority order (most specific first)
    if any(word in title_lower for word in ['intern', 'internship', 'trainee']):
        return 'intern'
    elif any(word in title_lower for word in ['principal', 'staff', 'distinguished']):
        return 'principal'
    elif any(word in title_lower for word in ['lead', 'head of', 'director', 'vp', 'chief']):
        return 'lead'
    elif any(word in title_lower for word in ['senior', 'sr.', 'sr ']):
        return 'senior'
    elif any(word in title_lower for word in ['junior', 'jr.', 'jr ', 'entry']):
        return 'junior'
    else:
        # default to mid if no keywords found
        return 'mid'

# register as UDF
extract_seniority_udf = udf(extract_seniority_from_title, StringType())

print("Seniority extraction function defined")

# test on sample titles
test_titles = [
    "Senior Software Engineer",
    "Data Science Intern",
    "Lead Machine Learning Engineer",
    "Software Developer",
    "Principal Data Scientist"
]
print("\nTest seniority extraction:")
for title in test_titles:
    seniority = extract_seniority_from_title(title)
    print(f"  '{title}' -> {seniority}")

# %%
# add seniority column to linkedin jobs
linkedin_df = linkedin_df.withColumn(
    "seniority",
    extract_seniority_udf(col("job_title"))
)

# check seniority distribution
print("LinkedIn seniority distribution:")
linkedin_df.groupBy("seniority").count().orderBy(col("count").desc()).show()

# %%
# build embedding string function
# creates natural language summary from job fields
def build_embedding_string(title, company, location, skills, seniority, 
                          salary_min, salary_max, remote):
    
    # natural language embedding string from job fields, template:
    # "Role of {title} at {company} in {location}. Required skills: {skills}. 
    # Experience level: {seniority}. Salary: {salary}. Work type: {remote}."

    # handle missing values
    title = str(title).strip() if title and str(title).strip() else 'Unknown Position'
    company = str(company).strip() if company and str(company).strip() else 'a company'
    
    # start building string
    parts = [f"Role of {title} at {company}"]
    
    # add location if available
    if location and str(location).strip():
        parts[0] += f" in {str(location).strip()}"
    parts[0] += "."
    
    # skills - limit to top 10 to keep string manageable
    if skills and str(skills).strip():
        skill_list = [s.strip() for s in str(skills).split(',')[:10]]
        if skill_list:
            parts.append(f"Required skills: {', '.join(skill_list)}.")
    
    # seniority with expanded descriptions
    if seniority and str(seniority).strip():
        seniority_map = {
            'intern': 'Intern level, entry position',
            'junior': 'Junior level, 0-2 years experience',
            'mid': 'Mid-level, 3-5 years experience',
            'senior': 'Senior level, 5+ years experience',
            'lead': 'Lead level, 7+ years experience with leadership',
            'principal': 'Principal level, expert with technical leadership'
        }
        level = seniority_map.get(str(seniority).lower().strip(), str(seniority))
        parts.append(f"Experience level: {level}.")
    
    # salary range
    if salary_min and float(salary_min) > 0:
        if salary_max and float(salary_max) > 0 and float(salary_max) > float(salary_min):
            parts.append(f"Salary range: ${float(salary_min):,.0f} to ${float(salary_max):,.0f}.")
        else:
            parts.append(f"Minimum salary: ${float(salary_min):,.0f}.")
    
    # remote status
    if remote and str(remote).strip():
        remote_map = {
            'remote': 'Remote work available',
            'hybrid': 'Hybrid work, partially remote',
            'onsite': 'Onsite'
        }
        work_type = remote_map.get(str(remote).lower().strip(), str(remote))
        parts.append(f"Work type: {work_type}.")
    
    return ' '.join(parts)

# register as UDF
build_embedding_udf = udf(build_embedding_string, StringType())

print("Embedding string builder defined")

# test on sample data
test_str = build_embedding_string(
    title="Senior Data Scientist",
    company="Tech Corp",
    location="San Francisco, CA",
    skills="Python, Machine Learning, SQL, Spark",
    seniority="senior",
    salary_min=120000,
    salary_max=180000,
    remote="hybrid"
)
print("\nTest embedding string:")
print(test_str)

# %%
# add embedding strings to linkedin jobs
# linkedin doesn't have salary_min, salary_max, remote_status, use lit(None)
from pyspark.sql.functions import lit

linkedin_df = linkedin_df.withColumn(
    "embedding_text",
    build_embedding_udf(
        col("job_title"),
        col("company"),
        col("job_location"),
        col("skills"),
        col("seniority"),
        lit(None),  # salary_min not available
        lit(None),  # salary_max not available
        lit(None)   # remote_status not available
    )
)

print("LinkedIn embedding strings generated")
print("\nSample LinkedIn embedding strings:")
linkedin_df.select("job_title", "embedding_text").show(3, truncate=100)

# %%
# add embedding strings to indeed/glassdoor jobs
# these also don't have salary/remote columns, so use lit(None)
indeed_glassdoor_df = indeed_glassdoor_df.withColumn(
    "embedding_text",
    build_embedding_udf(
        col("job_title"),
        col("company"),
        col("job_location"),
        col("skills"),
        col("seniority"),
        lit(None),  # salary_min not available
        lit(None),  # salary_max not available
        lit(None)   # remote_status not available
    )
)

print("Indeed/Glassdoor embedding strings generated")
print("\nSample Indeed/Glassdoor embedding strings:")
indeed_glassdoor_df.select("job_title", "embedding_text").show(3, truncate=100)

# %%
# select common columns for union,keep only columns that exist in both dataframes
common_cols = [
    "id",
    "job_title",
    "company",
    "job_location",
    "skills",
    "seniority",
    "embedding_text"
]

linkedin_subset = linkedin_df.select(*common_cols)
ig_subset = indeed_glassdoor_df.select(*common_cols)

# union all sources
all_jobs_df = linkedin_subset.union(ig_subset)
total_before = all_jobs_df.count()
print(f"Total jobs before dedup: {total_before:,}")
print(f"  LinkedIn: {linkedin_count:,}")
print(f"  Indeed/Glassdoor: {ig_count:,}")

# %%
# remove duplicate embedding strings
# keep first occurrence to preserve UUIDs
all_jobs_df = all_jobs_df.dropDuplicates(["embedding_text"])
total_after = all_jobs_df.count()
removed = total_before - total_after

print(f"Total jobs after dedup: {total_after:,}")
print(f"Removed {removed:,} duplicates ({removed/total_before*100:.2f}%)")

# %%
# check for missing embedding strings
missing_count = all_jobs_df.filter(col("embedding_text").isNull() | (col("embedding_text") == "")).count()
print(f"Missing embedding strings: {missing_count}")

if missing_count > 0:
    print("\n!!!!!!! Found records with missing embedding strings!")
    all_jobs_df.filter(col("embedding_text").isNull() | (col("embedding_text") == "")).show(5)
else:
    print("All records have embedding strings")

# %%
# check embedding string lengths
from pyspark.sql.functions import length, avg, min as spark_min, max as spark_max

lengths_df = all_jobs_df.withColumn("text_length", length(col("embedding_text")))
length_stats = lengths_df.agg(
    avg("text_length").alias("avg_len"),
    spark_min("text_length").alias("min_len"),
    spark_max("text_length").alias("max_len")
).collect()[0]

print("Embedding string length statistics:")
print(f"  Average: {length_stats['avg_len']:.0f} chars")
print(f"  Min: {length_stats['min_len']} chars")
print(f"  Max: {length_stats['max_len']} chars")

# check for very short strings (might indicate data quality issues)
too_short = lengths_df.filter(col("text_length") < 50).count()
print(f"\nStrings under 50 chars: {too_short} ({too_short/total_after*100:.2f}%)")

# %%
# show sample embedding strings from different seniority levels
print("Sample embedding strings by seniority:")
for level in ['intern', 'junior', 'mid', 'senior', 'lead', 'principal']:
    sample = all_jobs_df.filter(col("seniority") == level).limit(1).collect()
    if sample:
        print(f"\n[{level.upper()}]")
        print(sample[0]['embedding_text'])

# %%
# save unified jobs with embedding strings
output_path = os.path.join(PROJECT_ROOT, "ingest_job_postings", "output", "unified_job_postings", "unified_jobs.parquet")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
all_jobs_df.write.mode("overwrite").parquet(output_path)

print(f"Saved to: {output_path}")
print(f"Total records: {total_after:,}")
print(f"Columns: {all_jobs_df.columns}")

# %%
# final summary
print(f"LinkedIn jobs: {linkedin_count:,}")
print(f"Indeed/Glassdoor jobs: {ig_count:,}")
print(f"Total before dedup: {total_before:,}")
print(f"Duplicates removed: {removed:,}")
print(f"Final job count: {total_after:,}")
print(f"\nOutput: {output_path}")
print(f"\nThis is the UNIFIED output from all sources, next need to run notebook 05 to assign sequential B IDs")


# %%
# cleanup
spark.stop()
print("Done")


