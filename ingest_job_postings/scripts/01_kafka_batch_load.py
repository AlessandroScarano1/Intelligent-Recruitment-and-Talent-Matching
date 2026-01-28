# %%
# imports
import os
import time
import json
import uuid
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

print('Imports loaded')

# %%
# setup paths
# this notebook expects to run from notebooks/ directory at project root
import shutil

cwd = os.getcwd()

# detect if we're in notebooks/ or project root
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'raw_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'raw_job_postings')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output')

# kafka config - using docker network name
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka-broker:29092')
TOPIC = 'raw_job_postings'
DLQ_TOPIC = 'dlq_job_postings'

print(f'Project root: {PROJECT_ROOT}')
print(f'Raw data: {RAW_DATA_DIR}')
print(f'Output: {OUTPUT_DIR}')
print(f'Kafka broker: {KAFKA_BROKER}')

# clean up ALL output folders for fresh pipeline run
print('\nCleaning up ALL output folders for fresh pipeline run...')
folders_to_clean = [
    'raw_job_postings',
    'processed',
    'skill_dictionary',
    'unified_job_postings',
    'final',
    'embeddings',
    'splits',
]

for folder in folders_to_clean:
    folder_path = os.path.join(OUTPUT_ROOT, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f'Cleared: {folder}')
    else:
        print(f'Not found (ok): {folder}')

print('Output folders cleaned')

# %%
# test kafka connection
print('Testing Kafka connection')

try:
    admin = AdminClient({'bootstrap.servers': KAFKA_BROKER})
    cluster = admin.list_topics(timeout=10)
    print(f'[OK] Connected to Kafka')
    print(f'Existing topics: {list(cluster.topics.keys())}')
except Exception as e:
    print(f'[ERROR] Cannot connect to Kafka: {e}')
    print('\nMake sure Kafka is running:')
    print('docker-compose up -d')
    raise SystemExit('Kafka not available')

# %%
# function to create kafka topics
def create_topics(broker, topics_config):
    #create Kafka topics if they don't exist
    admin = AdminClient({'bootstrap.servers': broker})
    
    for cfg in topics_config:
        topic = NewTopic(
            cfg['name'], 
            num_partitions=cfg['partitions'], 
            replication_factor=cfg['replication']
        )
        
        try:
            fs = admin.create_topics([topic])
            for t, f in fs.items():
                try:
                    f.result()
                    print(f' Created topic: {t}')
                except Exception as e:
                    if 'already exists' in str(e).lower():
                        print(f'  Topic exists: {t}')
                    else:
                        print(f'  Error with {t}: {e}')
        except Exception as e:
            print(f'  Error: {e}')

def reset_topics(broker, topics_config):
    # delete and recreate topics for a clean slate.
    # This prevents accumulation of old messages from previous runs, without this, consumer with auto.offset.reset='earliest' would
    # read ALL messages ever produced to the topic
    admin = AdminClient({'bootstrap.servers': broker})
    topic_names = [cfg['name'] for cfg in topics_config]
    
    print('Cleaning up old topic data')
    
    # delete existing topics
    try:
        fs = admin.delete_topics(topic_names, operation_timeout=30)
        for topic, future in fs.items():
            try:
                future.result()
                print(f'Deleted: {topic}')
            except Exception as e:
                if 'does not exist' in str(e).lower() or 'unknown topic' in str(e).lower():
                    print(f'Not found (ok): {topic}')
                else:
                    print(f'Error deleting {topic}: {e}')
    except Exception as e:
        print(f' Delete error: {e}')
    
    # wait for deletion to complete
    print('Waiting for topic deletion')
    time.sleep(3)
    
    # recreate topics
    print('Recreating topics')
    create_topics(broker, topics_config)

# topic configuration
TOPICS_CONFIG = [
    {'name': 'raw_job_postings', 'partitions': 3, 'replication': 1},
    {'name': 'dlq_job_postings', 'partitions': 1, 'replication': 1},
]

# IMPORTANT: reset topics to prevent duplicate data from previous runs
print('Resetting Kafka topics for clean slate')
reset_topics(KAFKA_BROKER, TOPICS_CONFIG)
print('Topics ready')

# %%
# check data files exist
print('Checking data files')

data_files = {
    'linkedin': 'linkedin_job_postings.csv',
    'linkedin_skills': 'job_skills.csv',
    'indeed': 'indeed_job_listings.csv',
    'glassdoor': 'job_descriptions.csv',
}

for name, filename in data_files.items():
    path = os.path.join(RAW_DATA_DIR, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        # count lines
        with open(path, 'rb') as f:
            lines = sum(1 for _ in f) - 1
        print(f'  {name}: {lines:,} rows, {size_mb:.1f} MB')
    else:
        print(f' Missing: {filename}')

# %%
# column mappings for each data source
# each source has different column names, so we map them to a unified schema
SOURCE_MAPPINGS = {
    'linkedin': {
        'columns': {
            'job_title': 'job_title',
            'company': 'company',
            'job_location': 'job_location',
            'job_level': 'job_level',
            'job_type': 'job_type',
            'job_link': 'job_link',
        },
    },
    'indeed': {
        'columns': {
            'job_title': 'job_title',
            'company': 'company',
            'job_location': 'location',
            'description': 'summary',
        },
    },
    'glassdoor': {
        'columns': {
            'job_title': 'position',
            'company': 'company',
            'job_location': 'location',
            'description': 'Job Description',
        },
    },
}

def map_row_to_message(row, source_name):
    # map a CSV row to unified message format
    mapping = SOURCE_MAPPINGS[source_name]['columns']
    
    msg = {
        'id': str(uuid.uuid4()),
        'source': source_name,
        'job_title': str(row.get(mapping.get('job_title', ''), '') or ''),
        'company': str(row.get(mapping.get('company', ''), '') or ''),
        'job_location': str(row.get(mapping.get('job_location', ''), '') or ''),
    }
    
    # linkedin specific fields
    if source_name == 'linkedin':
        msg['job_link'] = str(row.get('job_link', '') or '')
        msg['job_level'] = str(row.get('job_level', '') or '')
        msg['job_type'] = str(row.get('job_type', '') or '')
    
    # description for indeed and glassdoor
    if 'description' in mapping and mapping['description']:
        msg['description'] = str(row.get(mapping['description'], '') or '')
    
    return msg

print('source mappings defined')

# %%
# kafka producer class
class JobPostingProducer:
    #producer to send job postings to Kafka
    
    def __init__(self, broker):
        self.producer = Producer({
            'bootstrap.servers': broker,
            'client.id': 'batch-producer',
            'queue.buffering.max.messages': 500000,
            'queue.buffering.max.kbytes': 1048576,
            'batch.num.messages': 10000,
            'linger.ms': 100,
        })
        self.produced = 0
        self.errors = 0
    
    def delivery_callback(self, err, msg):
        # callback for message delivery
        if err:
            self.errors += 1
    
    def produce_source(self, topic, source_name, filepath, max_rows=None):
        # produce messages from a CSV file
        print(f'\nProducing {source_name} from {os.path.basename(filepath)}')
        
        if not os.path.exists(filepath):
            print(f'  File not found, skipping')
            return 0
        
        # load data
        # load data in chunks to save memory
        chunk_size = 10000
        
        print(f'  Processing in chunks of {chunk_size}...')
        
        count = 0
        start = time.time()
        
        # Create an iterator
        if max_rows:
            csv_iterator = pd.read_csv(filepath, nrows=max_rows, low_memory=False, chunksize=chunk_size)
        else:
            csv_iterator = pd.read_csv(filepath, low_memory=False, chunksize=chunk_size)
            
        for chunk_df in csv_iterator:
            print(f'processing chunk ({len(chunk_df)} rows)')
            
            # send each row to kafka
            for _, row in chunk_df.iterrows():
                msg = map_row_to_message(row.to_dict(), source_name)
                
                self.producer.produce(
                    topic,
                    value=json.dumps(msg).encode('utf-8'),
                    callback=self.delivery_callback
                )
                
                count += 1
                self.produced += 1
                
                # show progress
                if count % 10000 == 0:
                    self.producer.poll(0)
                    elapsed = time.time() - start
                    rate = count / elapsed
                    print(f' {count:,} sent ({rate:,.0f} msg/sec)')
                    
            # Flush periodically to free up producer buffer memory
            self.producer.flush()
            
        # Final cleanup not needed as loop ends
        print(f'Finished file.')
        
        # flush remaining messages
        self.producer.flush()
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0
        print(f' Done: {count:,} messages in {elapsed:.1f}s ({rate:,.0f} msg/sec)')
        
        return count

print('Producer class defined')

# %%
# produce all data to kafka
print('PRODUCING JOB POSTINGS TO KAFKA')

producer = JobPostingProducer(KAFKA_BROKER)

# sources to process - NO LIMITS for production
sources = [
    ('linkedin', 'linkedin_job_postings.csv', None),
    ('indeed', 'indeed_job_listings.csv', None),
    ('glassdoor', 'job_descriptions.csv', None),
]

total_start = time.time()
total_count = 0

for source_name, filename, max_rows in sources:
    filepath = os.path.join(RAW_DATA_DIR, filename)
    count = producer.produce_source(TOPIC, source_name, filepath, max_rows)
    total_count += count

total_elapsed = time.time() - total_start

print()
print('PRODUCER SUMMARY')
print(f'Total messages: {total_count:,}')
print(f'Total time: {total_elapsed:.1f}s')
print(f'Errors: {producer.errors}')

# %%
# DLQ producer for failed messages
class DLQProducer:
    #producer for dead letter queue
    
    def __init__(self, broker, dlq_topic='dlq_job_postings'):
        self.dlq_topic = dlq_topic
        self.producer = Producer({
            'bootstrap.servers': broker, 
            'client.id': 'dlq-producer'
        })
        self.failed_count = 0
    
    def send_to_dlq(self, original_message, error_type, error_message):
        # send failed message to DLQ
        dlq_record = {
            'original_message': original_message,
            'error_type': error_type,
            'error_message': str(error_message),
            'failed_at': datetime.now().isoformat(),
        }
        try:
            self.producer.produce(
                self.dlq_topic, 
                value=json.dumps(dlq_record).encode('utf-8')
            )
            self.producer.poll(0)
            self.failed_count += 1
        except:
            pass
    
    def flush(self):
        self.producer.flush()

print('DLQ producer defined')

# %%
# define message fields to save
MESSAGE_FIELDS = [
    'id', 'source', 'job_link', 'job_title', 'company', 
    'job_location', 'job_level', 'job_type', 'description'
]

def write_batch(batch, output_dir, file_num, consumer):
    # write a batch of messages to parquet
    df = pd.DataFrame(batch)
    
    # add missing columns
    for col in MESSAGE_FIELDS:
        if col not in df.columns:
            df[col] = ''
    
    df = df[MESSAGE_FIELDS]
    
    # write to parquet
    filename = f'part-{file_num:04d}.parquet'
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath, index=False)
    
    # commit offset
    consumer.commit()
    return filename

print('Batch writer defined')

# %%
# kafka consumer function
def consume_all(topic, broker, output_dir, batch_size=10000):
    # consume all messages from Kafka and write to parquet
    consumer = Consumer({
        'bootstrap.servers': broker,
        'group.id': 'full-batch-consumer-v2',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'fetch.message.max.bytes': 52428800,
        'max.poll.interval.ms': 600000,
    })
    consumer.subscribe([topic])
    
    dlq = DLQProducer(broker)
    os.makedirs(output_dir, exist_ok=True)
    
    processed_ids = set()
    messages_consumed = 0
    duplicates = 0
    files_written = 0
    batch = []
    empty_polls = 0
    max_empty_polls = 15
    
    start_time = time.time()
    
    print('CONSUMING MESSAGES FROM KAFKA')
    print(f'Topic: {topic}')
    print(f'Output: {output_dir}')
    print(f'Batch size: {batch_size:,}')
    print()
    
    try:
        while empty_polls < max_empty_polls:
            msg = consumer.poll(timeout=2.0)
            
            if msg is None:
                empty_polls += 1
                continue
            
            empty_polls = 0
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            
            raw_value = None
            try:
                raw_value = msg.value().decode('utf-8')
                data = json.loads(raw_value)
                
                # check for duplicates
                msg_id = data.get('id', '')
                if msg_id in processed_ids:
                    duplicates += 1
                    continue
                
                record = {field: data.get(field, '') for field in MESSAGE_FIELDS}
                
                if not record.get('id'):
                    raise ValueError('Missing id')
                
                batch.append(record)
                processed_ids.add(msg_id)
                messages_consumed += 1
                
                # show progress
                if messages_consumed % 100000 == 0:
                    elapsed = time.time() - start_time
                    rate = messages_consumed / elapsed
                    print(f'  {messages_consumed:,} consumed ({rate:,.0f} msg/sec)')
                
                # write batch if full
                if len(batch) >= batch_size:
                    fname = write_batch(batch, output_dir, files_written, consumer)
                    files_written += 1
                    batch = []
            
            except json.JSONDecodeError as e:
                dlq.send_to_dlq(raw_value, 'JSONDecodeError', str(e))
            except Exception as e:
                dlq.send_to_dlq(raw_value, type(e).__name__, str(e))
        
        # write remaining batch
        if batch:
            fname = write_batch(batch, output_dir, files_written, consumer)
            files_written += 1
    
    finally:
        dlq.flush()
        consumer.close()
    
    elapsed = time.time() - start_time
    
    print()
    print('CONSUMER SUMMARY')
    print(f'Messages consumed: {messages_consumed:,}')
    print(f'Duplicates skipped: {duplicates:,}')
    print(f'Failed (DLQ): {dlq.failed_count:,}')
    print(f'Files written: {files_written}')
    print(f'Time: {elapsed:.1f}s')
    if elapsed > 0:
        print(f'Rate: {messages_consumed/elapsed:,.0f} msg/sec')
    
    return messages_consumed, files_written

print('Consumer function defined')

# %%
# clear old output
import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f'Cleared old output: {OUTPUT_DIR}')

# run consumer
consumed, files = consume_all(TOPIC, KAFKA_BROKER, OUTPUT_DIR, batch_size=10000)

# %%
# verify output files
print('Verifying output')

parquet_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.parquet')])
total_records = 0
total_size = 0

for f in parquet_files:
    path = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(path) / 1e6
    total_size += size_mb
    df = pd.read_parquet(path)
    total_records += len(df)

print(f'Files: {len(parquet_files)}')
print(f'Total records: {total_records:,}')
print(f'Total size: {total_size:.1f} MB')

# count by source
print('\nRecords by source:')
all_dfs = [pd.read_parquet(os.path.join(OUTPUT_DIR, f)) for f in parquet_files]
combined = pd.concat(all_dfs, ignore_index=True)
print(combined['source'].value_counts())

# %%
# show sample records
print('\nSample records:')

for source in ['linkedin', 'indeed', 'glassdoor']:
    sample = combined[combined['source'] == source].head(1)
    if len(sample) > 0:
        print(f'\n{source.upper()}:')
        row = sample.iloc[0]
        print(f'  Title: {row["job_title"][:60]}')
        print(f'  Company: {row["company"][:40]}')
        print(f'  Location: {row["job_location"][:40]}')
        if source == 'linkedin':
            print(f'  Level: {row["job_level"]}')
            print(f'  Link: {row["job_link"][:60]}...')

# %%
print('KAFKA BATCH LOAD COMPLETE')
print(f'Output: {OUTPUT_DIR}')
print(f'Records: {total_records:,}')
print(f'Files: {len(parquet_files)}')


