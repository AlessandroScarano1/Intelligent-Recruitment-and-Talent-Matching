import os
import time
import json
import shutil
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

print('Imports loaded')

# setup paths
cwd = os.getcwd()

if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'raw_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output', 'raw_cvs')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'ingest_cv', 'output')

KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka-broker:29092')
TOPIC = 'raw_cvs'
DLQ_TOPIC = 'dlq_cvs'

print(f'Project root: {PROJECT_ROOT}')
print(f'Raw data: {RAW_DATA_DIR}')
print(f'Output: {OUTPUT_DIR}')
print(f'Kafka broker: {KAFKA_BROKER}')

# clean up output folder for fresh run
print('\nCleaning up output folder for fresh pipeline run')
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f'Cleared: raw_cvs')
else:
    print(f'Not found (ok): raw_cvs')

print('Output folder cleaned')

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

def create_topics(broker, topics_config):
    admin = AdminClient({'bootstrap.servers': broker})
    for cfg in topics_config:
        topic = NewTopic(cfg['name'], num_partitions=cfg['partitions'], replication_factor=cfg['replication'])
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
    admin = AdminClient({'bootstrap.servers': broker})
    topic_names = [cfg['name'] for cfg in topics_config]
    print('Cleaning up old topic data')
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
    print('Waiting for topic deletion')
    time.sleep(3)
    print('Recreating topics')
    create_topics(broker, topics_config)

TOPICS_CONFIG = [
    {'name': 'raw_cvs', 'partitions': 3, 'replication': 1},
    {'name': 'dlq_cvs', 'partitions': 1, 'replication': 1},
]

print('Resetting Kafka topics for clean slate')
reset_topics(KAFKA_BROKER, TOPICS_CONFIG)
print('Topics ready')

# kafka producer class
class CVProducer:

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
        if err:
            self.errors += 1

    def produce_cvs(self, topic, filepath, dlq):
        print(f'\nProducing CVs from {os.path.basename(filepath)}')

        if not os.path.exists(filepath):
            print(f'  File not found, skipping')
            return 0

        count = 0
        malformed_count = 0
        start = time.time()

        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    cv_record = json.loads(line)

                    msg = {
                        'id': f'A{idx}',
                        'source': 'huggingface_resumes',
                        'raw_data': json.dumps(cv_record)
                    }

                    self.producer.produce(
                        topic,
                        value=json.dumps(msg).encode('utf-8'),
                        callback=self.delivery_callback
                    )

                    count += 1
                    self.produced += 1

                    if count % 1000 == 0:
                        self.producer.poll(0)
                        elapsed = time.time() - start
                        rate = count / elapsed
                        print(f' {count:,} sent ({rate:,.0f} msg/sec)')

                except json.JSONDecodeError as e:
                    malformed_count += 1
                    dlq.send_to_dlq(line, 'JSONDecodeError', str(e))

        self.producer.flush()
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0
        print(f' Done: {count:,} messages in {elapsed:.1f}s ({rate:,.0f} msg/sec)')
        print(f' Malformed: {malformed_count}')

        return count

print('Producer class defined')

# DLQ producer for failed messages
class DLQProducer:

    def __init__(self, broker, dlq_topic='dlq_cvs'):
        self.dlq_topic = dlq_topic
        self.producer = Producer({
            'bootstrap.servers': broker,
            'client.id': 'dlq-producer'
        })
        self.failed_count = 0

    def send_to_dlq(self, original_message, error_type, error_message):
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

# define message fields to save
MESSAGE_FIELDS = ['id', 'source', 'raw_data']

def write_batch(batch, output_dir, file_num, consumer):
    df = pd.DataFrame(batch)

    for col in MESSAGE_FIELDS:
        if col not in df.columns:
            df[col] = ''

    df = df[MESSAGE_FIELDS]

    filename = f'part-{file_num:04d}.parquet'
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath, index=False)

    consumer.commit()
    return filename

print('Batch writer defined')

# kafka consumer function
def consume_all(topic, broker, output_dir, batch_size=10000):
    consumer = Consumer({
        'bootstrap.servers': broker,
        'group.id': 'cv-batch-consumer-v1',
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

                if messages_consumed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = messages_consumed / elapsed
                    print(f'  {messages_consumed:,} consumed ({rate:,.0f} msg/sec)')

                if len(batch) >= batch_size:
                    fname = write_batch(batch, output_dir, files_written, consumer)
                    files_written += 1
                    batch = []

            except json.JSONDecodeError as e:
                dlq.send_to_dlq(raw_value, 'JSONDecodeError', str(e))
            except Exception as e:
                dlq.send_to_dlq(raw_value, type(e).__name__, str(e))

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

# produce CVs to kafka
print('PRODUCING CVS TO KAFKA')

dlq = DLQProducer(KAFKA_BROKER)
producer = CVProducer(KAFKA_BROKER)

cv_file = os.path.join(RAW_DATA_DIR, 'master_resumes.jsonl')
total_count = producer.produce_cvs(TOPIC, cv_file, dlq)

dlq.flush()

print()
print('PRODUCER SUMMARY')
print(f'Total messages: {total_count:,}')
print(f'Errors: {producer.errors}')
print(f'DLQ: {dlq.failed_count}')

# clear old output
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f'Cleared old output: {OUTPUT_DIR}')

# run consumer
consumed, files = consume_all(TOPIC, KAFKA_BROKER, OUTPUT_DIR, batch_size=10000)

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

# compare producer vs consumer
print('\nPipeline verification')
print(f'Producer sent: {total_count:,}')
print(f'Consumer wrote: {total_records:,}')
if total_count == total_records:
    print('Match: No message loss')
else:
    print(f'Mismatch: {total_count - total_records} messages lost')
