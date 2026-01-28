#file watcher service for automatic document ingestion
# monitors incoming/cv/ and incoming/job/ for new files

# usage:
#     python demo/scripts/file_watcher.py

# features:
#     - debounced file detection (avoids duplicate events)
#     - multi-format document parsing
#     - optional Kafka integration
#     - graceful shutdown on SIGTERM/SIGINT

import os
import sys
import time
import signal
import logging
import json
import uuid
from pathlib import Path
from threading import Timer
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.scripts.document_parser import parse_document, detect_document_type

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# directories to watch
CV_DIR = Path("incoming/cv")
JOB_DIR = Path("incoming/job")
PROCESSED_DIR = Path("incoming/processed")

# debounce delay (seconds) - handles multiple events per file
DEBOUNCE_DELAY = 2.0

# supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.csv'}


class DebouncedFileHandler(FileSystemEventHandler):
    # file handler with debouncing to avoid processing duplicate events
    # watchdog fires multiple events per file (created, modified, modified...)

    def __init__(self, doc_type, callback):
        super().__init__()
        self.doc_type = doc_type
        self.callback = callback
        self.timers = {}  # filepath -> Timer

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule_processing(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._schedule_processing(event.src_path)

    def _schedule_processing(self, filepath):
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return

        # cancel previous timer if exists
        if filepath in self.timers:
            self.timers[filepath].cancel()

        # schedule new processing
        timer = Timer(
            DEBOUNCE_DELAY,
            self._execute_callback,
            args=[filepath]
        )
        self.timers[filepath] = timer
        timer.start()

        logger.debug(f"Scheduled processing for {filepath}")

    def _execute_callback(self, filepath):
        # remove timer from dict
        self.timers.pop(filepath, None)

        # execute callback
        try:
            self.callback(filepath, self.doc_type)
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")


class FileWatcher:
    # main file watcher service

    def __init__(self, on_document_ready=None, use_kafka=False):
        # initialize file watcher
        # args:
        #     on_document_ready: callback(filepath, doc_type, parsed_text) for new documents
        #     use_kafka: if true, send parsed documents to Kafka
        self.on_document_ready = on_document_ready
        self.use_kafka = use_kafka
        self.observer = Observer()
        self.running = False

        # kafka producer (optional)
        self.producer = None
        if use_kafka:
            self._init_kafka()

        # create directories
        CV_DIR.mkdir(parents=True, exist_ok=True)
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _init_kafka(self):
        # initialize Kafka producer if available
        try:
            from confluent_kafka import Producer
            self.producer = Producer({
                'bootstrap.servers': 'localhost:9092',
                'client.id': 'file_watcher'
            })
            logger.info("Kafka producer initialized")
        except ImportError:
            logger.warning("confluent_kafka not installed, Kafka disabled")
            self.use_kafka = False
        except Exception as e:
            logger.warning(f"Kafka connection failed: {e}")
            self.use_kafka = False

    def process_file(self, filepath, doc_type):
        # process a detected file
        filepath = Path(filepath)
        logger.info(f"Processing {doc_type}: {filepath.name}")

        # parse document
        parsed = parse_document(filepath)
        if not parsed:
            logger.warning(f"Failed to parse {filepath.name}")
            return

        # generate ID
        doc_id = f"{doc_type}_{uuid.uuid4().hex[:8]}"

        # prepare message
        message = {
            'id': doc_id,
            'type': doc_type,
            'text': parsed['text'],
            'filename': parsed['filename'],
            'file_type': parsed['file_type'],
            'word_count': parsed['word_count'],
            'timestamp': datetime.now().isoformat()
        }

        # send to Kafka if enabled
        if self.use_kafka and self.producer:
            topic = f"{doc_type}_ingestion"
            try:
                self.producer.produce(
                    topic=topic,
                    key=doc_id,
                    value=json.dumps(message).encode('utf-8')
                )
                self.producer.flush()
                logger.info(f"Sent to Kafka topic {topic}: {doc_id}")
            except Exception as e:
                logger.error(f"Kafka send failed: {e}")

        # callback
        if self.on_document_ready:
            self.on_document_ready(filepath, doc_type, parsed['text'])

        # move to processed
        dest = PROCESSED_DIR / f"{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
        try:
            filepath.rename(dest)
            logger.info(f"Moved to {dest}")
        except Exception as e:
            logger.warning(f"Could not move file: {e}")

    def start(self):
        # start watching folders
        cv_handler = DebouncedFileHandler('cv', self.process_file)
        job_handler = DebouncedFileHandler('job', self.process_file)

        self.observer.schedule(cv_handler, str(CV_DIR), recursive=False)
        self.observer.schedule(job_handler, str(JOB_DIR), recursive=False)

        self.observer.start()
        self.running = True

        logger.info(f"Watching directories:")
        logger.info(f"  CVs: {CV_DIR.absolute()}")
        logger.info(f"  Jobs: {JOB_DIR.absolute()}")
        logger.info(f"  Processed: {PROCESSED_DIR.absolute()}")
        logger.info(f"  Kafka: {'enabled' if self.use_kafka else 'disabled'}")
        logger.info("Press Ctrl+C to stop")

    def stop(self):
        # stop watching
        self.running = False
        self.observer.stop()
        self.observer.join()
        if self.producer:
            self.producer.flush()
        logger.info("File watcher stopped")

    def wait(self):
        # wait for watcher to finish
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    # run file watcher as standalone service
    import argparse

    parser = argparse.ArgumentParser(description='File watcher for document ingestion')
    parser.add_argument('--kafka', action='store_true', help='Enable Kafka integration')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    args = parser.parse_args()

    if args.test:
        # test mode: process a single file
        print("Test mode: creating sample CV")
        test_file = CV_DIR / "test_resume.txt"
        CV_DIR.mkdir(parents=True, exist_ok=True)

        test_file.write_text('''
        Jane Smith
        Data Scientist

        Experience:
        - Lead Data Scientist at Analytics Corp (2021-present)
        - Data Analyst at DataCo (2019-2021)

        Skills: Python, Machine Learning, SQL, TensorFlow, Spark
        Education: M.S. Data Science, Tech University, 2019
        ''')

        print(f"Created: {test_file}")
        print("Start watcher and observe processing...")
        time.sleep(1)

    # define callback (for demo, just print)
    def on_document(filepath, doc_type, text):
        print(f"\nNew {doc_type.upper()} detected!")
        print(f"File: {filepath}")
        print(f"Text preview: {text[:200]}")
        print()

    # start watcher
    watcher = FileWatcher(
        on_document_ready=on_document,
        use_kafka=args.kafka
    )

    # handle signals
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    watcher.start()
    watcher.wait()


if __name__ == "__main__":
    main()
