
import os
import sys
import time
import importlib

def check_step(name, func):
    print(f"\nChecking {name}...")
    try:
        func()
        print(f"✅ {name}: OK")
        return True
    except Exception as e:
        print(f"❌ {name}: FAILED")
        print(f"   Error: {e}")
        return False

def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Torch")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Version: {torch.version.cuda}")

def check_spark():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("TestCheck").master("local[*]").getOrCreate()
    data = [("test", 1)]
    df = spark.createDataFrame(data, ["col1", "col2"])
    if df.count() != 1:
        raise RuntimeError("Spark DF count mismatch")
    spark.stop()

def check_kafka():
    from confluent_kafka import AdminClient
    # Kafka is named 'kafka-broker' in docker-compose, port 9092
    conf = {'bootstrap.servers': 'kafka-broker:29092'} 
    admin = AdminClient(conf)
    topics = admin.list_topics(timeout=10)
    if not topics.topics:
        raise RuntimeError("Connected to Kafka but no topics found (or timeout)")
    print(f"   Topics found: {list(topics.topics.keys())}")

def check_models():
    # Check if we can load the sentence transformer (checks cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('intfloat/e5-base-v2')
    print("   e5-base-v2 loaded successfully")

def check_dirs():
    required = [
        "ingest_job_postings/output", 
        "ingest_cv/output", 
        "training/output",
        "demo/notebooks"
    ]
    for d in required:
        if not os.path.isdir(d):
            print(f"   Creating missing dir: {d}")
            os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    print("="*50)
    print("ENVIRONMENT VALIDATION SCRIPT")
    print("="*50)
    
    checks = [
        ("Directory Structure", check_dirs),
        ("CUDA/GPU", check_cuda),
        ("Spark Integration", check_spark),
        ("Kafka Connection", check_kafka),
        ("Model Logic", check_models),
    ]
    
    success = True
    for name, func in checks:
        if not check_step(name, func):
            success = False
            
    print("\n" + "="*50)
    if success:
        print("🎉 ALL SYSTEMS GO! Ready for Demo.")
    else:
        print("⚠️ SOME CHECKS FAILED. See output above.")
    print("="*50)
