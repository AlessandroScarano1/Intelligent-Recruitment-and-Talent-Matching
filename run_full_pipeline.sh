#!/bin/bash
# Full pipeline execution script
# Runs CV ingestion -> Job ingestion -> Training -> Index building

set -e  # exit on error

PROJECT_ROOT="/home/developer/project"
cd "$PROJECT_ROOT"

# create output directory and log file
OUTPUT_DIR="$PROJECT_ROOT/full_run_output"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/full_pipeline_${TIMESTAMP}.log"

# redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "FULL PIPELINE EXECUTION"
echo ""
echo "Pipeline order:"
echo "  1. CV Ingestion (scripts 01-05)"
echo "  2. Job Ingestion (scripts 01-06)"
echo "  3. Training (scripts 07-10)"
echo ""
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# function to run a script with timing
run_script() {
    local script_path=$1
    local script_name=$(basename "$script_path")

    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $script_name"
    echo "------------------------------------------------------------"

    start_time=$(date +%s)

    if python "$script_path"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo "[OK] $script_name completed in ${duration}s"
    else
        echo ""
        echo "[FAILED] $script_name failed"
        exit 1
    fi
}

# PHASE 1: CV INGESTION PIPELINE

echo ""
echo "PHASE 1: CV INGESTION PIPELINE"

run_script "ingest_cv/scripts/01_kafka_batch_load_cv.py"
run_script "ingest_cv/scripts/02_normalize_fields.py"
run_script "ingest_cv/scripts/03_aggregate_skills.py"
run_script "ingest_cv/scripts/04_embedding_strings.py"
run_script "ingest_cv/scripts/05_cv_splits.py"

echo ""
echo "[COMPLETE] CV ingestion pipeline finished"
echo ""

# PHASE 2: JOB INGESTION PIPELINE

echo ""
echo "PHASE 2: JOB INGESTION PIPELINE"

run_script "ingest_job_postings/scripts/01_kafka_batch_load.py"
run_script "ingest_job_postings/scripts/02_spark_processing.py"
run_script "ingest_job_postings/scripts/03_nlp_extraction.py"
run_script "ingest_job_postings/scripts/04_embedding_strings.py"
run_script "ingest_job_postings/scripts/05_final_embedding_output.py"
run_script "ingest_job_postings/scripts/06_generate_embeddings.py"

echo ""
echo "[COMPLETE] Job ingestion pipeline finished"
echo ""

# PHASE 3: TRAINING PIPELINE

echo ""
echo "PHASE 3: TRAINING PIPELINE"

run_script "training/scripts/07_train_val_test_splits.py"
run_script "training/scripts/08_training_data_prep.py"
run_script "training/scripts/09_biencoder_training.py"
run_script "training/scripts/10_build_job_index.py"

echo ""
echo "[COMPLETE] Training pipeline finished"
echo ""

# SUMMARY

echo ""
echo "PIPELINE EXECUTION COMPLETE"
echo ""
echo "All phases completed successfully:"
echo "  [✓] CV Ingestion (5 scripts)"
echo "  [✓] Job Ingestion (6 scripts)"
echo "  [✓] Training (4 scripts)"
echo ""
echo "End time: $(date)"
echo ""
echo "Output locations:"
echo "  - CV data: ingest_cv/output/"
echo "  - Job data: ingest_job_postings/output/"
echo "  - Models: training/output/models/"
echo "  - Indexes: training/output/indexes/"
echo "  - Log file: $LOG_FILE"
echo ""
echo "Ready to run demo scripts:"
echo "  python demo/demo_scripts/11_retrieval_demo.py"
echo "  python demo/demo_scripts/12_interactive_matching.py"
echo "  python demo/demo_scripts/13_recruiter_matching.py"
echo ""