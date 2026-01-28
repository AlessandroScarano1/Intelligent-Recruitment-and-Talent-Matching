#!/bin/bash




# !! backup_outputs.sh, backup and clear output folders for fresh pipeline testing
# Usage: ./scripts/backup_outputs.sh

set -e

BACKUP_DIR="output_backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Output Backup Script"
echo ""

# Check if backup already exists
if [ -d "$BACKUP_DIR" ]; then
    echo "WARNING: Backup folder already exists at $BACKUP_DIR"
    echo "Contents:"
    ls -la "$BACKUP_DIR"
    echo ""
    read -p "Overwrite existing backup? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted. Use restore_outputs.sh to restore existing backup first."
        exit 1
    fi
    rm -rf "$BACKUP_DIR"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
echo "Created backup directory: $BACKUP_DIR"

# Backup each output folder
echo ""
echo "Backing up output folders"

if [ -d "ingest_job_postings/output" ]; then
    cp -r ingest_job_postings/output "$BACKUP_DIR/ingest_job_postings_output"
    echo "  [OK] ingest_job_postings/output"
else
    echo "  [SKIP] ingest_job_postings/output (not found)"
fi

if [ -d "training/output" ]; then
    cp -r training/output "$BACKUP_DIR/training_output"
    echo "  [OK] training/output"
else
    echo "  [SKIP] training/output (not found)"
fi

if [ -d "ingest_cv/output" ]; then
    cp -r ingest_cv/output "$BACKUP_DIR/ingest_cv_output"
    echo "  [OK] ingest_cv/output"
else
    echo "  [SKIP] ingest_cv/output (not found)"
fi

if [ -d "demo/data/feedback" ]; then
    cp -r demo/data/feedback "$BACKUP_DIR/demo_feedback"
    echo "  [OK] demo/data/feedback"
else
    echo "  [SKIP] demo/data/feedback (not found)"
fi

# show backup size
echo ""
echo "Backup size:"
du -sh "$BACKUP_DIR"

# delete original outputs
echo ""
echo "Clearing output folders for fresh testing"

rm -rf ingest_job_postings/output/*
rm -rf training/output/*
# keep ingest_cv/output
# rm -rf ingest_cv/output/*
rm -rf demo/data/feedback/*

# Recreate .gitkeep files
touch ingest_job_postings/output/.gitkeep 2>/dev/null || true
touch training/output/.gitkeep 2>/dev/null || true
touch demo/data/feedback/.gitkeep 2>/dev/null || true

echo ""
echo "Backup Completed"
echo "Outputs backed up to: $BACKUP_DIR"
echo "Output folders cleared (ready for fresh pipeline run)"
echo ""
echo "NOTE: ingest_cv/output was NOT cleared"
echo ""
echo "To restore: ./scripts/restore_outputs.sh"
