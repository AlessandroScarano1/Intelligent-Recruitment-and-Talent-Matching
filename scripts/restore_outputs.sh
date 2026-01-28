#!/bin/bash
# restore_outputs.sh, restore output folders from backup
# usage: ./scripts/restore_outputs.sh

set -e

BACKUP_DIR="output_backup"

echo "Output Restore Script"
echo ""

# check if backup exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: Backup folder not found at $BACKUP_DIR"
    echo "Run backup_outputs.sh first to create a backup."
    exit 1
fi

echo "Found backup at: $BACKUP_DIR"
echo "Contents:"
ls -la "$BACKUP_DIR"
echo ""

read -p "Restore outputs from backup? This will overwrite current outputs. (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# clear current outputs
echo ""
echo "Clearing current outputs"
rm -rf ingest_job_postings/output
rm -rf training/output
rm -rf demo/data/feedback

# restore from backup
echo "Restoring from backup"

if [ -d "$BACKUP_DIR/ingest_job_postings_output" ]; then
    cp -r "$BACKUP_DIR/ingest_job_postings_output" ingest_job_postings/output
    echo "  [OK] ingest_job_postings/output"
else
    mkdir -p ingest_job_postings/output
    echo "  [SKIP] ingest_job_postings/output (no backup found)"
fi

if [ -d "$BACKUP_DIR/training_output" ]; then
    cp -r "$BACKUP_DIR/training_output" training/output
    echo "  [OK] training/output"
else
    mkdir -p training/output
    echo "  [SKIP] training/output (no backup found)"
fi

if [ -d "$BACKUP_DIR/demo_feedback" ]; then
    mkdir -p demo/data
    cp -r "$BACKUP_DIR/demo_feedback" demo/data/feedback
    echo "  [OK] demo/data/feedback"
else
    mkdir -p demo/data/feedback
    echo "  [SKIP] demo/data/feedback (no backup found)"
fi

echo ""
echo "Restore Complete"
echo ""

# Ask if user wants to delete backup
read -p "Delete backup folder? (y/N): " delete_confirm
if [ "$delete_confirm" = "y" ] || [ "$delete_confirm" = "Y" ]; then
    rm -rf "$BACKUP_DIR"
    echo "Backup deleted."
else
    echo "Backup kept at: $BACKUP_DIR"
fi
