# %%
# Feedback Loop & Pipeline Refresh
# 1. Update Skill Dictionary from Feedback
# 2. Retrain Bi-Encoder Model
# 3. Refresh Embeddings (Script 06)
# 4. Rebuild Index (Script 10)

import os
import sys
import sqlite3
import pandas as pd
import spacy
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Setup Paths
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))
else:
    PROJECT_ROOT = cwd

sys.path.insert(0, PROJECT_ROOT)

from demo.scripts.feedback_storage import split_db_path, DB_PATH
from demo.scripts.model_retrainer import retrain_from_feedback

# %%
def update_skill_dictionary():
    print("Updating Skill Dictionary")
    
    # Load existing dictionary
    dict_path = os.path.join(PROJECT_ROOT, 'ingest_job_postings', 'output', 'skill_dictionary', 'all_skills')
    if not os.path.exists(dict_path):
        print("  Error: Skill dictionary not found.")
        return

    try:
        current_dict_df = pd.read_parquet(dict_path)
        print(f"  Current dictionary size: {len(current_dict_df)} skills")
    except Exception as e:
        print(f"  Error reading dictionary: {e}")
        return

    # Load feedback CVs
    conn = sqlite3.connect(DB_PATH)
    cv_texts = pd.read_sql_query("SELECT cv_text FROM user_actions WHERE cv_text IS NOT NULL", conn)
    conn.close()

    if cv_texts.empty:
        print("  No feedback CVs found. Skipping.")
        return

    print(f"Analyzing {len(cv_texts)} CVs for new skills")
    
    # NLP extraction
    nlp = spacy.load("en_core_web_sm")
    new_candidates = []
    
    # Stopwords/Generic filters (basic list)
    STOP_TERMS = {'experience', 'years', 'skills', 'work', 'team', 'project'}

    for text in cv_texts['cv_text']:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.lower().strip()
            if len(clean_chunk) > 2 and clean_chunk not in STOP_TERMS:
                new_candidates.append(clean_chunk)

    # Filter candidates
    candidate_counts = pd.Series(new_candidates).value_counts()
    # Only keep terms appearing at least twice to avoid noise
    valid_new = candidate_counts[candidate_counts >= 2].index.tolist()

    # Add to dictionary
    existing_skills = set(current_dict_df['skill'].str.lower())
    really_new = [s for s in valid_new if s not in existing_skills]

    print(f"  Found {len(really_new)} potential new skills")
    
    if really_new:
        new_df = pd.DataFrame({'skill': really_new, 'count': 1}) # Init count as 1
        updated_df = pd.concat([current_dict_df, new_df], ignore_index=True)
        updated_df.to_parquet(dict_path)
        print(f"  Updated dictionary saved. New size: {len(updated_df)}")
    else:
        print("  No new unique skills added.")

# %%
def run_retraining(epochs=2):
    print("\nFine-Tuning Model")
    # calls the existing logic from model_retrainer.py
    
    result = retrain_from_feedback(threshold=1, epochs=epochs, learning_rate=1e-5)
    
    if result['success']:
        print(f"  Success, new model saved at: {result['model_path']}")
        return True, result['model_path']
    else:
        print(f"  Skipping training: {result.get('reason')}")
        return False, None

# %%
def refresh_pipeline(quick_mode=False):
    print("\nStep 3: Refreshing Embeddings (Script 06)...")
    
    cmd_06 = [
        "python", 
        "ingest_job_postings/scripts/06_generate_embeddings.py"
    ]
    if quick_mode:
        cmd_06.append("--quick")
        
    process_06 = subprocess.run(cmd_06, cwd=PROJECT_ROOT)
    if process_06.returncode != 0:
        print("  Error in Script 06. Stopping.")
        return False

    print("\nStep 4: Rebuilding Index (Script 10)...")
    
    cmd_10 = [
        "python", 
        "training/scripts/10_build_job_index.py"
    ]
    if quick_mode:
        cmd_10.append("--quick")

    process_10 = subprocess.run(cmd_10, cwd=PROJECT_ROOT)
    if process_10.returncode != 0:
        print("  Error in Script 10. Stopping.")
        return False
        
    return True

# %%
def main():
    parser = argparse.ArgumentParser(description="Retrain Model & Refresh Pipeline")
    parser.add_argument("--quick", action="store_true", help="Use quick mode for scripts")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()

    print("Feedback Retraining & Pipeline Refresh")
    
    # update dictionary
    update_skill_dictionary()
    
    # retrain model
    is_trained, model_path = run_retraining(epochs=args.epochs)
    
    # refresh pipeline (only if we trained a new model OR checking pipeline integrity)
    # but even if we don't retrain (not enough data), we might want to refresh if new skills added
    # for now, let's refresh only if we successfully trained OR if the user explicitly wants to
    
    if is_trained:
        print("\nNew model available. Updating full pipeline")
        
        # copy best model to best path so scripts 06 and 10 pick it up
        import shutil
        best_dst = os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5-best')
        if os.path.exists(best_dst):
            shutil.rmtree(best_dst)
        shutil.copytree(model_path, best_dst)
        print(f"  Promoted to best model: {best_dst}")
        
        # Run scripts
        success = refresh_pipeline(quick_mode=args.quick)
        
        if success:
            print("\nRefresh Complete")
            print("System is now running with the new model and updated index.")
    else:
        print("\nNo new model trained. Skipping pipeline refresh.")

if __name__ == "__main__":
    main()
