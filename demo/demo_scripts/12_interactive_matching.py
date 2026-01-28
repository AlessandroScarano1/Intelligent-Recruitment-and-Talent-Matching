#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CV-Job Matching (Terminal Version)
Job Seeker Mode: Upload CV -> Find matching jobs from 1.3M+ postings

Usage:
    python demo/demo_scripts/12_interactive_matching.py

    Or with a CV file:
    python demo/demo_scripts/12_interactive_matching.py --cv path/to/cv.pdf
"""

import os
import sys
import uuid
import time
import argparse
from pathlib import Path

# find project root
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print("Loading models")

# imports
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# our modules
from demo.scripts.feedback_storage import init_db, log_action, get_action_count
from demo.scripts.document_parser import parse_document


# paths
MODEL_PATH = PROJECT_ROOT / "training" / "output" / "models" / "cv-job-matcher-e5"
INDEX_PATH = PROJECT_ROOT / "training" / "output" / "indexes" / "jobs_full_index.faiss"
IDS_PATH = PROJECT_ROOT / "training" / "output" / "indexes" / "jobs_full_ids.npy"
JOBS_PATH = PROJECT_ROOT / "ingest_job_postings" / "output" / "unified_job_postings" / "unified_jobs.parquet"


def check_files():
    """check required files exist"""
    files = [
        (MODEL_PATH, "Bi-encoder model"),
        (INDEX_PATH, "Faiss index"),
        (IDS_PATH, "Job IDs"),
        (JOBS_PATH, "Jobs parquet")
    ]

    all_ok = True
    for path, name in files:
        if path.exists():
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name}: {path}")
            all_ok = False

    if not all_ok:
        print("\nERROR: Some files missing. Run pipeline notebooks first.")
        sys.exit(1)


def load_models():
    """load ml models and data"""
    global bi_encoder, cross_encoder, jobs_index, job_ids, jobs_df, job_id_to_row

    # bi-encoder
    print("\nLoading bi-encoder")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bi_encoder = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
        model_kwargs={"torch_dtype": torch.float16}
    )
    print(f"  Bi-encoder loaded on {device}")

    # cross-encoder
    print("Loading cross-encoder")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)
    print(f"  Cross-encoder loaded")

    # faiss index
    print("Loading job index")
    jobs_index = faiss.read_index(str(INDEX_PATH))
    jobs_index.nprobe = 20
    print(f"  Index loaded: {jobs_index.ntotal:,} jobs")

    # job ids and data
    print("Loading job data")
    job_ids = np.load(str(IDS_PATH), allow_pickle=True)
    jobs_df = pd.read_parquet(str(JOBS_PATH))
    job_id_to_row = {jid: idx for idx, jid in enumerate(jobs_df['id'])}
    print(f"  Job data loaded: {len(jobs_df):,} rows")

    # feedback db
    init_db()
    print(f"  Feedback actions logged: {get_action_count()}")


def find_matches(cv_text, top_k=50):
    """find top-k job matches using bi-encoder"""
    prefix = "query: "
    clean_text = cv_text.replace("query: ", "").replace("Query: ", "").replace("passage: ", "")
    prefixed_text = prefix + clean_text

    query_emb = bi_encoder.encode(
        [prefixed_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities, indices = jobs_index.search(query_emb, top_k)

    matches = []
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        job_id = job_ids[idx]
        if job_id in job_id_to_row:
            row_idx = job_id_to_row[job_id]
            job_row = jobs_df.iloc[row_idx]
        else:
            job_row = jobs_df.iloc[idx]

        matches.append({
            'rank': rank,
            'job_id': str(job_id),
            'bi_score': float(sim),
            'title': job_row.get('job_title', 'Unknown Title'),
            'company': job_row.get('company', 'Unknown Company'),
            'location': job_row.get('job_location', 'Unknown'),
            'skills': job_row.get('skills', ''),
            'seniority': job_row.get('seniority', ''),
            'text': job_row.get('embedding_text', '')
        })

    return matches


def rerank_matches(cv_text, matches, top_k=30):
    """rerank matches using cross-encoder"""
    clean_query = cv_text.replace("query: ", "").replace("Query: ", "").replace("passage: ", "")

    pairs = []
    for m in matches:
        doc_text = m['text'].replace("passage: ", "")
        pairs.append((clean_query, doc_text))

    cross_scores = cross_encoder.predict(pairs, batch_size=128)

    for m, score in zip(matches, cross_scores):
        m['cross_score'] = float(score)

    reranked = sorted(matches, key=lambda x: x['cross_score'], reverse=True)
    return reranked[:top_k]


def display_match(match, idx):
    """display a single match in terminal"""
    score = match['cross_score']

    # score indicator
    if score > 5:
        indicator = "[GREAT]"
    elif score > 0:
        indicator = "[GOOD] "
    else:
        indicator = "[WEAK] "

    print(f"\n{'='*60}")
    print(f"#{idx+1} {indicator} {match['title']}")
    print(f"{'='*60}")
    print(f"Company:  {match['company']}")
    print(f"Location: {match['location']}")
    print(f"Level:    {match['seniority']}")
    print(f"Score:    {score:.2f} (bi-encoder: {match['bi_score']:.4f})")

    skills = match.get('skills', '')
    if skills:
        if isinstance(skills, list):
            skills_str = ', '.join(skills[:8])
        else:
            skills_str = ', '.join(str(skills).split(',')[:8])
        print(f"Skills:   {skills_str}")

    # preview
    preview = match['text'][:300].replace('passage: ', '')
    print(f"\nPreview: {preview}")


def display_full_job(match):
    """display full job details"""
    print("\n" + "="*60)
    print("FULL JOB DETAILS")
    print("="*60)

    print(f"\nTitle:    {match['title']}")
    print(f"Company:  {match['company']}")
    print(f"Location: {match['location']}")

    # seniority description
    seniority_map = {
        'intern': 'Intern level, entry position',
        'junior': 'Junior level, 0-2 years experience',
        'mid': 'Mid-level, 3-5 years experience',
        'senior': 'Senior level, 5+ years experience',
        'lead': 'Lead level, 7+ years with leadership',
        'principal': 'Principal level, expert'
    }
    seniority_desc = seniority_map.get(match.get('seniority', 'mid'), match.get('seniority', ''))
    print(f"Level:    {seniority_desc}")
    print(f"Score:    {match['cross_score']:.2f}")

    # skills
    skills = match.get('skills', '')
    if skills:
        if isinstance(skills, list):
            skills_str = ', '.join(skills)
        else:
            skills_str = str(skills)
        print(f"\nRequired Skills:")
        print(f"  {skills_str}")

    # what the model sees
    print(f"\nWhat the model sees:")
    print(f"  {match['text']}")

    print("="*60)


def get_cv_text():
    """get CV text from user input or file"""
    print("\n" + "="*60)
    print("ENTER YOUR CV")
    print("="*60)
    print("\nOptions:")
    print("  1. Paste CV text directly")
    print("  2. Enter path to CV file (PDF/DOCX/TXT)")
    print("  3. Use sample CV")
    print("  q. Quit")

    choice = input("\nYour choice (1/2/3/q): ").strip().lower()

    if choice == 'q':
        return None

    if choice == '3':
        # sample cv
        return """Senior Python Developer with 8 years of experience in Django, PostgreSQL,
and AWS. Led teams of 5+ engineers. Expert in microservices, REST APIs,
and CI/CD pipelines. Strong background in machine learning and data
engineering. Looking for Staff Engineer or Lead roles."""

    if choice == '2':
        # file path
        filepath = input("Enter file path: ").strip()
        if not filepath:
            print("No path entered.")
            return get_cv_text()

        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return get_cv_text()

        print(f"Parsing {filepath.name}")
        parsed = parse_document(filepath)
        if parsed and parsed.get('text'):
            print(f"Parsed: {parsed['word_count']} words")
            return parsed['text']
        else:
            print("Could not parse file. Try pasting text instead.")
            return get_cv_text()

    # default: paste text
    print("\nPaste your CV text below (press Enter twice when done):")
    lines = []
    empty_count = 0
    while True:
        line = input()
        if line == '':
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append('')
        else:
            empty_count = 0
            lines.append(line)

    text = '\n'.join(lines).strip()
    if not text:
        print("No text entered.")
        return get_cv_text()

    return text


def handle_feedback(match, action, session_id, cv_text, cv_id):
    """log user feedback and show confirmation"""
    log_action(
        session_id=session_id,
        role='job_seeker',
        doc_id=cv_id,
        match_id=match['job_id'],
        action=action,
        similarity=match['cross_score'],
        cv_text=cv_text[:2000],
        job_text=match['text'][:2000]
    )

    messages = {
        'apply': 'Applied! Good luck!',
        'save': 'Saved for later.',
        'skip': 'Skipped.',
        'not_interested': 'Marked as not interested.',
        'view_full': 'Showing full job details'
    }

    print(f"\n  >> {messages.get(action, 'Action recorded.')}")
    print(f"  >> Total feedback: {get_action_count()}")


def review_matches(matches, session_id, cv_text, cv_id, page_size=10):
    """interactive loop to review matches with pagination"""
    total = len(matches)
    current_page = 0
    total_pages = (total + page_size - 1) // page_size

    while True:
        start = current_page * page_size
        end = min(start + page_size, total)

        print("\n" + "="*60)
        print(f"JOB MATCHES - Page {current_page + 1}/{total_pages} (showing {start+1}-{end} of {total})")
        print("="*60)
        print("\nActions: [a]pply, [s]ave, [v]iew full, s[k]ip, [n]ot interested")
        print("Navigation: [enter] next job, [number] jump to, [p]rev page, [x]next page, [f]inish, [q]uit")

        idx = start
        while idx < end:
            match = matches[idx]
            display_match(match, idx)

            prompt = f"\nJob {idx+1}/{total} - Action (a/s/v/k/n) or p/x/f/q: "
            action = input(prompt).strip().lower()

            if action == 'q':
                print("\nQuitting")
                return False

            if action == 'f':
                print("\nFinishing review. Thank you for your feedback!")
                return True

            if action == 'p':
                # previous page
                if current_page > 0:
                    current_page -= 1
                    break
                else:
                    print("Already on first page")
                    continue

            if action == 'x':
                # next page
                if current_page < total_pages - 1:
                    current_page += 1
                    break
                else:
                    print("Already on last page")
                    continue

            if action == '' or action == 'enter':
                idx += 1
                if idx >= end and current_page < total_pages - 1:
                    current_page += 1
                    break
                continue

            if action.isdigit():
                new_idx = int(action) - 1
                if 0 <= new_idx < total:
                    # switch to correct page
                    current_page = new_idx // page_size
                    idx = new_idx
                    if idx < start or idx >= end:
                        break  # will restart with new page
                else:
                    print(f"Invalid number. Use 1-{total}")
                continue

            # feedback actions
            action_map = {
                'a': 'apply',
                's': 'save',
                'v': 'view_full',
                'k': 'skip',
                'n': 'not_interested'
            }

            if action in action_map:
                feedback_action = action_map[action]
                handle_feedback(match, feedback_action, session_id, cv_text, cv_id)

                # show full job if requested
                if feedback_action == 'view_full':
                    display_full_job(match)
                    input("Press Enter to continue")
                else:
                    idx += 1
                    if idx >= end and current_page < total_pages - 1:
                        current_page += 1
                        break
            else:
                print("Unknown action. Use: a, s, v, k, n, p, x, f, q, or Enter")

        # check if we've gone through all pages
        if idx >= total:
            print("\nReached end of matches. Thank you for your feedback!")
            return True


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(description='Interactive CV-Job Matching')
    parser.add_argument('--cv', type=str, help='Path to CV file (PDF/DOCX/TXT)')
    args = parser.parse_args()

    print("="*60)
    print("INTERACTIVE CV-JOB MATCHING")
    print("Job Seeker Mode: Upload CV -> Find matching jobs")
    print("="*60)

    # check files
    print("\nChecking required files")
    check_files()

    # load models
    load_models()

    # session
    session_id = str(uuid.uuid4())[:8]
    print(f"\nSession ID: {session_id}")

    while True:
        # get cv text
        if args.cv:
            filepath = Path(args.cv)
            if filepath.exists():
                print(f"\nParsing {filepath.name}")
                parsed = parse_document(filepath)
                if parsed and parsed.get('text'):
                    cv_text = parsed['text']
                    print(f"Parsed: {parsed['word_count']} words")
                else:
                    print("Could not parse file.")
                    cv_text = get_cv_text()
            else:
                print(f"File not found: {args.cv}")
                cv_text = get_cv_text()
            args.cv = None  # only use once
        else:
            cv_text = get_cv_text()

        if cv_text is None:
            print("\nGoodbye!")
            break

        cv_id = f"cv_{uuid.uuid4().hex[:8]}"

        # search
        print(f"\nSearching {jobs_index.ntotal:,} jobs")
        start = time.time()
        candidates = find_matches(cv_text, top_k=50)
        bi_time = time.time() - start
        print(f"  Found {len(candidates)} candidates in {bi_time*1000:.0f}ms")

        print("Reranking with cross-encoder")
        start = time.time()
        matches = rerank_matches(cv_text, candidates, top_k=30)
        cross_time = time.time() - start
        print(f"  Reranked to top {len(matches)} in {cross_time*1000:.0f}ms")

        # review with pagination
        continue_session = review_matches(matches, session_id, cv_text, cv_id, page_size=10)

        if not continue_session:
            break

        # ask if want to search again
        again = input("\nSearch with another CV? (y/n): ").strip().lower()
        if again != 'y':
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
