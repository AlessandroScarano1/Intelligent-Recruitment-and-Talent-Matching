#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Recruiter Matching (Terminal Version)
Recruiter Mode: Upload job posting -> Find matching CVs from 7,299 candidates

Usage:
    python demo/demo_scripts/13_recruiter_matching.py

    Or with a job file:
    python demo/demo_scripts/13_recruiter_matching.py --job path/to/job.txt
"""

import os
import sys
import re
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
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, CrossEncoder

# our modules
from demo.scripts.feedback_storage import init_db, log_action, get_action_count
from demo.scripts.document_parser import parse_document


# paths
MODEL_PATH = PROJECT_ROOT / "training" / "output" / "models" / "cv-job-matcher-e5"
CV_INDEX_PATH = PROJECT_ROOT / "training" / "output" / "indexes" / "cvs_index.faiss"
CV_DATA_PATH = PROJECT_ROOT / "ingest_cv" / "output" / "cv_query_text.parquet"
SKILL_DICT_PATH = PROJECT_ROOT / "ingest_job_postings" / "output" / "skill_dictionary" / "all_skills"


# global variables
bi_encoder = None
cross_encoder = None
cvs_index = None
cvs_df = None
nlp = None
matcher = None
skill_set = None


def check_files():
    """check required files exist"""
    files = [
        (MODEL_PATH, "Bi-encoder model"),
        (CV_INDEX_PATH, "CV index"),
        (CV_DATA_PATH, "CV data"),
        (SKILL_DICT_PATH, "Skill dictionary")
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
    global bi_encoder, cross_encoder, cvs_index, cvs_df, nlp, matcher, skill_set

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
    print("Loading CV index")
    cvs_index = faiss.read_index(str(CV_INDEX_PATH))
    if hasattr(cvs_index, 'nprobe'):
        cvs_index.nprobe = 20
    print(f"  Index loaded: {cvs_index.ntotal:,} CVs")

    # cv data
    print("Loading CV data")
    cvs_df = pd.read_parquet(str(CV_DATA_PATH))
    print(f"  CV data loaded: {len(cvs_df):,} rows")

    # skill dictionary and matcher
    print("Loading skill dictionary")
    skills_df = pd.read_parquet(str(SKILL_DICT_PATH))
    skill_col = skills_df.columns[0]
    skill_set = set(skills_df[skill_col].str.lower().str.strip().tolist())
    print(f"  Loaded {len(skill_set):,} skills")

    print("Building PhraseMatcher")
    nlp = spacy.blank('en')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    skill_list = list(skill_set)
    batch_size = 10000
    for i in range(0, len(skill_list), batch_size):
        batch = skill_list[i:i+batch_size]
        patterns = [nlp.make_doc(skill) for skill in batch]
        matcher.add(f"SKILLS_{i}", patterns)
    print(f"  PhraseMatcher ready")

    # feedback db
    init_db()
    print(f"  Feedback actions logged: {get_action_count()}")


def extract_skills_from_job(job_text):
    """extract skills from job posting using PhraseMatcher"""
    if not job_text or not str(job_text).strip():
        return []

    doc = nlp(job_text.lower())
    matches = matcher(doc)

    skills = []
    for match_id, start, end in matches:
        skill = doc[start:end].text
        if len(skill) > 1:
            skills.append(skill)

    seen = set()
    unique_skills = []
    for s in skills:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique_skills.append(s)

    return unique_skills


def extract_job_fields(job_text):
    """extract structured fields from job posting"""
    fields = {
        'title': '',
        'company': '',
        'location': '',
        'skills': [],
        'experience_years': '',
        'salary_min': None,
        'salary_max': None,
        'remote_status': '',
        'seniority': 'mid'
    }

    if not job_text:
        return fields

    fields['skills'] = extract_skills_from_job(job_text)

    # title
    lines = job_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith(('company', 'location', 'salary', 'type')):
            fields['title'] = line[:100]
            break

    # company
    company_match = re.search(r'company[:\s]+([^\n]+)', job_text, re.I)
    if company_match:
        fields['company'] = company_match.group(1).strip()[:100]

    # location
    location_match = re.search(r'location[:\s]+([^\n]+)', job_text, re.I)
    if location_match:
        fields['location'] = location_match.group(1).strip()[:100]

    # salary
    salary_match = re.search(r'\$?([\d,]+)\s*[-\u2013]\s*\$?([\d,]+)', job_text)
    if salary_match:
        try:
            fields['salary_min'] = int(salary_match.group(1).replace(',', ''))
            fields['salary_max'] = int(salary_match.group(2).replace(',', ''))
        except ValueError:
            pass

    # experience
    exp_match = re.search(r'(\d+)\+?\s*years?', job_text, re.I)
    if exp_match:
        fields['experience_years'] = exp_match.group(1) + '+'

    # remote
    if re.search(r'\bremote\b', job_text, re.I):
        fields['remote_status'] = 'remote'
    elif re.search(r'\bhybrid\b', job_text, re.I):
        fields['remote_status'] = 'hybrid'
    else:
        fields['remote_status'] = 'onsite'

    # seniority
    title_lower = fields['title'].lower()
    if any(w in title_lower for w in ['intern', 'internship', 'trainee']):
        fields['seniority'] = 'intern'
    elif any(w in title_lower for w in ['principal', 'staff', 'distinguished']):
        fields['seniority'] = 'principal'
    elif any(w in title_lower for w in ['lead', 'head of', 'director', 'vp', 'chief']):
        fields['seniority'] = 'lead'
    elif any(w in title_lower for w in ['senior', 'sr.', 'sr ']):
        fields['seniority'] = 'senior'
    elif any(w in title_lower for w in ['junior', 'jr.', 'jr ', 'entry']):
        fields['seniority'] = 'junior'
    else:
        fields['seniority'] = 'mid'

    return fields


def build_job_embedding_string(fields):
    """build embedding string from extracted fields"""
    parts = []

    title = fields.get('title', 'Unknown Position')
    company = fields.get('company', 'a company')
    location = fields.get('location', '')

    role_part = f"Role of {title} at {company}"
    if location:
        role_part += f" in {location}"
    parts.append(role_part + ".")

    skills = fields.get('skills', [])
    if skills:
        skills_str = ', '.join(skills[:10])
        parts.append(f"Required skills: {skills_str}.")

    seniority = fields.get('seniority', 'mid')
    seniority_map = {
        'intern': 'Intern level, entry position',
        'junior': 'Junior level, 0-2 years experience',
        'mid': 'Mid-level, 3-5 years experience',
        'senior': 'Senior level, 5+ years experience',
        'lead': 'Lead level, 7+ years experience with leadership',
        'principal': 'Principal level, expert with technical leadership'
    }
    level_desc = seniority_map.get(seniority, seniority)
    parts.append(f"Experience level: {level_desc}.")

    salary_min = fields.get('salary_min')
    salary_max = fields.get('salary_max')
    if salary_min and salary_max:
        parts.append(f"Salary range: ${salary_min:,} to ${salary_max:,}.")

    remote = fields.get('remote_status', '')
    if remote:
        remote_map = {
            'remote': 'Remote work available',
            'hybrid': 'Hybrid work, partially remote',
            'onsite': 'Onsite work'
        }
        work_type = remote_map.get(remote, remote)
        parts.append(f"Work type: {work_type}.")

    return "passage: " + ' '.join(parts)


def find_matching_cvs(job_text, top_k=50):
    """find top-k matching CVs using bi-encoder"""
    fields = extract_job_fields(job_text)
    embedding_text = build_job_embedding_string(fields)

    job_embedding = bi_encoder.encode(
        [embedding_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = cvs_index.search(job_embedding, top_k)

    matches = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx >= 0 and idx < len(cvs_df):
            cv_row = cvs_df.iloc[idx]
            matches.append({
                'rank': rank,
                'cv_id': cv_row['id'],
                'bi_score': float(dist),
                'text': cv_row['text'],
                'idx': int(idx)
            })

    return matches, fields


def rerank_cvs(job_text, matches, top_k=30):
    """rerank CVs with cross-encoder"""
    if not matches:
        return []

    clean_job = job_text.replace("passage: ", "").replace("query: ", "")

    pairs = []
    for m in matches:
        clean_cv = m['text'].replace("query: ", "").replace("Query: ", "")
        pairs.append((clean_job, clean_cv))

    cross_scores = cross_encoder.predict(pairs, batch_size=50)

    for m, score in zip(matches, cross_scores):
        m['cross_score'] = float(score)

    reranked = sorted(matches, key=lambda x: x['cross_score'], reverse=True)
    return reranked[:top_k]


def display_cv_match(match, idx, job_skills):
    """display a single CV match in terminal"""
    score = match['cross_score']

    # score indicator
    if score > 5:
        indicator = "[GREAT]"
    elif score > 0:
        indicator = "[GOOD] "
    else:
        indicator = "[WEAK] "

    print(f"\n{'='*60}")
    print(f"#{idx+1} {indicator} CV: {match['cv_id']}")
    print(f"{'='*60}")
    print(f"Score: {score:.2f} (bi-encoder: {match['bi_score']:.4f})")

    # matching skills
    cv_text_lower = match['text'].lower()
    matching_skills = [s for s in job_skills if s.lower() in cv_text_lower]
    if matching_skills:
        skills_str = ', '.join(matching_skills[:8])
        print(f"Matching skills: {skills_str}")

    # preview
    preview = match['text'][:400].replace('query: ', '').replace('Query: ', '')
    print(f"\nPreview: {preview}")


def display_full_cv(match):
    """display full CV details"""
    print("\n" + "="*60)
    print("FULL CV DETAILS")
    print("="*60)

    print(f"\nCV ID: {match['cv_id']}")
    print(f"Score: {match['cross_score']:.2f} (bi-encoder: {match['bi_score']:.4f})")

    # full text
    full_text = match['text'].replace('query: ', '').replace('Query: ', '')
    print(f"\nFull CV Text:")
    print(full_text)

    print("="*60)


def get_job_text():
    """get job text from user input or file"""
    print("\n" + "="*60)
    print("ENTER JOB POSTING")
    print("="*60)
    print("\nOptions:")
    print("  1. Paste job posting text directly")
    print("  2. Enter path to job file (PDF/DOCX/TXT)")
    print("  3. Use sample job posting")
    print("  q. Quit")

    choice = input("\nYour choice (1/2/3/q): ").strip().lower()

    if choice == 'q':
        return None

    if choice == '3':
        return """Senior Python Developer

Company: TechCorp Inc.
Location: San Francisco, CA (Remote OK)
Salary: $150,000 - $180,000

Requirements:
- 5+ years of Python development
- Django or FastAPI experience
- PostgreSQL and AWS knowledge
- Experience with microservices architecture
- Strong communication skills"""

    if choice == '2':
        filepath = input("Enter file path: ").strip()
        if not filepath:
            print("No path entered.")
            return get_job_text()

        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return get_job_text()

        print(f"Parsing {filepath.name}")
        parsed = parse_document(filepath)
        if parsed and parsed.get('text'):
            print(f"Parsed: {parsed['word_count']} words")
            return parsed['text']
        else:
            print("Could not parse file. Try pasting text instead.")
            return get_job_text()

    # default: paste text
    print("\nPaste your job posting below (press Enter twice when done):")
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
        return get_job_text()

    return text


def handle_feedback(match, action, session_id, job_text, job_id):
    """log recruiter feedback and show confirmation"""
    log_action(
        session_id=session_id,
        role='recruiter',
        doc_id=job_id,
        match_id=match['cv_id'],
        action=action,
        similarity=match['cross_score'],
        cv_text=match['text'][:2000],
        job_text=job_text[:2000]
    )

    messages = {
        'contact': 'Marked for contact!',
        'save': 'Added to shortlist.',
        'skip': 'Skipped.',
        'not_interested': 'Marked as not a match.',
        'view_full': 'Showing full CV details'
    }

    print(f"\n  >> {messages.get(action, 'Action recorded.')}")
    print(f"  >> Total feedback: {get_action_count()}")


def review_cvs(matches, session_id, job_text, job_id, job_skills, page_size=10):
    """interactive loop to review CVs with pagination"""
    total = len(matches)
    current_page = 0
    total_pages = (total + page_size - 1) // page_size

    while True:
        start = current_page * page_size
        end = min(start + page_size, total)

        print("\n" + "="*60)
        print(f"MATCHING CVs - Page {current_page + 1}/{total_pages} (showing {start+1}-{end} of {total})")
        print("="*60)
        print("\nActions: [c]ontact, [s]ave, [v]iew full, s[k]ip, [n]ot a match")
        print("Navigation: [enter] next CV, [number] jump to, [p]rev page, [x]next page, [f]inish, [q]uit")

        idx = start
        while idx < end:
            match = matches[idx]
            display_cv_match(match, idx, job_skills)

            prompt = f"\nCV {idx+1}/{total} - Action (c/s/v/k/n) or p/x/f/q: "
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
                'c': 'contact',
                's': 'save',
                'v': 'view_full',
                'k': 'skip',
                'n': 'not_interested'
            }

            if action in action_map:
                feedback_action = action_map[action]
                handle_feedback(match, feedback_action, session_id, job_text, job_id)

                # show full CV if requested
                if feedback_action == 'view_full':
                    display_full_cv(match)
                    input("Press Enter to continue")
                else:
                    idx += 1
                    if idx >= end and current_page < total_pages - 1:
                        current_page += 1
                        break
            else:
                print("Unknown action. Use: c, s, v, k, n, p, x, f, q, or Enter")

        # check if we've gone through all pages
        if idx >= total:
            print("\nReached end of CVs. Thank you for your feedback!")
            return True


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(description='Interactive Recruiter Matching')
    parser.add_argument('--job', type=str, help='Path to job file (PDF/DOCX/TXT)')
    args = parser.parse_args()

    print("="*60)
    print("INTERACTIVE RECRUITER MATCHING")
    print("Recruiter Mode: Upload job -> Find matching CVs")
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
        # get job text
        if args.job:
            filepath = Path(args.job)
            if filepath.exists():
                print(f"\nParsing {filepath.name}")
                parsed = parse_document(filepath)
                if parsed and parsed.get('text'):
                    job_text = parsed['text']
                    print(f"Parsed: {parsed['word_count']} words")
                else:
                    print("Could not parse file.")
                    job_text = get_job_text()
            else:
                print(f"File not found: {args.job}")
                job_text = get_job_text()
            args.job = None
        else:
            job_text = get_job_text()

        if job_text is None:
            print("\nGoodbye!")
            break

        job_id = f"job_{uuid.uuid4().hex[:8]}"

        # search
        print(f"\nSearching {cvs_index.ntotal:,} CVs")
        start = time.time()
        candidates, fields = find_matching_cvs(job_text, top_k=50)
        bi_time = time.time() - start
        print(f"  Found {len(candidates)} candidates in {bi_time*1000:.0f}ms")
        print(f"  Extracted {len(fields['skills'])} skills: {fields['skills'][:5]}")

        print("Reranking with cross-encoder")
        start = time.time()
        matches = rerank_cvs(job_text, candidates, top_k=30)
        cross_time = time.time() - start
        print(f"  Reranked to top {len(matches)} in {cross_time*1000:.0f}ms")

        # review with pagination
        continue_session = review_cvs(matches, session_id, job_text, job_id, fields['skills'], page_size=10)

        if not continue_session:
            break

        again = input("\nSearch with another job posting? (y/n): ").strip().lower()
        if again != 'y':
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
