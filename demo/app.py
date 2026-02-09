# CV-Job Matcher Streamlit App
# interactive demo for job seekers and recruiters

import streamlit as st
import os
import sys
import uuid
import time
import math
import tempfile
from pathlib import Path

# project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# imports
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# our modules
from demo.scripts.document_parser import parse_document, detect_document_type
from demo.scripts.feedback_storage import (
    init_db, log_action, get_action_count,
    get_action_summary, ACTION_WEIGHTS
)
from demo.scripts.matching_utils import (
    reformat_cv_for_matching as reformat_cv,
    extract_job_fields, build_job_embedding_string,
    prepare_for_biencoder, strip_prefixes,
    logistic_percentage, parse_cv_summary,
    extract_skills_from_text, get_device
)
from demo.scripts.skill_tracker import track_skills_from_feedback, get_skill_proposals

import re

print(f"Project root: {PROJECT_ROOT}")


# parse_cv_summary is now imported from matching_utils


# reformat_cv_for_matching is now imported from matching_utils as reformat_cv
# uses PhraseMatcher with 22K skills instead of hardcoded 60


# load models and data once using streamlit cache
@st.cache_resource
def load_models():
    # load all ML models and indexes
    # this runs once per server, not per user session

    print("Loading models and indexes")

    # check device
    device = get_device()
    print(f"Using device: {device}")

    # bi-encoder model
    model_path = PROJECT_ROOT / "training" / "output" / "models" / "cv-job-matcher-e5-best"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "training" / "output" / "models" / "cv-job-matcher-e5"

    print(f"Loading bi-encoder from {model_path.name}")
    bi_encoder = SentenceTransformer(
        str(model_path),
        device=device,
        model_kwargs={"dtype": torch.float16}
    )
    print(f"  Bi-encoder loaded on {device}")

    # cross-encoder
    print("Loading cross-encoder")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)
    print("  Cross-encoder loaded")

    # job index - full 1.3M jobs (same as CLI script)
    jobs_index_path = PROJECT_ROOT / "training" / "output" / "indexes" / "jobs_full_index.faiss"
    jobs_ids_path = PROJECT_ROOT / "training" / "output" / "indexes" / "jobs_full_ids.npy"
    print(f"Loading jobs index from {jobs_index_path.name}")
    jobs_index = faiss.read_index(str(jobs_index_path))
    # set nprobe for IVF index
    if hasattr(jobs_index, 'nprobe'):
        jobs_index.nprobe = 20
    print(f"  Jobs index loaded: {jobs_index.ntotal:,} vectors")

    # load job IDs (UUIDs matching faiss vector positions)
    print("Loading job IDs")
    job_ids = np.load(str(jobs_ids_path), allow_pickle=True)
    print(f"  Loaded {len(job_ids):,} job IDs")

    # load full job data
    print("Loading job details")
    jobs_df = pd.read_parquet(
        PROJECT_ROOT / "ingest_job_postings" / "output" / "unified_job_postings" / "unified_jobs.parquet"
    )
    # keep first occurrence of each job ID (some IDs are duplicated in dataset)
    job_id_to_row = {}
    for idx, jid in enumerate(jobs_df['id']):
        if jid not in job_id_to_row:
            job_id_to_row[jid] = idx
    print(f"  Loaded {len(jobs_df):,} job records")

    # load CVs data for recruiter mode
    print("Loading CVs data")
    cvs_df = pd.read_parquet(
        PROJECT_ROOT / "ingest_cv" / "output" / "cv_query_text.parquet",
        columns=['id', 'text']
    )
    cvs_index_path = PROJECT_ROOT / "training" / "output" / "indexes" / "cvs_index.faiss"
    cvs_index = faiss.read_index(str(cvs_index_path))
    # set nprobe if index supports it (IVF indexes only)
    if hasattr(cvs_index, 'nprobe'):
        cvs_index.nprobe = 10
    print(f"  Loaded {len(cvs_df):,} CVs and index with {cvs_index.ntotal:,} vectors")

    print("All models loaded successfully")

    return {
        'bi_encoder': bi_encoder,
        'cross_encoder': cross_encoder,
        'jobs_index': jobs_index,
        'job_ids': job_ids,
        'jobs_df': jobs_df,
        'job_id_to_row': job_id_to_row,
        'cvs_df': cvs_df,
        'cvs_index': cvs_index,
        'device': device
    }


def find_matching_jobs(cv_text, top_k=50):
    # find top-k matching jobs using bi-encoder
    # uses full 1.3M job index with UUID-based ID mapping

    models = load_models()
    bi_encoder = models['bi_encoder']
    jobs_index = models['jobs_index']
    job_ids = models['job_ids']
    jobs_df = models['jobs_df']
    job_id_to_row = models['job_id_to_row']

    # prepare query text with e5 prefix
    prefixed_text = prepare_for_biencoder(cv_text, mode='cv')

    # encode query
    query_emb = bi_encoder.encode(
        [prefixed_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # search index
    similarities, indices = jobs_index.search(query_emb, top_k)

    # build match results
    matches = []
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        # get UUID directly from job_ids array
        job_id = job_ids[idx]

        # lookup job details
        if job_id in job_id_to_row:
            row_idx = job_id_to_row[job_id]
            job_row = jobs_df.iloc[row_idx]
        else:
            job_row = jobs_df.iloc[idx]

        match = {
            'rank': rank,
            'job_id': str(job_id),
            'bi_score': float(sim),
            'title': job_row.get('job_title', 'Unknown'),
            'company': job_row.get('company', 'Unknown'),
            'location': job_row.get('job_location', 'Unknown'),
            'skills': job_row.get('skills', ''),
            'seniority': job_row.get('seniority', ''),
            'text': job_row.get('embedding_text', '')
        }
        matches.append(match)

    return matches


def rerank_results(query_text, matches, top_k=None):
    # rerank matches using cross-encoder
    # returns sorted list of matches with cross_score added

    models = load_models()
    cross_encoder = models['cross_encoder']

    # strip prefixes from query
    clean_query = strip_prefixes(query_text)

    # build pairs
    pairs = []
    for m in matches:
        doc_text = strip_prefixes(m['text'])
        pairs.append((clean_query, doc_text))

    # score with cross-encoder
    cross_scores = cross_encoder.predict(pairs, batch_size=128)

    # add scores to matches and convert to percentage
    for m, score in zip(matches, cross_scores):
        m['cross_score'] = float(score)
        # logistic mapping to percentage
        m['match_pct'] = logistic_percentage(score)

    # sort by cross score
    reranked = sorted(matches, key=lambda x: x['cross_score'], reverse=True)

    # deduplicate jobs by ID (some job IDs appear twice in dataset)
    seen_ids = set()
    deduped = []
    for m in reranked:
        if m['job_id'] not in seen_ids:
            seen_ids.add(m['job_id'])
            deduped.append(m)
    reranked = deduped

    # return top k
    if top_k:
        return reranked[:top_k]
    return reranked


def find_matching_cvs(job_text, top_k=50):
    # find top-k matching CVs using bi-encoder
    # recruiter mode: job description -> matching CVs

    models = load_models()
    bi_encoder = models['bi_encoder']
    cvs_index = models['cvs_index']
    cvs_df = models['cvs_df']

    # prepare job text with e5 prefix (jobs use passage prefix)
    prefixed_text = prepare_for_biencoder(job_text, mode='job')

    # encode job
    job_emb = bi_encoder.encode(
        [prefixed_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # search CV index
    similarities, indices = cvs_index.search(job_emb, top_k)

    # build match results
    matches = []
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        # check bounds (cvs_df might be smaller than index)
        if idx >= len(cvs_df):
            continue

        cv_row = cvs_df.iloc[idx]
        match = {
            'rank': rank,
            'cv_id': cv_row['id'],
            'bi_score': float(sim),
            'text': cv_row['text']
        }
        matches.append(match)

    return matches


def rerank_results_recruiter(job_text, matches, top_k=None):
    # rerank CV matches using cross-encoder
    # for recruiter mode

    models = load_models()
    cross_encoder = models['cross_encoder']

    # strip prefixes from job
    clean_job = strip_prefixes(job_text)

    # build pairs - (job, cv) for cross-encoder
    pairs = []
    for m in matches:
        clean_cv = strip_prefixes(m['text'])
        pairs.append((clean_job, clean_cv))

    # score with cross-encoder
    cross_scores = cross_encoder.predict(pairs, batch_size=128)

    # add scores and convert to percentage
    for m, score in zip(matches, cross_scores):
        m['cross_score'] = float(score)
        # logistic mapping to percentage
        m['match_pct'] = logistic_percentage(score)

    # sort by cross score
    reranked = sorted(matches, key=lambda x: x['cross_score'], reverse=True)

    # deduplicate by text content - some CVs have identical text with different IDs
    seen_texts = set()
    deduped = []
    for m in reranked:
        text_key = m['text'].strip()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped.append(m)
    reranked = deduped

    # return top k
    if top_k:
        return reranked[:top_k]
    return reranked


# session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

if 'job_seeker_results' not in st.session_state:
    st.session_state.job_seeker_results = None

if 'recruiter_results' not in st.session_state:
    st.session_state.recruiter_results = None


# page config
st.set_page_config(
    page_title="CV-Job Matcher",
    layout="wide"
)

# initialize database
try:
    init_db()
except Exception as e:
    print(f"Database init error: {e}")

# header
st.title("CV-Job Matcher")
st.caption("Intelligent Recruitment and Talent Matching System")

# sidebar - skill discovery stats
with st.sidebar:
    st.subheader("Skill Discovery")
    try:
        proposals = get_skill_proposals(min_frequency=3)
        if proposals:
            st.info(f"{len(proposals)} new skills discovered")
            with st.expander("View proposed skills"):
                for skill, count in proposals[:10]:
                    st.write(f"- {skill} ({count} occurrences)")
        else:
            st.write("No new skills yet")
    except Exception:
        st.write("Skill tracking not available")

# tabs
tab1, tab2, tab3 = st.tabs(["Job Seeker", "Recruiter", "Pipeline Overview"])


# JOB SEEKER TAB
with tab1:
    st.header("Find Matching Jobs")

    # input methods
    col1, col2 = st.columns(2)

    with col1:
        cv_text_input = st.text_area(
            "Paste your CV text:",
            height=200,
            key="cv_text_input"
        )

    with col2:
        cv_file_upload = st.file_uploader(
            "Or upload a file:",
            type=["pdf", "docx", "txt"],
            key="cv_file_upload"
        )

    # example buttons
    st.subheader("Or try an example:")

    col_ex1, col_ex2, col_ex3, col_clear = st.columns(4)

    with col_ex1:
        if st.button("Python Developer", key="ex_python"):
            st.session_state.example_cv = "I am a Python Developer with 5 years of experience in Django, Flask, REST APIs, PostgreSQL, and AWS. I have worked on data pipelines and web applications."
            st.session_state.job_seeker_results = None

    with col_ex2:
        if st.button("Data Scientist", key="ex_data"):
            st.session_state.example_cv = "I am a Data Scientist with expertise in machine learning, Python, TensorFlow, scikit-learn, SQL, and data visualization. 3 years experience in predictive modeling."
            st.session_state.job_seeker_results = None

    with col_ex3:
        if st.button("Project Manager", key="ex_pm"):
            st.session_state.example_cv = "Experienced Project Manager with 8 years leading cross-functional teams. PMP certified. Skills in Agile, Scrum, stakeholder management, budgeting, and risk assessment."
            st.session_state.job_seeker_results = None

    with col_clear:
        if st.button("Clear", key="clear_js"):
            st.session_state.example_cv = None
            st.session_state.job_seeker_results = None

    # use example if set
    if 'example_cv' in st.session_state and st.session_state.example_cv:
        cv_text_input = st.session_state.example_cv
        st.info(f"Using example: {cv_text_input[:80]}...")

    # result count slider
    num_results = st.slider(
        "Number of results:",
        min_value=5,
        max_value=20,
        value=10,
        key="num_results"
    )

    # save for training checkbox
    save_for_training_js = st.checkbox("Save document for future model training", value=False, key="save_training_js")

    # search button
    if st.button("Find Matching Jobs", key="search_jobs_btn", type="primary"):
        # determine input text
        input_text = None

        if cv_file_upload:
            # parse uploaded file
            with st.spinner("Parsing uploaded file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=cv_file_upload.name) as tmp:
                    tmp.write(cv_file_upload.read())
                    tmp_path = tmp.name

                try:
                    parsed = parse_document(tmp_path)
                    if parsed:
                        raw_text = parsed['text']
                        # reformat for better matching
                        input_text = reformat_cv(raw_text)
                        word_count = len(input_text.split())
                        if word_count < 30:
                            st.warning(f"Short CV detected ({word_count} words). Results may be less accurate. Try pasting more details about your skills and experience.")
                    else:
                        st.error("Failed to parse uploaded file")
                finally:
                    os.unlink(tmp_path)

        elif cv_text_input:
            input_text = cv_text_input

        else:
            st.warning("Please paste CV text or upload a file")

        # search for matches
        if input_text:
            with st.spinner("Searching for matching jobs..."):
                # bi-encoder search
                matches = find_matching_jobs(input_text, top_k=50)

                # cross-encoder rerank
                reranked = rerank_results(input_text, matches, top_k=num_results)

                # store results in session
                st.session_state.job_seeker_results = {
                    'matches': reranked,
                    'cv_text': input_text
                }

                # track new skills from uploaded CV
                try:
                    track_skills_from_feedback(cv_text=input_text)
                except Exception as e:
                    pass  # dont crash if tracking fails

                # save document for future training if requested
                if save_for_training_js:
                    try:
                        save_path = PROJECT_ROOT / "demo" / "data" / "incoming" / "cv" / f"{uuid.uuid4()}.txt"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(input_text)
                        st.info(f"Document saved for future training: {save_path.name}")
                    except Exception as e:
                        st.warning(f"Could not save document: {e}")

                st.success(f"Found {len(reranked)} matching jobs")

    # display results
    if st.session_state.job_seeker_results:
        results_data = st.session_state.job_seeker_results
        matches = results_data['matches']
        cv_text = results_data['cv_text']

        st.subheader(f"Top {len(matches)} Matching Jobs")

        for i, match in enumerate(matches):
            # feedback tracking key
            feedback_key = f"js_{match['job_id']}"

            # expander header with match percentage
            header = f"#{i+1} | {match['title']} at {match['company']} | {match['location']} | {match['match_pct']}% match"

            # create columns for expander and feedback buttons
            exp_col, fb_col = st.columns([0.85, 0.15])

            with exp_col:
                with st.expander(header):
                    # job details inside expander
                    st.write(f"**Company:** {match['company']}")
                    st.write(f"**Location:** {match['location']}")
                    st.write(f"**Seniority:** {match['seniority']}")
                    st.write(f"**Match Score:** {match['match_pct']}%")
                    st.write(f"**Bi-encoder score:** {match['bi_score']:.4f}")
                    st.write(f"**Cross-encoder score:** {match['cross_score']:.2f}")

                    # skills
                    if match['skills']:
                        skills_list = match['skills'] if isinstance(match['skills'], list) else str(match['skills']).split(',')
                        skills_display = ', '.join(s.strip() for s in skills_list[:10])
                        st.write(f"**Skills:** {skills_display}")

                        # matching skills between CV and job
                        try:
                            cv_skills = extract_skills_from_text(cv_text)
                            job_skills_lower = [s.strip().lower() for s in skills_list]
                            cv_skills_lower = [s.lower() for s in cv_skills]
                            overlap = [s for s in job_skills_lower if s in cv_skills_lower]
                            if overlap:
                                st.write(f"**Matching skills:** {', '.join(overlap[:8])}")
                        except Exception:
                            pass

                    # job text preview
                    preview = match['text'][:500]
                    st.write(f"**Description:**")
                    st.text(preview)

                    # action buttons inside expander
                    act_col1, act_col2, act_col3 = st.columns(3)

                    with act_col1:
                        if st.button("Apply (+1.0)", key=f"apply_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'job_seeker',
                                'pasted_cv',
                                match['job_id'],
                                'apply',
                                similarity=match['match_pct']/100,
                                cv_text=cv_text,
                                job_text=match['text']
                            )
                            st.success("Logged: Apply")

                    with act_col2:
                        if st.button("Save (+0.5)", key=f"save_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'job_seeker',
                                'pasted_cv',
                                match['job_id'],
                                'save',
                                similarity=match['match_pct']/100,
                                cv_text=cv_text,
                                job_text=match['text']
                            )
                            st.success("Logged: Save")

                    with act_col3:
                        if st.button("Not Interested (-0.5)", key=f"notint_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'job_seeker',
                                'pasted_cv',
                                match['job_id'],
                                'not_interested',
                                similarity=match['match_pct']/100,
                                cv_text=cv_text,
                                job_text=match['text']
                            )
                            st.info("Logged: Not Interested")

            with fb_col:
                # quick feedback buttons outside expander
                fb_row1, fb_row2 = st.columns(2)

                with fb_row1:
                    if st.button("👍", key=f"thumbup_js_{i}"):
                        if feedback_key not in st.session_state.feedback:
                            log_action(
                                st.session_state.session_id,
                                'job_seeker',
                                'pasted_cv',
                                match['job_id'],
                                'save',
                                similarity=match['match_pct']/100,
                                cv_text=cv_text,
                                job_text=match['text']
                            )
                            st.session_state.feedback[feedback_key] = 'thumbup'

                with fb_row2:
                    if st.button("👎", key=f"thumbdown_js_{i}"):
                        if feedback_key not in st.session_state.feedback:
                            log_action(
                                st.session_state.session_id,
                                'job_seeker',
                                'pasted_cv',
                                match['job_id'],
                                'not_interested',
                                similarity=match['match_pct']/100,
                                cv_text=cv_text,
                                job_text=match['text']
                            )
                            st.session_state.feedback[feedback_key] = 'thumbdown'

                # show feedback status
                if feedback_key in st.session_state.feedback:
                    if st.session_state.feedback[feedback_key] == 'thumbup':
                        st.caption("✓ Liked")
                    else:
                        st.caption("✗ Passed")


# RECRUITER TAB
with tab2:
    st.header("Find Matching Candidates")

    # input methods
    col1, col2 = st.columns(2)

    with col1:
        job_text_input = st.text_area(
            "Paste job description:",
            height=200,
            key="job_text_input"
        )

    with col2:
        job_file_upload = st.file_uploader(
            "Or upload a file:",
            type=["pdf", "docx", "txt"],
            key="job_file_upload"
        )

    # example buttons
    st.subheader("Or try an example:")

    col_ex1, col_ex2, col_ex3, col_clear_rec = st.columns(4)

    with col_ex1:
        if st.button("Senior Python Developer", key="ex_python_job"):
            st.session_state.example_job = "We are looking for a Senior Python Developer with experience in Django, REST APIs, PostgreSQL, and cloud services (AWS/GCP). Must have 5+ years experience with Python and familiarity with CI/CD pipelines."
            st.session_state.recruiter_results = None

    with col_ex2:
        if st.button("Data Analyst", key="ex_data_job"):
            st.session_state.example_job = "Seeking a Data Analyst proficient in SQL, Python, Excel, and data visualization tools like Tableau or Power BI. Experience with statistical analysis and reporting required. 2+ years experience."
            st.session_state.recruiter_results = None

    with col_ex3:
        if st.button("DevOps Engineer", key="ex_devops_job"):
            st.session_state.example_job = "Looking for a DevOps Engineer experienced with Docker, Kubernetes, CI/CD, AWS, Linux, and infrastructure as code (Terraform). Must have strong scripting skills in Python or Bash."
            st.session_state.recruiter_results = None

    with col_clear_rec:
        if st.button("Clear", key="clear_rec"):
            st.session_state.example_job = None
            st.session_state.recruiter_results = None

    # use example if set
    if 'example_job' in st.session_state and st.session_state.example_job:
        job_text_input = st.session_state.example_job
        st.info(f"Using example: {job_text_input[:80]}...")

    # result count slider
    num_results_recruiter = st.slider(
        "Number of results:",
        min_value=5,
        max_value=20,
        value=10,
        key="num_results_recruiter"
    )

    # save for training checkbox
    save_for_training_rec = st.checkbox("Save document for future model training", value=False, key="save_training_rec")

    # search button
    if st.button("Find Matching Candidates", key="search_cvs_btn", type="primary"):
        # determine input text
        input_text = None

        if job_file_upload:
            # parse uploaded file
            with st.spinner("Parsing uploaded file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=job_file_upload.name) as tmp:
                    tmp.write(job_file_upload.read())
                    tmp_path = tmp.name

                try:
                    parsed = parse_document(tmp_path)
                    if parsed:
                        input_text = parsed['text']
                    else:
                        st.error("Failed to parse uploaded file")
                finally:
                    os.unlink(tmp_path)

        elif job_text_input:
            input_text = job_text_input

        else:
            st.warning("Please paste job description or upload a file")

        # search for matches
        if input_text:
            with st.spinner("Searching for matching candidates..."):
                # extract fields and build structured text for better matching
                fields = extract_job_fields(input_text)
                structured_job = build_job_embedding_string(fields)

                # bi-encoder search with structured job text
                matches = find_matching_cvs(structured_job, top_k=50)

                # cross-encoder rerank with structured job text
                reranked = rerank_results_recruiter(structured_job, matches, top_k=num_results_recruiter)

                # store results in session
                st.session_state.recruiter_results = {
                    'matches': reranked,
                    'job_text': input_text,
                    'job_skills': fields.get('skills', [])
                }

                # track new skills from job posting
                try:
                    track_skills_from_feedback(job_text=input_text)
                except Exception as e:
                    pass  # dont crash if tracking fails

                # save document for future training if requested
                if save_for_training_rec:
                    try:
                        save_path = PROJECT_ROOT / "demo" / "data" / "incoming" / "job" / f"{uuid.uuid4()}.txt"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(input_text)
                        st.info(f"Document saved for future training: {save_path.name}")
                    except Exception as e:
                        st.warning(f"Could not save document: {e}")

                st.success(f"Found {len(reranked)} matching candidates")

    # display results
    if st.session_state.recruiter_results:
        results_data = st.session_state.recruiter_results
        matches = results_data['matches']
        job_text = results_data['job_text']

        st.subheader(f"Top {len(matches)} Matching Candidates")

        for i, match in enumerate(matches):
            # feedback tracking key
            feedback_key = f"rec_{match['cv_id']}"

            # extract role info from CV text for better display
            cv_role, cv_exp, cv_skills = parse_cv_summary(match['text'])

            # expander header with role info and match percentage
            header = f"#{i+1} | {cv_role} ({match['cv_id']}) | {match['match_pct']}% match"

            # create columns for expander and feedback buttons
            exp_col, fb_col = st.columns([0.85, 0.15])

            with exp_col:
                with st.expander(header):
                    # CV details inside expander
                    st.write(f"**Role:** {cv_role}")
                    if cv_exp:
                        st.write(f"**Experience:** {cv_exp}")
                    if cv_skills:
                        st.write(f"**Key Skills:** {cv_skills}")

                    # matching skills between job and CV
                    try:
                        job_skills_list = results_data.get('job_skills', [])
                        if job_skills_list:
                            cv_text_lower = match['text'].lower()
                            overlap = [s for s in job_skills_list if s.lower() in cv_text_lower]
                            if overlap:
                                st.write(f"**Matching skills:** {', '.join(overlap[:8])}")
                    except Exception:
                        pass

                    st.write(f"**CV ID:** {match['cv_id']}")
                    st.write(f"**Match Score:** {match['match_pct']}%")
                    st.write(f"**Bi-encoder score:** {match['bi_score']:.4f}")
                    st.write(f"**Cross-encoder score:** {match['cross_score']:.2f}")

                    # CV text preview
                    preview = match['text'].replace("query: ", "").replace("Query: ", "")[:500]
                    st.write(f"**CV Preview:**")
                    st.text(preview)

                    # action buttons inside expander
                    act_col1, act_col2, act_col3 = st.columns(3)

                    with act_col1:
                        if st.button("Contact (+1.0)", key=f"contact_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'recruiter',
                                'pasted_job',
                                match['cv_id'],
                                'contact',
                                similarity=match['match_pct']/100,
                                cv_text=match['text'],
                                job_text=job_text
                            )
                            st.success("Logged: Contact")

                    with act_col2:
                        if st.button("Save (+0.5)", key=f"save_rec_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'recruiter',
                                'pasted_job',
                                match['cv_id'],
                                'save',
                                similarity=match['match_pct']/100,
                                cv_text=match['text'],
                                job_text=job_text
                            )
                            st.success("Logged: Save")

                    with act_col3:
                        if st.button("Not Interested (-0.5)", key=f"notint_rec_{i}"):
                            log_action(
                                st.session_state.session_id,
                                'recruiter',
                                'pasted_job',
                                match['cv_id'],
                                'not_interested',
                                similarity=match['match_pct']/100,
                                cv_text=match['text'],
                                job_text=job_text
                            )
                            st.info("Logged: Not Interested")

            with fb_col:
                # quick feedback buttons outside expander
                fb_row1, fb_row2 = st.columns(2)

                with fb_row1:
                    if st.button("👍", key=f"thumbup_rec_{i}"):
                        if feedback_key not in st.session_state.feedback:
                            log_action(
                                st.session_state.session_id,
                                'recruiter',
                                'pasted_job',
                                match['cv_id'],
                                'save',
                                similarity=match['match_pct']/100,
                                cv_text=match['text'],
                                job_text=job_text
                            )
                            st.session_state.feedback[feedback_key] = 'thumbup'

                with fb_row2:
                    if st.button("👎", key=f"thumbdown_rec_{i}"):
                        if feedback_key not in st.session_state.feedback:
                            log_action(
                                st.session_state.session_id,
                                'recruiter',
                                'pasted_job',
                                match['cv_id'],
                                'not_interested',
                                similarity=match['match_pct']/100,
                                cv_text=match['text'],
                                job_text=job_text
                            )
                            st.session_state.feedback[feedback_key] = 'thumbdown'

                # show feedback status
                if feedback_key in st.session_state.feedback:
                    if st.session_state.feedback[feedback_key] == 'thumbup':
                        st.caption("✓ Liked")
                    else:
                        st.caption("✗ Passed")


# PIPELINE OVERVIEW TAB
with tab3:
    st.header("Pipeline Overview")

    # system architecture diagram
    st.subheader("System Architecture")

    # load architecture diagrams
    arch_img = PROJECT_ROOT / "demo" / "images" / "system_architecture.png"
    flow_img = PROJECT_ROOT / "demo" / "images" / "data_flow_pipeline.png"

    if arch_img.exists():
        st.image(str(arch_img), caption="Logical System View", width='stretch')
    else:
        st.info("Architecture diagram not found")

    st.markdown("""
    **Data Flow:**
    1. Job postings ingested via Kafka -> Spark processing with NLP
    2. CVs parsed and text extracted -> NLP processing
    3. Both embedded using fine-tuned e5-base-v2 model
    4. Stored in Faiss indexes for fast similarity search
    5. Bi-encoder finds top-K candidates, cross-encoder reranks
    6. User feedback stored in SQLite for continuous improvement
    """)

    st.subheader("Pipeline Data Flow")

    if flow_img.exists():
        st.image(str(flow_img), caption="CV-Job Matching Pipeline", width='stretch')
    else:
        st.info("Pipeline diagram not found")

    # dataset statistics
    st.subheader("Dataset Statistics")

    models = load_models()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", f"{len(models['jobs_df']):,}")

    with col2:
        st.metric("Jobs in Index", f"{models['jobs_index'].ntotal:,}")

    with col3:
        st.metric("Total CVs", f"{len(models['cvs_df']):,}")

    with col4:
        st.metric("CVs in Index", f"{models['cvs_index'].ntotal:,}")

    # second row of stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Embedding Dims", "768")

    with col2:
        st.metric("Model", "e5-base-v2")

    with col3:
        st.metric("Cross-Encoder", "MiniLM-L12")

    with col4:
        st.metric("Device", models['device'].upper())

    # technology stack
    st.subheader("Technology Stack")

    tech_stack = {
        "Component": ["Ingestion", "Processing", "NLP", "Embeddings", "Search", "Reranking", "Storage", "Frontend", "Hardware"],
        "Technology": [
            "Apache Kafka, confluent_kafka",
            "Apache Spark, PySpark",
            "spaCy, PhraseMatcher, regex",
            "intfloat/e5-base-v2 (fine-tuned)",
            "Faiss (Facebook AI)",
            "cross-encoder/ms-marco-MiniLM-L12-v2",
            "Parquet, SQLite",
            "Streamlit",
            "NVIDIA RTX 3090"
        ]
    }

    st.table(tech_stack)

    # feedback statistics
    st.subheader("Feedback Statistics")

    # refresh button
    if st.button("Refresh Stats", key="refresh_stats_btn"):
        st.session_state.refresh_stats = True

    # get feedback summary
    summary = get_action_summary()

    if summary and summary.get('total', 0) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Actions", summary['total'])

        with col2:
            positive = sum(v['total_weight'] for v in summary.get('by_action', {}).values() if v['total_weight'] > 0)
            negative = abs(sum(v['total_weight'] for v in summary.get('by_action', {}).values() if v['total_weight'] < 0))
            if positive + negative > 0:
                ratio = positive / (positive + negative) * 100
                st.metric("Positive Ratio", f"{ratio:.1f}%")
            else:
                st.metric("Positive Ratio", "N/A")

        with col3:
            weighted = get_action_count()
            st.metric("Meaningful Actions", weighted)

        # actions by type
        if summary.get('by_action'):
            st.write("**Actions by Type:**")
            action_data = []
            for action, stats in summary['by_action'].items():
                action_data.append({
                    "Action": action,
                    "Count": stats['count'],
                    "Weight": f"{stats['total_weight']:.1f}"
                })
            st.table(action_data)

        # recent actions
        if summary.get('recent'):
            st.write("**Recent Actions:**")
            recent_data = []
            for r in summary['recent'][:5]:
                recent_data.append({
                    "Session": r[0][:8],
                    "Role": r[1],
                    "Action": r[2],
                    "Score": f"{r[3]:.2f}" if r[3] else "N/A",
                    "Time": r[4][:16]
                })
            st.table(recent_data)
    else:
        st.info("No feedback recorded yet. Try the Job Seeker or Recruiter tabs first!")

    # model retraining
    st.subheader("Model Retraining")

    feedback_count = get_action_count()
    threshold = 50

    st.write(f"**Current feedback:** {feedback_count} actions")
    st.write(f"**Threshold for retraining:** {threshold} actions")

    if feedback_count >= 1:
        st.write("✓ Enough feedback to trigger retraining (demo threshold: 1 action)")
    else:
        st.write(f"⚠ Need {threshold - feedback_count} more actions")

    # retrain button
    if st.button("Retrain Model", key="retrain_btn", type="primary"):
        if feedback_count < 1:
            st.warning(f"Need at least 1 feedback action for demo. Current: {feedback_count}")
        else:
            with st.spinner("Retraining model... (this may take several minutes, check terminal for progress)"):
                import subprocess
                import sys

                # run retrainer with subprocess for safety
                result = subprocess.run(
                    [sys.executable, '-c',
                     'import sys; sys.path.insert(0, "."); from demo.scripts.model_retrainer import retrain_from_feedback; retrain_from_feedback(threshold=1)'],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=600
                )

                if result.returncode == 0:
                    st.success("Model retrained successfully!")
                    # show last 500 chars of output
                    output = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                    st.code(output)
                else:
                    st.error("Retraining failed")
                    error = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    st.code(error)

    st.info("**Note:** For demo purposes, threshold is set to 1 action. In production, would require 50+ actions.")
