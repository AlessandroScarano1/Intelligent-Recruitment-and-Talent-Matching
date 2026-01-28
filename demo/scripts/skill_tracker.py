# skill tracker for discovering new skills from user feedback
# Monitors uploaded CVs and jobs for skills not in the dictionary

# Usage:
#   from demo.scripts.skill_tracker import track_skills_from_feedback, get_skill_proposals

#  after user uploads document
#     track_skills_from_feedback(cv_text, job_text)

#  to see proposed new skills
#   proposals = get_skill_proposals(min_frequency=3)

# pattern: Same as Phase 9 skill_dictionary_refresh.py but sources from feedback table.

import re
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# database path (same as feedback_storage)
DB_PATH = Path("data/feedback/feedback.db")

# skill dictionary path
# Use PROJECT_ROOT for proper absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILL_DICT_PATH = PROJECT_ROOT / "training" / "output" / "skill_dictionary" / "all_skills"


def load_skill_dictionary():
    # load existing skills from dictionary
    if not SKILL_DICT_PATH.exists():
        logger.warning(f"Skill dictionary not found at {SKILL_DICT_PATH}")
        return set()

    import pandas as pd
    try:
        df = pd.read_parquet(SKILL_DICT_PATH)
        if 'skill_name' in df.columns:
            skills = set(df['skill_name'].str.lower().tolist())
        elif 'skill' in df.columns:
            skills = set(df['skill'].str.lower().tolist())
        else:
            # try first column
            skills = set(df.iloc[:, 0].str.lower().tolist())

        logger.info(f"Loaded {len(skills):,} skills from dictionary")
        return skills
    except Exception as e:
        logger.error(f"Failed to load skill dictionary: {e}")
        return set()


def extract_potential_skills(text):
    # extract potential skill mentions from text
    # uses simple heuristics, not full NLP extraction
    # args:
    #     text: CV or job text

    # returns:
    #     list of potential skill strings (lowercase)
    if not text:
        return []

    # common skill patterns (simplified from Phase 7 ensemble extractor)
    patterns = [
        # technology names (capitalized words)
        r'\b([A-Z][a-z]+(?:\.[A-Za-z]+)?)\b',
        # framework/library patterns
        r'\b([A-Za-z]+(?:JS|SQL|ML|AI|DB|API))\b',
        # version patterns
        r'\b([A-Za-z]+\s*\d+(?:\.\d+)?)\b',
        # acronyms
        r'\b([A-Z]{2,6})\b',
    ]

    potential = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        potential.extend([m.lower() for m in matches if len(m) > 2])

    # also look for items after "skills:" or similar headers
    skills_section = re.search(r'skills[:\s]+([^.]+)', text.lower())
    if skills_section:
        items = re.split(r'[,;|]', skills_section.group(1))
        potential.extend([item.strip().lower() for item in items if len(item.strip()) > 2])

    return list(set(potential))


def track_skills_from_feedback(cv_text=None, job_text=None, db_path=None):
    # track potential new skills from feedback text
    # updates skill_updates table with frequency counts
    # args:
    #     cv_text: CV text from user upload
    #     job_text: Job text from matched job
    #     db_path: Optional custom database path
    if db_path is None:
        db_path = DB_PATH

    # load existing dictionary
    known_skills = load_skill_dictionary()

    # extract potential skills from both texts
    potential = []
    if cv_text:
        potential.extend(extract_potential_skills(cv_text))
    if job_text:
        potential.extend(extract_potential_skills(job_text))

    if not potential:
        return

    # filter to only new skills (not in dictionary)
    new_skills = [s for s in potential if s not in known_skills and len(s) > 2]

    if not new_skills:
        return

    # update database
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    for skill in new_skills:
        # upsert: insert or update frequency
        cursor.execute('''
            INSERT INTO skill_updates (skill_name, frequency, first_seen, last_seen)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(skill_name) DO UPDATE SET
                frequency = frequency + 1,
                last_seen = ?
        ''', (skill, now, now, now))

    conn.commit()
    conn.close()

    logger.debug(f"Tracked {len(new_skills)} potential new skills")


def get_skill_proposals(min_frequency=3, db_path=None):
    # get proposed new skills that appear frequently
    # args:
    #     min_frequency: minimum occurrences to propose (default 3, same as Phase 9)
    #     db_path: optional custom database path
    # returns:
    #     list of (skill_name, frequency) tuples sorted by frequency
    if db_path is None:
        db_path = DB_PATH

    if not Path(db_path).exists():
        return []

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    proposals = cursor.execute('''
        SELECT skill_name, frequency
        FROM skill_updates
        WHERE frequency >= ? AND added_to_dict = 0
        ORDER BY frequency DESC
    ''', (min_frequency,)).fetchall()

    conn.close()

    return proposals


def approve_skill(skill_name, db_path=None):
    # mark a skill as approved (added to dictionary)
    # note: actual dictionary integration happens via Spark batch job
    # args:
    #     skill_name: skill to approve
    #     db_path: optional custom database path
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE skill_updates
        SET added_to_dict = 1
        WHERE skill_name = ?
    ''', (skill_name.lower(),))

    conn.commit()
    conn.close()

    logger.info(f"Approved skill: {skill_name}")


def get_skill_statistics(db_path=None):
    # get statistics about tracked skills
    # returns:
    #     dict with counts and top proposals
    if db_path is None:
        db_path = DB_PATH

    if not Path(db_path).exists():
        return {'total_tracked': 0, 'pending': 0, 'approved': 0, 'top_proposals': []}

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    total = cursor.execute('SELECT COUNT(*) FROM skill_updates').fetchone()[0]
    pending = cursor.execute('SELECT COUNT(*) FROM skill_updates WHERE added_to_dict = 0').fetchone()[0]
    approved = cursor.execute('SELECT COUNT(*) FROM skill_updates WHERE added_to_dict = 1').fetchone()[0]

    top = cursor.execute('''
        SELECT skill_name, frequency
        FROM skill_updates
        WHERE added_to_dict = 0
        ORDER BY frequency DESC
        LIMIT 10
    ''').fetchall()

    conn.close()

    return {
        'total_tracked': total,
        'pending': pending,
        'approved': approved,
        'top_proposals': top
    }


def test_skill_tracker():
    # test skill tracker functions
    print("Skill Tracker Test")

    # test extraction
    test_cv = """
    John Doe - Senior Developer
    Skills: Python, Django, FastAPI, PostgreSQL, Redis, Kubernetes
    Experience with machine learning and data engineering.
    """

    potential = extract_potential_skills(test_cv)
    print(f"1. Extracted {len(potential)} potential skills from test CV")
    print(f"   Sample: {potential[:5]}")

    # Test tracking
    track_skills_from_feedback(cv_text=test_cv)
    print("2. Tracked skills from test CV")

    # get statistics
    stats = get_skill_statistics()
    print(f"3. Stats: {stats['total_tracked']} tracked, {stats['pending']} pending")

    # get proposals
    proposals = get_skill_proposals(min_frequency=1)  # Low threshold for test
    print(f"4. Proposals: {proposals[:5]}")

    print("\nTest complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--proposals', action='store_true', help='Show proposals')
    parser.add_argument('--min-freq', type=int, default=3, help='Min frequency for proposals')
    args = parser.parse_args()

    if args.test:
        test_skill_tracker()
    elif args.status:
        stats = get_skill_statistics()
        print(f"Total tracked: {stats['total_tracked']}")
        print(f"Pending approval: {stats['pending']}")
        print(f"Already approved: {stats['approved']}")
        if stats['top_proposals']:
            print("\nTop proposals:")
            for skill, freq in stats['top_proposals']:
                print(f"  {skill}: {freq} occurrences")
    elif args.proposals:
        proposals = get_skill_proposals(min_frequency=args.min_freq)
        if proposals:
            print(f"Proposed new skills (min freq: {args.min_freq}):")
            for skill, freq in proposals:
                print(f"  {skill}: {freq}")
        else:
            print(f"No proposals with frequency >= {args.min_freq}")
    else:
        parser.print_help()
