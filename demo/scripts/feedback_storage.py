# feedback storage module for interactive demo
# Handles SQLite database for user actions and model retraining logs

# Usage:
#     from demo.scripts.feedback_storage import init_db, log_action, get_action_count
#     init_db()
#     log_action(session_id, role, doc_id, match_id, action, weight, similarity, cv_text, job_text)

import sqlite3
import os
from datetime import datetime
from pathlib import Path

# default database path
DB_PATH = Path("data/feedback/feedback.db")

# action weights (from CONTEXT.md)
ACTION_WEIGHTS = {
    'apply': 1.0,
    'contact': 1.0,
    'hire': 1.0,
    'save': 0.5,
    'view_full': 0.3,
    'skip': 0.0,
    'not_interested': -0.5
}


def init_db(db_path=None):
    #initialize SQLite database with schema
    if db_path is None:
        db_path = DB_PATH

    # create directory if needed
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # user actions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            match_id TEXT NOT NULL,
            action TEXT NOT NULL,
            weight REAL NOT NULL,
            similarity_score REAL,
            cv_text TEXT,
            job_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # retraining log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS retraining_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            previous_model TEXT,
            num_actions_used INTEGER,
            num_positive_pairs INTEGER,
            num_hard_negatives INTEGER,
            training_time_sec REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # skill updates table (for dictionary refresh)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skill_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_name TEXT NOT NULL UNIQUE,
            frequency INTEGER DEFAULT 1,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            added_to_dict BOOLEAN DEFAULT 0
        )
    ''')

    # create indexes for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_actions_session ON user_actions(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_actions_action ON user_actions(action)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON user_actions(timestamp)')

    # enable WAL mode for concurrent access
    cursor.execute('PRAGMA journal_mode=WAL')

    conn.commit()
    conn.close()

    print(f"Database initialized at {db_path}")
    return str(db_path)


def log_action(session_id, role, doc_id, match_id, action, similarity=None,
               cv_text=None, job_text=None, db_path=None):
    # log a user action to the database

    # args:
    #     session_id: unique session identifier
    #     role: 'recruiter' or 'job_seeker'
    #     doc_id: ID of uploaded document (CV or job)
    #     match_id: ID of matched document
    #     action: action taken ('apply', 'contact', 'save', etc.)
    #     similarity: similarity score shown to user
    #     cv_text: CV text (for training pairs)
    #     job_text: job text (for training pairs)
    #     db_path: optional custom database path

    # returns:
    #     action ID

    if db_path is None:
        db_path = DB_PATH

    weight = ACTION_WEIGHTS.get(action, 0.0)

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO user_actions
        (session_id, role, doc_id, match_id, action, weight, similarity_score, cv_text, job_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, role, doc_id, match_id, action, weight, similarity, cv_text, job_text))

    action_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return action_id


def get_action_count(db_path=None):
    #get count of meaningful actions (weight != 0)
    if db_path is None:
        db_path = DB_PATH

    if not Path(db_path).exists():
        return 0

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()
    count = cursor.execute(
        "SELECT COUNT(*) FROM user_actions WHERE weight != 0"
    ).fetchone()[0]
    conn.close()

    return count


def get_feedback_pairs(min_positive_weight=0.5, db_path=None):
    # get training pairs from user feedback

    # returns:
    #     positive_pairs: list of (cv_text, job_text) for positive actions
    #     hard_negatives: list of (cv_text, job_text) for negative actions
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    # positive pairs (Apply/Contact/Hire)
    positive = cursor.execute('''
        SELECT cv_text, job_text FROM user_actions
        WHERE weight >= ? AND cv_text IS NOT NULL AND job_text IS NOT NULL
    ''', (min_positive_weight,)).fetchall()

    # hard negatives (Not interested)
    negatives = cursor.execute('''
        SELECT cv_text, job_text FROM user_actions
        WHERE weight < 0 AND cv_text IS NOT NULL AND job_text IS NOT NULL
    ''').fetchall()

    conn.close()

    return positive, negatives


def log_retraining(model_version, previous_model, num_actions, num_positive,
                   num_negatives, training_time, db_path=None):
    #log a retraining event
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO retraining_log
        (model_version, previous_model, num_actions_used, num_positive_pairs,
         num_hard_negatives, training_time_sec)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (model_version, previous_model, num_actions, num_positive, num_negatives, training_time))

    conn.commit()
    conn.close()


def get_action_summary(db_path=None):
    #get summary of user actions
    if db_path is None:
        db_path = DB_PATH

    if not Path(db_path).exists():
        return {}

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()

    summary = {}

    # total actions
    summary['total'] = cursor.execute(
        "SELECT COUNT(*) FROM user_actions"
    ).fetchone()[0]

    # actions by type
    actions = cursor.execute('''
        SELECT action, COUNT(*), SUM(weight)
        FROM user_actions
        GROUP BY action
        ORDER BY COUNT(*) DESC
    ''').fetchall()

    summary['by_action'] = {a[0]: {'count': a[1], 'total_weight': a[2]} for a in actions}

    # actions by role
    roles = cursor.execute('''
        SELECT role, COUNT(*)
        FROM user_actions
        GROUP BY role
    ''').fetchall()

    summary['by_role'] = {r[0]: r[1] for r in roles}

    # recent actions (last 10)
    recent = cursor.execute('''
        SELECT session_id, role, action, similarity_score, timestamp
        FROM user_actions
        ORDER BY timestamp DESC
        LIMIT 10
    ''').fetchall()

    summary['recent'] = recent

    conn.close()
    return summary


# Test function
def test_storage():
    #test storage functions
    import uuid
    test_db = Path("data/feedback/test_feedback.db")

    try:
        # initialize
        init_db(test_db)
        print("1. Database initialized")

        # log actions
        session = str(uuid.uuid4())
        log_action(session, 'job_seeker', 'cv_123', 'job_456', 'apply',
                   0.85, 'sample cv text', 'sample job text', test_db)
        log_action(session, 'job_seeker', 'cv_123', 'job_789', 'not_interested',
                   0.65, 'sample cv text', 'another job', test_db)
        print("2. Actions logged")

        # get count
        count = get_action_count(test_db)
        assert count == 2, f"Expected 2 actions, got {count}"
        print(f"3. Action count: {count}")

        # get pairs
        positive, negative = get_feedback_pairs(db_path=test_db)
        assert len(positive) == 1, f"Expected 1 positive, got {len(positive)}"
        assert len(negative) == 1, f"Expected 1 negative, got {len(negative)}"
        print(f"4. Feedback pairs: {len(positive)} positive, {len(negative)} negative")

        # get summary
        summary = get_action_summary(test_db)
        print(f"5. Summary: {summary['total']} total actions")

        print("\nAll tests passed")

    finally:
        # cleanup
        if test_db.exists():
            test_db.unlink()
        wal = Path(str(test_db) + "-wal")
        shm = Path(str(test_db) + "-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()


if __name__ == "__main__":
    test_storage()
