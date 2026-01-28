# feedback simulator --> generates fake feedback events for testing the feedback loop

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

from .config import EVENT_SIGNALS


def generate_synthetic_feedback(ground_truth_path, jobs_path, n_events=1000, noise_level=0.1, seed=42):
    #generates fake feedback events based on ground truth
    #---> if (cv, job) is a ground truth match, generate positive feedback
    #     if not, generate negative or neutral feedback

    random.seed(seed)
    np.random.seed(seed)

    #load data
    gt_df = pd.read_parquet(ground_truth_path)
    jobs_df = pd.read_parquet(jobs_path)

    #get all job ids -> need the index which has the job_id
    jobs_df = jobs_df.reset_index()
    all_job_ids = jobs_df.iloc[:, 0].tolist()  # first column after reset is job_id

    # build set of ground truth pairs for fast lookup
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        cv_id = row['anchor']
        for job_id in row['matches']:
            gt_pairs.add((cv_id, job_id))

    events = []
    cv_ids = gt_df['anchor'].tolist()

    for i in range(n_events):
        # pick random cv
        cv_id = random.choice(cv_ids)

        #50% chance to pick from ground truth 50% random
        if random.random() < 0.5:
            # pick from ground truth matches for this cv
            matches = gt_df[gt_df['anchor'] == cv_id]['matches'].iloc[0]
            if len(matches) > 0:
                job_id = random.choice(list(matches))
                is_match = True
            else:
                job_id = random.choice(all_job_ids)
                is_match = False
        else:
            #pick random job
            job_id = random.choice(all_job_ids)
            is_match = (cv_id, job_id) in gt_pairs

        #generate event type based on whether it's a match
        if is_match:
            #positive events for matches
            event_type = random.choice(['recruiter_hire', 'recruiter_interview',
                                        'recruiter_shortlist', 'candidate_apply'])
        else:
            #negative or neutral events for non-matches
            event_type = random.choice(['skip', 'recruiter_reject', 'candidate_view'])

        #add noise to signal strength
        base_signal = EVENT_SIGNALS[event_type]
        noise = np.random.normal(0, noise_level)
        signal = base_signal + noise

        #random timestamp in last 30 days
        days_ago = random.randint(0, 30)
        timestamp = datetime.now() - timedelta(days=days_ago)

        events.append({
            'cv_id': cv_id,
            'job_id': job_id,
            'event_type': event_type,
            'signal_strength': round(signal, 3),
            'timestamp': timestamp,
            'is_ground_truth': is_match
        })

    return pd.DataFrame(events)


if __name__ == "__main__":
    #quick test
    events = generate_synthetic_feedback(
        "training/validation_set_ids.parquet",
        "training/jobs_cleaned.parquet",
        n_events=100
    )
    print(f"Generated {len(events)} events")
    print(events.head(10))
    print("\nEvent type distribution:")
    print(events['event_type'].value_counts())
