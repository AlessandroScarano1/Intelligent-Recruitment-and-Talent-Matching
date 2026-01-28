# feedback aggregator ---> combines multiple feedback events into a single score per (cv, job) pair

import pandas as pd
import numpy as np


def aggregate_feedback(events_df):
    #takes a dataframe of events and aggregates them by (job_id, cv_id) pairs
    #returns a dataframe with cumulative scores

    #group by pair
    grouped = events_df.groupby(['job_id', 'cv_id'])

    #compute aggregates
    aggregated = grouped.agg({
        'signal_strength': ['sum', 'mean', 'count'],
        'timestamp': 'max'
    })

    #flatten column names
    aggregated.columns = ['cumulative_score', 'avg_signal', 'event_count', 'last_updated']

    #count positive and negative signals separately
    pos_mask = events_df['signal_strength'] > 0
    neg_mask = events_df['signal_strength'] < 0

    pos_counts = events_df[pos_mask].groupby(['job_id', 'cv_id']).size()
    neg_counts = events_df[neg_mask].groupby(['job_id', 'cv_id']).size()

    aggregated['positive_count'] = pos_counts.reindex(aggregated.index, fill_value=0)
    aggregated['negative_count'] = neg_counts.reindex(aggregated.index, fill_value=0)

    #reset index to make job_id, cv_id regular columns
    aggregated = aggregated.reset_index()

    #normalize score to [-1, 1] range
    max_abs = aggregated['cumulative_score'].abs().max()
    if max_abs > 0:
        aggregated['normalized_score'] = aggregated['cumulative_score'] / max_abs
    else:
        aggregated['normalized_score'] = 0.0

    return aggregated


def load_feedback_scores(path):
    #load aggregated feedback scores into a dict for fast lookup
    #returns: dict mapping (job_id, cv_id) -> normalized_score

    try:
        df = pd.read_parquet(path)
        scores = {}
        for _, row in df.iterrows():
            key = (row['job_id'], row['cv_id'])
            scores[key] = row['normalized_score']
        return scores
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    #quick test
    from simulator import generate_synthetic_feedback

    events = generate_synthetic_feedback(
        "../training/validation_set_ids.parquet",
        "../training/jobs_cleaned.parquet",
        n_events=500
    )

    aggregated = aggregate_feedback(events)
    print(f"Aggregated {len(events)} events into {len(aggregated)} pairs")
    print(aggregated.head(10))
