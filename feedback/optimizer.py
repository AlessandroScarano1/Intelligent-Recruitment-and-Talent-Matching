# weight optimizer to find the best weights for the scoring formula using grid search

import numpy as np
import pandas as pd
from itertools import product
from .config import DEFAULT_WEIGHTS

class WeightOptimizer:
    #optimizes weights by grid search over possible values
    #!! try to maximize agreement between predicted scores and feedback signals

    def __init__(self, feedback_df, ground_truth_df, search_space=None):
        self.feedback_df = feedback_df
        self.ground_truth_df = ground_truth_df

        #default search space if not provided
        if search_space is None:
            self.search_space = {
                'alpha': [0.8, 1.0, 1.2],
                'beta': [0.3, 0.5, 0.7],
                'gamma1': [0.05, 0.1, 0.15],
                'gamma2': [0.05, 0.1, 0.15],
                'delta': [0.0, 0.1, 0.2, 0.3]
            }
        else:
            self.search_space = search_space

    def compute_score(self, weights):
        #compute how good a set of weights predicts the feedback
        #returns correlation between feedback and weighted predictions

        #! check how many positive feedback pairs get high delta contribution
        pos_feedback = self.feedback_df[self.feedback_df['normalized_score'] > 0]

        if len(pos_feedback) == 0:
            return 0

        #the delta weight affects how much feedback contributes
        #!!!higher delta ----> more influence from feedback
        score = weights['delta'] * pos_feedback['normalized_score'].mean()

        # to keep other weights reasonable penalize extreme values
        penalty = 0
        for key in ['alpha', 'beta', 'gamma1', 'gamma2']:
            if weights[key] < 0.05 or weights[key] > 2.0:
                penalty += 0.1

        return score - penalty

    def grid_search(self, verbose=True):
        #try all combinations of weights and find the best

        #generate all combinations
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        combinations = list(product(*values))

        if verbose:
            print(f"testing {len(combinations)} weight combinations")

        best_score = -float('inf')
        best_weights = None

        for combo in combinations:
            weights = dict(zip(keys, combo))
            score = self.compute_score(weights)

            if score > best_score:
                best_score = score
                best_weights = weights

        if verbose:
            print(f"Best score: {best_score:.4f}")
            print(f"Best weights: {best_weights}")

        return best_weights, best_score


if __name__ == "__main__":
    #quick test
    from simulator import generate_synthetic_feedback
    from aggregator import aggregate_feedback

    #generate fake feedback
    events = generate_synthetic_feedback(
        "../training/validation_set_ids.parquet",
        "../training/jobs_cleaned.parquet",
        n_events=500
    )

    #aggregate
    aggregated = aggregate_feedback(events)

    #load ground truth
    gt_df = pd.read_parquet("../training/validation_set_ids.parquet")

    #optimize
    optimizer = WeightOptimizer(aggregated, gt_df)
    best_weights, best_score = optimizer.grid_search()

    print("\nOptimization complete!")
    print(f"Use these weights: {best_weights}")
