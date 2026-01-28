# model retrainer for feedback-driven improvement
# collects user feedback from SQLite and retrains bi-encoder

# usage:
#     from demo.scripts.model_retrainer import retrain_from_feedback
#     metrics = retrain_from_feedback(threshold=50)

# features:
#     - collects positive pairs from Apply/Contact/Hire actions
#     - mines hard negatives from "Not Interested" actions
#     - mixes original training data with feedback (80/20)
#     - saves new model version with timestamp

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.scripts.feedback_storage import (
    get_feedback_pairs,
    get_action_count,
    log_retraining,
    DB_PATH
)

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths - use PROJECT_ROOT for proper absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASE_MODEL_PATH = PROJECT_ROOT / "training" / "output" / "models" / "cv-job-matcher-e5"
ORIGINAL_TRAINING_PATH = PROJECT_ROOT / "training" / "output" / "training_data" / "training_dataset.parquet"
MODELS_DIR = PROJECT_ROOT / "training" / "output" / "models" / "retrained"


def check_retrain_needed(threshold=50):
    # check if retraining is needed based on action count
    # args:
    #     threshold: minimum actions before retraining
    # returns:
    #     (needed, action_count)
    count = get_action_count()
    return count >= threshold, count


def collect_training_data(min_positive_weight=0.5, mix_ratio=0.2):
    # collect training data from feedback and original dataset
    # args:
    #     min_positive_weight: minimum weight for positive pairs
    #     mix_ratio: fraction of feedback data in final mix (0.2 = 20% feedback)
    # returns:
    #     list of (anchor_text, positive_text) tuples
    # get feedback pairs
    positive, hard_negatives = get_feedback_pairs(min_positive_weight)

    if len(positive) == 0:
        logger.warning("No positive feedback pairs found")
        return [], []

    logger.info(f"Collected {len(positive)} positive pairs from feedback")
    logger.info(f"Collected {len(hard_negatives)} hard negatives from feedback")

    # load original training data
    if ORIGINAL_TRAINING_PATH.exists():
        original_df = pd.read_parquet(ORIGINAL_TRAINING_PATH)
        original_pairs = list(zip(
            original_df['anchor_text'].tolist(),
            original_df['positive_text'].tolist()
        ))
        logger.info(f"Loaded {len(original_pairs)} original training pairs")
    else:
        logger.warning("Original training data not found")
        original_pairs = []

    # calculate mix sizes
    # Target: feedback is mix_ratio of total
    # If feedback = 50, mix_ratio = 0.2, then total = 250, original = 200
    if len(original_pairs) > 0:
        feedback_count = len(positive)
        original_count = int(feedback_count * (1 - mix_ratio) / mix_ratio)

        # Sample from original if needed
        if original_count < len(original_pairs):
            indices = np.random.choice(len(original_pairs), original_count, replace=False)
            sampled_original = [original_pairs[i] for i in indices]
        else:
            sampled_original = original_pairs

        logger.info(f"Using {len(sampled_original)} original pairs (sampled)")
    else:
        sampled_original = []

    # Combine
    all_pairs = positive + sampled_original

    logger.info(f"Total training pairs: {len(all_pairs)}")
    logger.info(f"  - From feedback: {len(positive)} ({100*len(positive)/len(all_pairs):.1f}%)")
    logger.info(f"  - From original: {len(sampled_original)} ({100*len(sampled_original)/len(all_pairs):.1f}%)")

    return all_pairs, hard_negatives


def retrain_from_feedback(
    threshold=50,
    epochs=2,
    batch_size=32,
    learning_rate=1e-5,
    mix_ratio=0.2
):
    # retrain model from user feedback
    # args:
    #     threshold: minimum actions before retraining
    #     epochs: number of training epochs (keep low to avoid forgetting)
    #     batch_size: training batch size
    #     learning_rate: learning rate (low for fine-tuning)
    #     mix_ratio: fraction of feedback in training data
    # returns:
    #     dict with: success, model_path, metrics
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from datasets import Dataset

    # check if retraining needed
    needed, action_count = check_retrain_needed(threshold)
    if not needed:
        logger.info(f"Only {action_count} actions, need {threshold} for retraining")
        return {
            'success': False,
            'reason': f"Need {threshold} actions, have {action_count}",
            'action_count': action_count
        }

    logger.info(f"Starting retraining with {action_count} actions...")

    # collect training data
    train_pairs, hard_negatives = collect_training_data(mix_ratio=mix_ratio)

    if len(train_pairs) < 10:
        logger.error("Not enough training pairs")
        return {
            'success': False,
            'reason': "Not enough training pairs",
            'pairs_count': len(train_pairs)
        }

    # create dataset
    train_dataset = Dataset.from_dict({
        "anchor": [p[0] for p in train_pairs],
        "positive": [p[1] for p in train_pairs]
    })

    logger.info(f"Created dataset with {len(train_dataset)} pairs")

    # load base model
    logger.info(f"Loading base model from {BASE_MODEL_PATH}...")
    model = SentenceTransformer(str(BASE_MODEL_PATH))

    # configure loss (MNR uses in-batch negatives)
    loss = MultipleNegativesRankingLoss(model)

    # output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = MODELS_DIR / f"cv-job-matcher-{timestamp}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"  # disable wandb for quick retraining
    )

    # create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss
    )

    # train
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # save model
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")

    # log retraining event
    log_retraining(
        model_version=timestamp,
        previous_model=str(BASE_MODEL_PATH),
        num_actions=action_count,
        num_positive=len(train_pairs),
        num_negatives=len(hard_negatives),
        training_time=training_time
    )

    result = {
        'success': True,
        'model_path': str(output_path),
        'action_count': action_count,
        'training_pairs': len(train_pairs),
        'hard_negatives': len(hard_negatives),
        'training_time': training_time,
        'epochs': epochs
    }

    logger.info("retraining complete")
    logger.info(f"Model: {output_path}")
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Time: {training_time:.1f}s")

    return result


def get_latest_model():
    # get path to latest retrained model (or base model if none)
    if MODELS_DIR.exists():
        models = sorted(MODELS_DIR.glob("cv-job-matcher-*"))
        if models:
            return models[-1]

    return BASE_MODEL_PATH


def test_retrainer():
    # test retrainer functions
    print("Model Retrainer Test")

    # check if retraining needed
    needed, count = check_retrain_needed(threshold=50)
    print(f"1. Actions: {count}, Retrain needed: {needed}")

    # get latest model
    latest = get_latest_model()
    print(f"2. Latest model: {latest}")

    # try collecting data (don't actually retrain)
    if count > 0:
        pairs, negatives = collect_training_data()
        print(f"3. Collected: {len(pairs)} pairs, {len(negatives)} negatives")
    else:
        print("3. No actions yet, skipping data collection")

    print("\nTest complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Run retraining')
    parser.add_argument('--threshold', type=int, default=50, help='Action threshold')
    parser.add_argument('--test', action='store_true', help='Run tests')
    args = parser.parse_args()

    if args.test:
        test_retrainer()
    elif args.retrain:
        result = retrain_from_feedback(threshold=args.threshold)
        print(result)
    else:
        needed, count = check_retrain_needed(args.threshold)
        print(f"Actions: {count}/{args.threshold}")
        print(f"Retrain needed: {needed}")
