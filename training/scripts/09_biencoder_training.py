# %%
import os
import sys
import argparse

# Find project root (cv-job-matcher-project/)
# Notebooks are at */notebooks/ so we need to go up TWO levels
cwd = os.getcwd()
if 'notebooks' in cwd or 'scripts' in cwd:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(cwd))  # TWO levels up
else:
    PROJECT_ROOT = cwd
sys.path.insert(0, PROJECT_ROOT)

# parse arguments
parser = argparse.ArgumentParser(description='Train bi-encoder for CV-job matching')
parser.add_argument('--quick', action='store_true',
                    help='Quick mode: single training run with best hyperparams')
parser.add_argument('--lr', type=float, default=5e-05,
                    help='Learning rate for quick mode (default: 5e-05)')
parser.add_argument('--warmup', type=float, default=0.05,
                    help='Warmup ratio for quick mode (default: 0.05)')
args = parser.parse_args()

QUICK_MODE = args.quick
BEST_LR = args.lr
BEST_WARMUP = args.warmup

if QUICK_MODE:
    print(f'QUICK_MODE enabled: single run with lr={BEST_LR}, warmup={BEST_WARMUP}')
else:
    print('FULL MODE: running hyperparameter sweep (6 combinations)')

print(f"Project root: {PROJECT_ROOT}")

# %%
# Clean up previous sweep results before starting new sweep
import shutil
import os

sweep_dir = os.path.join(PROJECT_ROOT, "training", "output", "models", "sweep")
if os.path.exists(sweep_dir):
    print(f"Found existing sweep directory with:")
    for item in os.listdir(sweep_dir):
        print(f"  - {item}")
    shutil.rmtree(sweep_dir)
    print(f"\nDeleted {sweep_dir}")
else:
    print("No previous sweep results found")

# Also clean initial training model
initial_dir = os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5')
if os.path.exists(initial_dir):
    print(f"\nDeleting initial training model at: {initial_dir}")
    shutil.rmtree(initial_dir)
    print("Deleted")
else:
    print("\nNo initial training model found")

# Clean best model copy if exists
best_dir = os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5-best')
if os.path.exists(best_dir):
    print(f"\nDeleting previous best model at: {best_dir}")
    shutil.rmtree(best_dir)
    print("Deleted")

# Optional: clean local wandb logs (your runs are still saved online)
wandb_dir = "wandb"
if os.path.exists(wandb_dir):
    print(f"\nDeleting local W&B logs at: {wandb_dir}")
    shutil.rmtree(wandb_dir)
    print("Deleted (online logs preserved at wandb.ai)")

print("\n✓ Ready for fresh sweep")

# %%
# set working directory to project root
import os
os.chdir(PROJECT_ROOT)
print(f"Working directory: {os.getcwd()}")

# kill any lingering Spark sessions to free RAM
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    spark.stop()
    print("Stopped lingering Spark session")
except:
    print("No Spark session to stop")

# check available memory
import subprocess
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
print(f"\nSystem memory:\n{result.stdout}")

# %%
from nbconvert import export
import pandas as pd
import numpy as np
import torch
import wandb
import gc

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset
    
#CHECK: W&B API Key
if 'WANDB_API_KEY' not in os.environ:
    print("WARNING: WANDB_API_KEY not set in environment")
    print("Either set it with: export WANDB_API_KEY=your_key_here")
    print("Or run: wandb login")
else:
    print("WANDB_API_KEY found in environment")

# Check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("!!!No GPU available, training will be slow")

# CHECK: training data from Plan 01
assert os.path.exists(os.path.join(PROJECT_ROOT, 'training', 'output', 'training_data', 'training_dataset.parquet')), \
    "training data not found, run 08 first"
print("\nTraining data found, ready to proceed")

# %%
# load training data from previous output
train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'training', 'output', 'training_data', 'training_dataset.parquet'))
val_df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'training', 'output', 'training_data', 'validation_dataset.parquet'))
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# show sample
print("\nSample training pair:")
print(f"CV (anchor): {train_df.iloc[0]['anchor_text'][:300]}...")
print(f"Job (positive): {train_df.iloc[0]['positive_text'][:300]}...")

# %%
# !!! sentence-transformers v3 uses Dataset, NOT InputExample
# column names must be 'anchor' and 'positive' for MNR loss
train_dataset = Dataset.from_dict({
    "anchor": train_df['anchor_text'].tolist(),
    "positive": train_df['positive_text'].tolist()
})

val_dataset = Dataset.from_dict({
    "anchor": val_df['anchor_text'].tolist(),
    "positive": val_df['positive_text'].tolist()
})

print(f"train dataset: {train_dataset}")
print(f"val dataset: {val_dataset}")

# %%
# load base model
print("Loading e5-base-v2 model")
model = SentenceTransformer("intfloat/e5-base-v2")
print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")

# MNR loss (uses in-batch negatives)
base_loss = MultipleNegativesRankingLoss(model)

# wrap with MatryoshkaLoss for multi-dimension training
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64]  # train at all these dimensions
)
print("Configured MNR + MatryoshkaLoss")

# %%
# training arguments (SentenceTransformerTrainer uses HF Trainer backend)
args = SentenceTransformerTrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5'),
        num_train_epochs=10,
        per_device_train_batch_size=64,  # should fit high-end GPU (24GB VRAM) with fp16
    learning_rate=2e-5,
    warmup_ratio=0.1,  # 10% warmup to avoid forgetting
    fp16=True,  # mixed precision for speed

    # evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # keep only best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # logging
    logging_steps=10,
    run_name="cv-job-e5-mnr-matryoshka",  # W&B run name

    # Early stopping via Trainer
    greater_is_better=False,  # Lower loss is better
)

print("Training arguments configured")
print(f" epochs: {args.num_train_epochs}")
print(f" batch size: {args.per_device_train_batch_size}")
print(f" learning rate: {args.learning_rate}")

# %%
# initialize W&B
# will prompt for login if WANDB_API_KEY not set
wandb.init(
    project="talent-matching",
    name="cv-job-e5-mnr-matryoshka",
    config={
        "model": "intfloat/e5-base-v2",
        "loss": "MNR+Matryoshka",
        "matryoshka_dims": [768, 512, 256, 128, 64],
        "batch_size": 64,
        "learning_rate": 2e-5,
        "epochs": 10,
        "train_samples": len(train_df),
        "val_samples": len(val_df)
    }
)
print(f"W&B initialized: {wandb.run.url}")

# %%
# early stopping callback, stops if val loss doesn't improve for 3 epochs
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)

# trainer with early stopping
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    callbacks=[early_stopping]
)

# train
print("Starting training with early stopping (patience=3)")
trainer.train()

print("Training complete")
print(f"Best validation loss: {trainer.state.best_metric:.4f}")
print(f"Epochs trained: {trainer.state.epoch:.0f}")

# %%
# save the trained model
model.save(os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5'))
print(f"Model saved to: {os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5')}")

# list saved files
import os
for f in os.listdir(os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5')):
    size = os.path.getsize(os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5', f)) / 1e6
    print(f"  {f}: {size:.1f} MB")

# finish W&B run
wandb.finish()
print("W&B run finished")

# %%
# test the trained model
trained_model = SentenceTransformer(os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5'))

# encode a sample CV and job
sample_cv = "query: python developer with 5 years experience in Django and PostgreSQL"
sample_job = "passage: Title: Senior Python Developer. Required: Python, Django, PostgreSQL, 5+ years experience"

cv_emb = trained_model.encode(sample_cv)
job_emb = trained_model.encode(sample_job)

# compute similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([cv_emb], [job_emb])[0][0]
print(f"Sample similarity: {sim:.4f}")
print("(Higher is better, expect > 0.7 for good match)")

# %%
# hyperparameter sweep configuration, controlled by QUICK_MODE
if QUICK_MODE:
    # single run with best hyperparameters
    LEARNING_RATES = [BEST_LR]
    WARMUP_RATIOS = [BEST_WARMUP]
    SWEEP_EPOCHS = 9
    SWEEP_PATIENCE = 3
else:
    # full sweep
    LEARNING_RATES = [2e-5, 1e-5, 5e-6]
    WARMUP_RATIOS = [0.1, 0.05]
    SWEEP_EPOCHS = 13
    SWEEP_PATIENCE = 3

total_runs = len(LEARNING_RATES) * len(WARMUP_RATIOS)
print(f"Sweep will run {total_runs} experiments:")
print(f"  learning rates: {LEARNING_RATES}")
print(f"  warmup ratios: {WARMUP_RATIOS}")
print(f"  max epochs per run: {SWEEP_EPOCHS}")
print(f"  early stopping patience: {SWEEP_PATIENCE}")

# %%
# run hyperparameter sweep with proper memory management
sweep_results = []
run_num = 1

# cleanup: free memory from initial training (Sections 4-8) before starting sweep
try:
    del model, trainer, loss, base_loss
    print("Cleaned up initial training objects")
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()

for lr in LEARNING_RATES:
    for warmup in WARMUP_RATIOS:
        # cleanup before each run
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Log memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"\n{'='*50}")
            print(f"GPU memory before run: {allocated:.2f} GB allocated")
        
        run_name = f"sweep_lr{lr}_warmup{warmup}"
        output_dir = os.path.join(PROJECT_ROOT, "training", "output", "models", "sweep", run_name)
        
        print(f"RUN {run_num}/{total_runs}: lr={lr}, warmup={warmup}")
        
        # init wandb for this run
        wandb.init(
            project="talent-matching-sweep",
            name=run_name,
            config={"learning_rate": lr, "warmup_ratio": warmup, "epochs": SWEEP_EPOCHS},
            reinit=True
        )
        
        # load fresh model
        sweep_model = SentenceTransformer("intfloat/e5-base-v2")
        sweep_base_loss = MultipleNegativesRankingLoss(sweep_model)
        sweep_loss = MatryoshkaLoss(
            model=sweep_model,
            loss=sweep_base_loss,
            matryoshka_dims=[768, 512, 256, 128, 64]
        )
        
        # training args for this run
        sweep_args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=SWEEP_EPOCHS,
            per_device_train_batch_size=64,
            learning_rate=lr,
            warmup_ratio=warmup,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=20,
            report_to="wandb",
        )
        
        # early stopping
        sweep_early_stop = EarlyStoppingCallback(
            early_stopping_patience=SWEEP_PATIENCE,
            early_stopping_threshold=0.001
        )
        
        # trainer
        sweep_trainer = SentenceTransformerTrainer(
            model=sweep_model,
            args=sweep_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            loss=sweep_loss,
            callbacks=[sweep_early_stop]
        )
        
        # train
        sweep_trainer.train()
        
        # save result
        final_loss = sweep_trainer.state.best_metric
        epochs_done = sweep_trainer.state.epoch
        sweep_results.append({
            "run": run_name,
            "lr": lr,
            "warmup": warmup,
            "val_loss": final_loss,
            "epochs": epochs_done
        })
        
        print(f"Result: val_loss={final_loss:.4f}, epochs={epochs_done:.0f}")
        
        # save model
        sweep_model.save(output_dir)
        
        # finish wandb run
        wandb.finish()
        
        # aggressive cleanup after each run
        # must delete in correct order: trainer first (holds references), then model
        del sweep_trainer
        del sweep_loss, sweep_base_loss
        del sweep_model
        
        # force Python garbage collection
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        run_num += 1

print("\nSweep complete")

# %%
# compare sweep results
results_df = pd.DataFrame(sweep_results)
results_df = results_df.sort_values("val_loss")

print("SWEEP RESULTS (sorted by validation loss):")
print(results_df.to_string(index=False))

# best model
best = results_df.iloc[0]
print(f"\nBEST MODEL:")
print(f"  learning rate: {best['lr']}")
print(f"  warmup ratio: {best['warmup']}")
print(f"  validation loss: {best['val_loss']:.4f}")
print(f"  epochs trained: {best['epochs']:.0f}")

# %%
# copy best model to main location
import shutil

best_src = os.path.join(PROJECT_ROOT, "training", "output", "models", "sweep", best['run'])
best_dst = os.path.join(PROJECT_ROOT, 'training', 'output', 'models', 'cv-job-matcher-e5-best')

if os.path.exists(best_dst):
    shutil.rmtree(best_dst)
shutil.copytree(best_src, best_dst)

print(f"Best model copied to: {best_dst}")

# save sweep results
os.makedirs("output/models/sweep", exist_ok=True)
results_df.to_csv(os.path.join(PROJECT_ROOT, "training", "output", "models", "sweep", "sweep_results.csv"), index=False)
print(f"Results saved to: output/models/sweep/sweep_results.csv")


