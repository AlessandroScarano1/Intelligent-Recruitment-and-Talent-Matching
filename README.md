# Intelligent Recruitment and Talent Matching System

A scalable end-to-end big data system for semantic job-CV matching using bi-encoder retrieval and cross-encoder reranking, processing 1.35M job postings and achieving 79.42% Recall@50. Includes Kafka ingestion, Spark NLP processing, GPU-accelerated encoding, FAISS vector search, and interactive user interfaces with feedback-driven continuous learning.

# Project Overview

This system implements a semantic search engine for matching job seekers with relevant positions using:
- Dense retrieval with fine-tuned E5 embeddings (768D vectors)
- Two-stage ranking: Bi-encoder retrieval + cross-encoder reranking
- Big data pipeline: Kafka -> Spark -> GPU encoding -> FAISS indexing
- Real-world scale: 1.35M job postings, 4.8K CVs

# Key Results
- Recall@50: 79.42% | Recall@10: 53.91% | Recall@1: 14.77%
- TF-IDF baseline comparison: bi-encoder outperforms TF-IDF by 1.3-2.0x across all Recall@K
- Throughput: 2,320 encodings/sec on RTX 3090
- Query latency: 246ms for top-50 retrieval (1.35M vectors)

---

# System Architecture

```
Data Sources -> Kafka Ingestion -> Spark Processing -> GPU Encoding
                     |                    |              |
                 (3 partitions)     (NLP + JOIN)   (e5-base-v2)
                                                        |
                                              Training Pipeline
                                         (MNR + Matryoshka Loss)
                                                        |
                                         Re-encode ALL 1.35M jobs
                                                        |
                                            FAISS Index (~4.1GB)
                                                        |
                                    Matching: Bi-encoder -> Cross-encoder
```

---

# Repository Structure

```
.
├── ingest_cv/                  # CV ingestion pipeline
│   ├── scripts/               # Scripts 01-05: Kafka -> Spark -> Embeddings
│   └── raw_data/              # CV JSONL files (4,817 resumes)
│
├── ingest_job_postings/       # Job ingestion pipeline
│   ├── scripts/               # Scripts 01-06: LinkedIn + Indeed/Glassdoor
│   └── raw_data/              # Job CSV files (1.35M postings)
│
├── training/                   # Model training pipeline
│   ├── scripts/               # Scripts 07-10: Train -> Index
│   └── output/
│       ├── models/            # Fine-tuned e5-base-v2 model
│       └── indexes/           # FAISS indexes (jobs + CVs)
│
├── demo/                       # Interactive demo applications
│   ├── app.py                 # Streamlit web app (3 tabs: job seeker, recruiter, pipeline overview)
│   ├── demo_scripts/          # Scripts 11-15: Retrieval demos + TF-IDF baseline
│   └── scripts/               # Feedback storage, document parser, file watcher, skill tracker
│
├── scripts/                    # Utility scripts (backup/restore outputs, download from GDrive)
├── docs/                       # Architecture diagrams and documentation
├── docker-compose.yml          # GPU-enabled container setup
└── run_full_pipeline.sh        # Main execution script
```

---

# Getting Started

## Prerequisites

- Docker and Docker Compose
- 40GB+ disk space for datasets and models
- For GPU acceleration: NVIDIA GPU with CUDA support and `nvidia-container-toolkit`
- For macOS: Apple Silicon with MPS support (M1/M2/M3/M4/M5)

## Download Data

The raw datasets and pre-trained outputs are hosted on Google Drive (~17GB total):

```bash
# automated download (uses gdown)
bash scripts/download_gdrive.sh

# or download manually from:
# https://drive.google.com/drive/folders/18PoruD26-s-3OSpGpTZRIrY83kP-JqeS
```

Three zip files are available:
- `raw_data.zip` (~2GB) - raw CSV/JSONL datasets -> unzip in project root
- `ingestion_output.zip` (~1.3GB) - processed parquet files from ingestion
- `training_output.zip` (~14GB) - models, FAISS indexes, embeddings

To skip the pipeline and just run the demo, download `ingestion_output.zip` + `training_output.zip`.

---

# Platform Setup

Three docker-compose files and two conda environments are provided:

| File | Hardware | Environment | GPU Access |
|------|----------|-------------|------------|
| `docker-compose.yml` | Linux + NVIDIA GPU | `environment_project.yml` (pytorch-cuda, faiss-gpu) | CUDA |
| `docker-compose-cpu.yml` | Any machine, no GPU | `environment_nocuda.yml` (faiss-cpu, no CUDA) | None (CPU only) |
| `docker-compose-macos.yml` | macOS Apple Silicon | `environment_nocuda.yml` (faiss-cpu, no CUDA) | None in Docker |

IMPORTANT: Do NOT use `docker-compose.yml` on a machine without an NVIDIA GPU. The environment requires `pytorch-cuda=12.4` and `faiss-gpu` which will fail to install without NVIDIA CUDA.

## Option A: Linux with NVIDIA GPU (recommended for full pipeline)

This is the primary development setup. All scripts run inside the Docker container with full CUDA GPU acceleration.

```bash
# 1. Build and start the container (includes Kafka + project environment)
docker compose up -d

# 2. Enter the container
docker exec -it talent-matching-container bash

# 3. Activate the conda environment
conda activate talent-matching

# 4. Run the pipeline (inside container)
bash run_pipeline_quick_train.sh    # ingestion + 1 training run
# or
bash run_full_pipeline.sh           # ingestion + full hyperparameter sweep

# 5. Run the demo (inside container)
streamlit run demo/app.py --server.address 0.0.0.0
# Access at http://localhost:8501
```

**Notes:**
- Requires `nvidia-container-toolkit` installed on the host
- Spark driver memory is set to 4GB, enough for 1.35M jobs on machines with 16GB+ RAM
- Ports 8501 (Streamlit) and 9092 (Kafka) are exposed to the host

## Option B: Linux/Windows without GPU (CPU only)

For machines without an NVIDIA GPU. Everything runs inside Docker but uses CPU for encoding and training (significantly slower).

```bash
# 1. Build and start the container (CPU environment)
docker compose -f docker-compose-cpu.yml up -d

# 2. Enter the container
docker exec -it talent-matching-container bash

# 3. Activate the conda environment
conda activate talent-matching

# 4. Run the pipeline (inside container, CPU mode)
bash run_pipeline_quick_train.sh

# 5. Run the demo
streamlit run demo/app.py --server.address 0.0.0.0
```

**Notes:**
- Uses `environment_nocuda.yml` (faiss-cpu, no CUDA dependencies)
- Embedding generation and training are much slower on CPU
- All scripts auto-detect CPU via `get_device()` function, no code changes needed
- If you have limited RAM (<16GB), consider using the quick pipeline first to verify everything works

## Option C: macOS with Apple Silicon (MPS acceleration)

Docker on macOS runs a Linux VM with no GPU access. For Apple Silicon GPU acceleration (MPS), install the environment natively and only use Docker only for Kafka.

**compose-macos vs compose-cpu:** The macOS compose file adds `platform: linux/amd64` to force x86 emulation via Rosetta on Apple Silicon. Without it, Docker would try to pull ARM64 images which may not exist for all services (e.g. Kafka). On a Linux machine without GPU, use `compose-cpu` instead.

```bash
# 1. Create native conda environment (faiss-cpu, no CUDA)
conda env create -f environments/environment_nocuda.yml
conda activate talent_matching_nocuda
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# 2. Run only Kafka in Docker (macOS needs platform flag)
docker compose -f docker-compose-macos.yml up kafka -d

# 3. Set Kafka broker for native use (outside Docker network)
export KAFKA_BROKER=localhost:9092

# 4. Run scripts natively (will auto-detect MPS)
python -c "from demo.scripts.matching_utils import get_device; print(get_device())"
# Should print: mps

# 5. Run the pipeline
bash run_pipeline_quick_train.sh

# 6. After ingestion finishes (scripts 01), you can free Docker memory:
# docker compose -f docker-compose-macos.yml down
# The rest of the pipeline uses only local parquet files, no Kafka needed.

# 7. Run Streamlit
streamlit run demo/app.py
```

**Notes:**
- Always set `export KAFKA_BROKER=localhost:9092` before running scripts natively, since the default (`kafka-broker:29092`) only works inside the Docker network. Add it to `~/.zshrc` to make it permanent.
- If you get a Java version error with Spark: `conda install openjdk=17`
- spaCy models are downloaded without version pinning (e.g. `python -m spacy download en_core_web_sm`) so they auto-match the installed spaCy version
- All scripts automatically detect the best available device (CUDA > MPS > CPU) via the shared `get_device()` function, so no code changes are needed between platforms
- Running the full pipeline inside Docker on a 16GB MacBook will likely cause Spark OOM errors. Native execution is recommended.
- After Kafka ingestion completes, Docker can be stopped to free memory for the heavier pipeline stages

---

# Pipeline Stages

Two bash scripts are available to run the full pipeline:

- **`run_full_pipeline.sh`** - runs all 4 phases with full hyperparameter sweep and builds the full 1.35M job index (~35 min on GPU). **Required for the demo.**
- **`run_pipeline_quick_train.sh`** - same 4 phases but uses `--quick` flags: single training run (best config lr=1e-5, warmup=0.1) instead of 6-config sweep, and builds a 10K sample index. Good for verifying the pipeline works end-to-end, but the quick index is too small for the demo.

## Phase 1: CV Ingestion (Scripts 01-05)
- Kafka ingestion: 4,817 CVs from HuggingFace JSONL
- Spark processing: normalize fields, aggregate skills, build embedding strings
- Train/val/test splits for CV IDs

## Phase 2: Job Ingestion (Scripts 01-06)
- Kafka ingestion: 1.35M jobs in 41s (33K msg/sec)
- Spark processing: JOIN skills + filter (46s)
- spaCy NLP: Extract skills from Indeed/Glassdoor (16s)

## Phase 3: Training + Job Index (Scripts 07-10)
- Data preparation: Train/val/test splits (80/10/10, stratified by ISCO-08)
- Fine-tuning: E5 model with MNR + Matryoshka loss
- Hyperparameters: LR=1e-05, Batch=64, Epochs=10
- Results: Val_Loss=0.528, Recall@50=79.42%
- Re-encode ALL 1.35M jobs with fine-tuned model (9.3 min)
- Output: jobs_full_index.faiss (~4.1GB, 1.35M vectors x 768 dimensions)

## Phase 4: CV Index + Validation (Script 11)
- Build cvs_index.faiss (4,817 CV vectors, 15MB)
- Build cvs_embedded.parquet
- Run retrieval validation on 447 test pairs

---

# User Interaction

The system supports 3 interaction modes, all sharing the same matching logic (`matching_utils.py`) and feedback storage (`feedback_storage.py`):

## 1. Streamlit Web App (recommended for demos)
```bash
streamlit run demo/app.py
```
- **Job Seeker Tab**: paste CV or upload PDF/DOCX/TXT, get top-N matching jobs with match percentages, skills overlap, and action buttons (Apply/Save/Not Interested)
- **Recruiter Tab**: paste job description, get top-N matching CVs with scores and feedback buttons
- **Pipeline Overview Tab**: system architecture diagrams, dataset statistics, feedback analytics, and model retraining

## 2. CLI Scripts (terminal-based)
```bash
# CV -> Jobs matching
python demo/demo_scripts/12_interactive_matching.py
python demo/demo_scripts/12_interactive_matching.py --cv path/to/cv.pdf

# Job -> CVs matching
python demo/demo_scripts/13_recruiter_matching.py
python demo/demo_scripts/13_recruiter_matching.py --job path/to/job.txt

# Validation metrics
python demo/demo_scripts/11_retrieval_demo.py

# TF-IDF baseline comparison
python demo/demo_scripts/15_tfidf_baseline.py

# Feedback-driven retraining
python demo/demo_scripts/14_feedback_retraining.py
```

## 3. Feedback Loop
- User actions (apply, save, not interested) are stored in SQLite with weighted signals
- After 50+ actions, model can be retrained mixing 20% feedback with 80% original data
- New skills discovered from user uploads are tracked for dictionary updates

## Utility Scripts
```bash
# Download raw data and/or training outputs from Google Drive
bash scripts/download_gdrive.sh

# Backup ALL outputs before re-running pipeline
bash scripts/backup_outputs.sh

# Restore ALL outputs from backup
bash scripts/restore_outputs.sh

# Real-time file monitoring (streaming demo)
python demo/scripts/file_watcher.py           # watch mode
python demo/scripts/file_watcher.py --test    # create sample file
python demo/scripts/file_watcher.py --kafka   # with Kafka publishing
```

---

# Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Broker | Apache Kafka (KRaft) | Scalable data ingestion |
| Processing | PySpark 3.5.5 | Distributed data processing |
| NLP | spaCy + PhraseMatcher | Skill extraction |
| Embeddings | E5-base-v2 (fine-tuned) | Semantic representations |
| Training | Sentence-Transformers | Model fine-tuning |
| Vector Search | FAISS IndexFlatIP | Fast similarity search |
| Reranking | MS MARCO Cross-Encoder | Precision improvement |
| Feedback | SQLite (WAL mode) | User feedback storage |
| Web UI | Streamlit 1.54.0 | Interactive matching interface |
| File Monitoring | Watchdog | Real-time document ingestion |

---

# Performance Benchmarks

## Throughput
- Kafka: 33,000 msg/sec (producer)
- Spark: 29,300 jobs/sec (JOIN operations)
- GPU Encoding: 2,320 jobs/sec (RTX 3090, fp16)

## Retrieval Quality (447 validation pairs)
- Recall@1: 14.77% (66/447)
- Recall@10: 53.91% (241/447)
- Recall@50: 79.42% (355/447)

## TF-IDF Baseline Comparison (same 447 pairs)
| Metric | TF-IDF | Bi-encoder | Improvement |
|--------|--------|------------|-------------|
| Recall@1 | 9.17% | 14.77% | 1.6x |
| Recall@5 | 21.70% | 43.18% | 2.0x |
| Recall@10 | 31.32% | 53.91% | 1.7x |
| Recall@50 | 59.73% | 79.42% | 1.3x |

## Latency (1.35M vector index)
- Bi-encoder retrieval: 246ms (top-50)
- Cross-encoder reranking: 104ms (top-30)

---

# Troubleshooting

**Out of memory during encoding:**
- Reduce `CHUNK_SIZE` in `training/scripts/10_build_job_index.py`
- Or use CPU-only mode: `docker-compose-cpu.yml`

**CUDA out of memory:**
- Set `model_kwargs={"torch_dtype": torch.float16}` (already configured)
- Reduce batch size in training config

**Spark OOM on macOS Docker:**
- Run natively instead of inside Docker (see Option C above)
- Or reduce Docker Desktop memory allocation

**Re-running the pipeline from scratch:**
```bash
# 1. Backup current outputs (preserves indexes, models, feedback)
bash scripts/backup_outputs.sh

# 2. Run the full pipeline (rebuilds everything)
bash run_full_pipeline.sh

# 3. If something goes wrong, restore the backup
bash scripts/restore_outputs.sh
```

---

# Output Files

After running the pipeline, you will have:

```
training/output/
├── models/
│   ├── cv-job-matcher-e5/          # Final trained model
│   └── cv-job-matcher-e5-best/     # Best checkpoint
├── indexes/
│   ├── jobs_full_index.faiss       # 1.35M job vectors (~4.1GB)
│   ├── jobs_full_ids.npy           # Job ID mapping
│   └── cvs_index.faiss             # 4.8K CV vectors
└── embeddings/
    ├── jobs_embedded.parquet       # Job embeddings
    └── cvs_embedded.parquet        # CV embeddings
```

---

## Academic References

1. **E5 Embeddings**: Wang et al. (2024) - "Text Embeddings by Weakly-Supervised Contrastive Pre-training"
2. **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. **MNR Loss**: Henderson et al. (2017) - "Efficient Natural Language Response Suggestion for Smart Reply"
4. **Matryoshka Learning**: Kusupati et al. (2022) - "Matryoshka Representation Learning" (NeurIPS)
5. **FAISS**: Johnson et al. (2019) - "Billion-scale similarity search with GPUs"

---

# Author

This is an academic research project. For questions or collaboration:
- Author: Alessandro Scarano
- Institution: University of Trento
- GitHub: https://github.com/AlessandroScarano1
- Email: scaranoalex@gmail.com or alessandro.scarano-1@unitn.it

---

# License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- E5 embeddings by Microsoft Research
- Kaggle datasets
- Sentence-Transformers library by UKP Lab
- FAISS library by Meta AI Research
- MS MARCO dataset by Microsoft
