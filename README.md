# Intelligent Recruitment and Talent Matching System

A scalable end-to-end machine learning system for semantic job-CV matching using bi-encoder retrieval and cross-encoder reranking, processing 1.35M job postings and achieving 79.42% Recall@50.

# Project Overview

This system implements a semantic search engine for matching job seekers with relevant positions using:
- Dense retrieval with fine-tuned E5 embeddings (768D vectors)
- Two-stage ranking: Bi-encoder retrieval + cross-encoder reranking
- Big data pipeline: Kafka -> Spark -> GPU encoding -> FAISS indexing
- Real-world scale: 1.35M job postings, 4.8K CVs

# Key Results
- Recall@50: 79.42% | Recall@10: 53.91% | Recall@1: 14.77%
- Throughput: 2,320 encodings/sec on RTX 3090
- Query latency: 246ms for top-50 retrieval (1.35M vectors)

---

# System Architecture

```
Data Sources -> Kafka Ingestion -> Spark Processing -> GPU Encoding
                     ↓                    ↓              ↓
                 (3 partitions)     (NLP + JOIN)   (e5-base-v2)
                                                        ↓
                                              Training Pipeline
                                         (MNR + Matryoshka Loss)
                                                        ↓
                                         Re-encode ALL 1.35M jobs
                                                        ↓
                                            FAISS Index (484MB)
                                                        ↓
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
│   ├── demo_scripts/          # Scripts 11-14: Retrieval demos
│   └── scripts/               # Feedback storage & parsing
│
├── docs/                       # Architecture diagrams
├── docker-compose.yml          # GPU-enabled container setup
└── run_full_pipeline.sh        # Main execution script
```

---

# Quick Start

# Prerequisites

- Docker with GPU support (NVIDIA Docker runtime)
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 3090)
- 40GB+ disk space for datasets and models

# Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/AlessandroScarano1/Intelligent-Recruitment-and-Talent-Matching.git
cd Intelligent-Recruitment-and-Talent-Matching

# 2. Download datasets from google drive - https://drive.google.com/drive/folders/1Eb-3dJJOQ00kIC_dxB48rU5E5V5wrq5M?usp=sharing (place in ingest_job_postings/raw_data/ and ingest_cv/raw_data/ folders)
# - LinkedIn job postings CSV (1.35M rows)
# - CV JSONL file (4.8K resumes)
# - job_skills.csv mapping file
# - Glassdoor job postings CSV
# - Indeed job postings CSV

# 3. Start Kafka + Dev container
docker compose up -d

# 4. Enter the development container
docker compose exec dev bash

# 5. Run the full pipeline (inside container)
bash run_full_pipeline.sh
```

Pipeline Runtime: ~35 minutes for full execution (1.35M jobs)

---

# Pipeline Stages

# Stage 1: Data Ingestion (Scripts 01-06)
- Kafka ingestion: 1.35M jobs in 41s (33K msg/sec)
- Spark processing: JOIN skills + filter (46s)
- spaCy NLP: Extract skills from Indeed/Glassdoor (16s)

# Stage 2: Training (Scripts 07-09)
- Data preparation: Train/val/test splits (80/10/10)
- Fine-tuning: E5 model with MNR + Matryoshka loss
- Hyperparameters: LR=1e-05, Batch=64, Epochs=10
- Results: Val_Loss=0.528, Recall@50=79.42%

# Stage 3: Production Indexing (Script 10)
- Re-encode: ALL 1.35M jobs with fine-tuned model (9.3 min)
- FAISS index: IndexFlatIP for cosine similarity
- Output: 484MB index file

# Stage 4: Demo Applications (Scripts 11-14)
```bash
# CV → Jobs matching
python demo/demo_scripts/12_interactive_matching.py

# Job → CVs matching
python demo/demo_scripts/13_recruiter_matching.py --> will be less accurate, for now

# Validation metrics
python demo/demo_scripts/11_retrieval_demo.py
```

---

# Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Broker | Apache Kafka (KRaft) | Scalable data ingestion |
| Processing | PySpark 3.5.3 | Distributed data processing |
| NLP | spaCy + PhraseMatcher | Skill extraction |
| Embeddings | E5-base-v2 (fine-tuned) | Semantic representations |
| Training | Sentence-Transformers | Model fine-tuning |
| Vector Search | FAISS IndexFlatIP | Fast similarity search |
| Reranking | MS MARCO Cross-Encoder | Precision improvement |

---

# Performance Benchmarks

# Throughput
- Kafka: 33,000 msg/sec (producer)
- Spark: 29,300 jobs/sec (JOIN operations)
- GPU Encoding: 2,320 jobs/sec (RTX 3090, fp16)

# Retrieval Quality (447 validation pairs)
- Recall@1: 14.77% (66/447)
- Recall@10: 53.91% (241/447)
- Recall@50: 79.42% (355/447)

# Latency (1.35M vector index)
- Bi-encoder retrieval: 246ms (top-50)
- Cross-encoder reranking: 104ms (top-30)

---

## Academic References

## Core Papers
1. **E5 Embeddings**: Wang et al. (2024) - "Text Embeddings by Weakly-Supervised Contrastive Pre-training"
2. **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. **MNR Loss**: Henderson et al. (2017) - "Efficient Natural Language Response Suggestion for Smart Reply"
4. **Matryoshka Learning**: Kusupati et al. (2022) - "Matryoshka Representation Learning" (NeurIPS)
5. **FAISS**: Johnson et al. (2019) - "Billion-scale similarity search with GPUs"

See `docs/` for full citation list.

---

# Configuration



# Quick Mode (Fast Testing)
```bash
bash run_full_pipeline_quick.sh
```

# Quick mode with training only
```bash
# Process datasets and one round of training
bash run_full_pipeline_quick_train.sh
```

---

# Troubleshooting

Out of memory during encoding:
- Reduce `CHUNK_SIZE` in `training/scripts/10_build_job_index.py`
- Or use CPU-only mode: `docker-compose-cpu.yml`

CUDA out of memory:
- Set `model_kwargs={"torch_dtype": torch.float16}` (already configured)
- Reduce batch size in training config

---

# Output Files

After running the pipeline, you will have:

```
training/output/
├── models/
│   ├── cv-job-matcher-e5/          # Final trained model
│   └── cv-job-matcher-e5-best/     # Best checkpoint
├── indexes/
│   ├── jobs_full_index.faiss       # 1.35M job vectors (484MB)
│   ├── jobs_full_ids.npy           # Job ID mapping
│   └── cvs_index.faiss             # 4.8K CV vectors
└── embeddings/
    ├── jobs_embedded.parquet       # Job embeddings
    └── cvs_embedded.parquet        # CV embeddings
```

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
