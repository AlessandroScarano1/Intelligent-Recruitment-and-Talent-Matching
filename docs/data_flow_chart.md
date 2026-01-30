sequenceDiagram
   autonumber
   participant Source as Data Sources
   participant Ingest as Ingestion (Kafka)
   participant Process as Processing (Spark/NLP)
   participant GPU as GPU Encoding
   participant Train as Training Pipeline
   participant Search as Faiss Search


   Note over Source: LinkedIn: 1.35M jobs<br/>CVs: 4,817 resumes


   Source->>Ingest: Script 01: Batch Load
   Note over Ingest: Topics: raw_jobs, raw_cvs<br/>1.35M records


   rect rgb(240, 248, 255)
   Ingest->>Process: Script 02: JOIN Skills
   Note over Process: LinkedIn + job_skills.csv<br/>(96% Coverage)


   Ingest->>Process: Script 03: Extract Skills
   Note over Process: Indeed/Glassdoor<br/>(PhraseMatcher)


   Process->>GPU: Script 04-06: Embed Text
   Note over GPU: Template: "Role of X at Y..."<br/>Stratified Sample
   end


   GPU-->>Train: Script 06: Output Vectors
   Note over Train: Base e5-base-v2 (768D)<br/>For training pairs creation


   rect rgb(255, 240, 245)
   Train->>Train: Script 07-08: Split & Pair
   Note over Train: Train/Val/Test (80/10/10)<br/>


   Train->>Train: Script 09: Fine-Tune
   Note over Train: MNR + Matryoshka Loss<br/>LR=1e-05, Batch=64, Val_Loss=0.5280
   end


   Train->>Search: Script 10: Re-encode & Index
   Note over Search: Re-encode ALL 1.35M jobs<br/>with fine-tuned model<br/>IndexFlatIP (484MB)


   Search->>Search: Script 11: Validation
   Note right of Search: Recall@50: 79.42%<br/>Recall@10: 53.91%<br/>Recall@1: 14.77%



