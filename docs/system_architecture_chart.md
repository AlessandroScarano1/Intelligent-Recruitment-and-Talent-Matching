graph LR
 subgraph DataSources["Data Sources"]
     direction TB
     LinkedIn["LinkedIn CSV<br/>1.35M jobs"]
     Indeed["Indeed/Glassdoor<br/>~250 jobs"]
     CVs["CV JSONL<br/>4K resumes"]
 end








 subgraph Ingestion["Ingestion Layer"]
     direction TB
     JobProd["Job Producer"]
     CVProd["CV Producer"]
     Kafka{{"Kafka KRaft<br/>3 partitions"}}
 end








 subgraph Processing["Processing Layer"]
     direction TB
     Spark["Spark <br/>JOIN + Filter"]
     SpaCy["spaCy NLP<br/>PhraseMatcher"]
     EmbStr["Embedding Builder<br/>Template strings"]
 end








 subgraph Encoding["Encoding Layer"]
     direction TB
     GPU["GPU Encoder<br/>e5-base-v2"]
     PyArrow["PyArrow Writer<br/>Parquet output"]
 end








 subgraph Training["Training Pipeline"]
     direction TB
     STTrainer["ST Trainer<br/>MNR + Matryoshka<br/>(165k pairs)"]
     FineTuned["Fine-tuned Model<br/>cv-job-matcher-e5"]
 end


 subgraph Indexing["Production Indexing"]
     direction TB
     ReEncode["Re-encode ALL jobs<br/>with fine-tuned model"]
     FAISS["Build FAISS Index<br/>IndexFlatIP"]
 end








 subgraph Storage["Storage"]
     direction TB
     Parquet[("Training Data<br/>165k pairs")]
     JobsData[("Jobs Parquet<br/>1.35M jobs")]
     FAISSIndex[("FAISS Index<br/>1.35M vectors, 484MB")]
     SQLite[("SQLite<br/>Feedback DB")]
 end








 subgraph Matching["Matching Layer<br/>"]
     direction TB
     BiEnc["Bi-encoder<br/>Retrieval<br/>(fine-tuned)"]
     CrossEnc["Cross-encoder<br/>Reranking"]
 end








 %% Flow Connections
 LinkedIn & Indeed --> JobProd
 CVs --> CVProd








 JobProd & CVProd --> Kafka








 Kafka --> Spark & SpaCy








 Spark & SpaCy --> EmbStr








 EmbStr --> GPU
 GPU --> PyArrow








 PyArrow --> Parquet
 PyArrow --> JobsData
 Parquet --> STTrainer


 STTrainer --> FineTuned
 FineTuned --> ReEncode
 JobsData --> ReEncode
 ReEncode --> FAISS
 FAISS --> FAISSIndex








 FAISSIndex --> BiEnc
 BiEnc --> CrossEnc








 CrossEnc --> SQLite








 %% Styling
 style DataSources fill:#e1f5ff,stroke:#01579b
 style Ingestion fill:#fff4e1,stroke:#e65100
 style Processing fill:#f0e1ff,stroke:#4a148c
 style Encoding fill:#e1ffe8,stroke:#1b5e20
 style Training fill:#ffe1f5,stroke:#880e4f
 style Indexing fill:#ffebee,stroke:#c62828
 style Matching fill:#fff0e1,stroke:#bf360c
 style Storage fill:#e8e8e8,stroke:#424242
 style Kafka shape:hexagon,fill:#212121,color:#fff,stroke:#000





