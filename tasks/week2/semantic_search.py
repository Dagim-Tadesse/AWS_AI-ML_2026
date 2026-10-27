"""
AWS AI/ML Specialization - Week 2: Mini Semantic Search Engine
==============================================================
This module implements a complete Semantic Search pipeline fulfilling all Week 2 requirements:
1. Loads raw FAQ dataset (CSV file with automatic fallback list for Colab/portability)
2. Cleans data with modular named functions (missing values, duplicates, text normalization, type conversion, outlier filtering)
3. Vectorized text analysis & metric calculations using NumPy (mean, median, std, percentile, where, dot, linalg.norm)
4. Chunks text using sliding window with overlap for semantic representation
5. Embeds and indexes documents into ChromaDB vector database
6. Performs semantic search queries and returns top 3 matches with similarity scoring
"""

import os
import re
import html
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


# ==============================================================================
# 1. DATA LOADING FUNCTION
# ==============================================================================

# Embedded fallback dataset ensuring 100% portable, out-of-the-box execution in Google Colab
FALLBACK_FAQS = [
    {"id": 1, "category": "Compute", "question": "What is AWS Lambda?", "answer": "AWS Lambda is a serverless, event-driven compute service that lets you run code for virtually any type of application or backend service without provisioning or managing servers."},
    {"id": 2, "category": "Storage", "question": "What is Amazon S3 and what is it used for?", "answer": "Amazon Simple Storage Service (Amazon S3) is an object storage service offering industry-leading scalability, data availability, security, and performance. You can store and protect any amount of data for data lakes, websites, mobile applications, backup and restore, and big data analytics."},
    {"id": 3, "category": "Machine Learning", "question": "What is Amazon SageMaker?", "answer": "Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. It removes the heavy lifting from each step of the machine learning process."},
    {"id": 4, "category": "Generative AI", "question": "What is Amazon Bedrock?", "answer": "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like Anthropic, Meta, Mistral, and Amazon via a single API, along with capabilities to build generative AI applications with security and privacy."},
    {"id": 5, "category": "Security", "question": "How does AWS IAM work?", "answer": "AWS Identity and Access Management (IAM) helps you securely manage access to AWS services and resources by creating users, groups, and roles, and applying fine-grained permissions using IAM JSON policies."},
    {"id": 6, "category": "Database", "question": "What is Amazon DynamoDB?", "answer": "Amazon DynamoDB is a fully managed, serverless, key-value NoSQL database designed to run high-performance applications at any scale with single-digit millisecond latency."},
    {"id": 7, "category": "Machine Learning", "question": "How do I prepare training data in SageMaker?", "answer": "You can use SageMaker Data Wrangler to clean, normalize, and visualize data without writing code, or run SageMaker Processing jobs with Pandas, PySpark, and Scikit-learn scripts."},
    {"id": 8, "category": "Generative AI", "question": "What are Embeddings and Vector Databases?", "answer": "Embeddings are numerical vector representations of text, images, or audio that capture semantic meaning. Vector databases like Chroma store these high-dimensional vectors and perform fast approximate nearest neighbor (ANN) searches to power semantic search and Retrieval-Augmented Generation (RAG)."},
    {"id": 9, "category": "Compute", "question": "When should I use Amazon EC2 versus AWS Lambda?", "answer": "Use EC2 when you need long-running dedicated instances, custom OS-level kernels, GPU compute, or stateful applications. Use Lambda for short-lived, event-triggered stateless workloads that scale automatically to zero."},
    {"id": 10, "category": "Storage", "question": "What are S3 Storage Classes?", "answer": "Amazon S3 offers a range of storage classes including S3 Standard for frequent access, S3 Intelligent-Tiering for automatic cost savings, S3 Standard-IA for infrequent access, and S3 Glacier for long-term archiving."},
    {"id": 11, "category": "Networking", "question": "What is an Amazon VPC?", "answer": "Amazon Virtual Private Cloud (Amazon VPC) enables you to launch AWS resources into a virtual network that you have defined, providing complete control over IP address ranges, subnets, route tables, and network gateways."},
    {"id": 12, "category": "Machine Learning", "question": "What is SageMaker JumpStart?", "answer": "SageMaker JumpStart is a machine learning hub that provides pretrained, open-source foundation models, built-in algorithms, and prebuilt solutions that can be deployed with just a few clicks or lines of code."},
    {"id": 13, "category": "Security", "question": "What is the AWS Shared Responsibility Model?", "answer": "Under the Shared Responsibility Model, AWS is responsible for security OF the cloud (hardware, software, facilities), while the customer is responsible for security IN the cloud (customer data, IAM configurations, operating systems)."},
    {"id": 14, "category": "Generative AI", "question": "What is Retrieval-Augmented Generation (RAG)?", "answer": "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models by fetching relevant facts from an external knowledge base or vector database before generating a response."},
    {"id": 15, "category": "Monitoring", "question": "What is Amazon CloudWatch?", "answer": "Amazon CloudWatch is a monitoring and observability service that provides data and actionable insights for AWS, hybrid, and on-premises applications and infrastructure resources."},
    {"id": 16, "category": "DevOps", "question": "What is AWS CodePipeline?", "answer": "AWS CodePipeline is a fully managed continuous delivery service that helps you automate your release pipelines for fast and reliable application and infrastructure updates."},
    {"id": 17, "category": "Generative AI", "question": "What is Amazon Bedrock Knowledge Bases?", "answer": "Knowledge Bases for Amazon Bedrock connects foundation models to company data sources for Retrieval-Augmented Generation (RAG) workflows, managing chunking, embedding generation, vector storage, and query retrieval automatically."},
    {"id": 18, "category": "Database", "question": "What is Amazon Aurora?", "answer": "Amazon Aurora is a MySQL- and PostgreSQL-compatible relational database built for the cloud, combining enterprise performance with the simplicity and cost-effectiveness of open source."},
    {"id": 19, "category": "Compute", "question": "What is AWS Fargate?", "answer": "AWS Fargate is a serverless, pay-as-you-go compute engine that lets you build and run containers without managing servers, compatible with both Amazon ECS and Amazon EKS."},
    {"id": 20, "category": "Generative AI", "question": "How does semantic search differ from keyword search?", "answer": "Keyword search matches exact words using lexical algorithms like BM25, while semantic search uses dense vector embeddings to understand the contextual meaning and intent behind user queries, matching concepts even when distinct words are used."},
    # Raw anomalies to demonstrate cleaning functions:
    {"id": 21, "category": "Compute", "question": "What is AWS Lambda?", "answer": "AWS Lambda is a serverless, event-driven compute service that lets you run code for virtually any type of application or backend service without provisioning or managing servers."},
    {"id": 22, "category": "Security", "question": "   how does aws iam work?   ", "answer": "AWS Identity and Access Management (IAM) helps you securely manage access to AWS services &amp; resources &lt;br&gt; with permissions."},
    {"id": 23, "category": "Machine Learning", "question": None, "answer": None},
    {"id": 24, "category": "General", "question": "N/A", "answer": "   "},
    {"id": 25, "category": "General", "question": "What is AI?", "answer": "AI"}
]


def load_faq_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw FAQ data from a CSV file if present; otherwise falls back to
    the embedded list of dictionaries. Guarantees raw data loading anywhere.
    """
    if file_path and os.path.exists(file_path):
        print(f"[Data Loader] Loading raw dataset from CSV: {file_path}")
        df = pd.read_csv(file_path)
    else:
        print("[Data Loader] CSV not found at target path. Initializing from raw dictionary list.")
        df = pd.DataFrame(FALLBACK_FAQS)
    print(f"[Data Loader] Successfully loaded {len(df)} raw records.")
    return df


# ==============================================================================
# 2. NAMED CLEANING FUNCTIONS (Data Quality & Preprocessing)
# ==============================================================================

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing, null, or empty string values across question and answer fields.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    # Replace empty/whitespace strings with NaN
    df_clean["question"] = df_clean["question"].replace(r"^\s*$", np.nan, regex=True)
    df_clean["answer"] = df_clean["answer"].replace(r"^\s*$", np.nan, regex=True)
    
    # Drop rows where question or answer is missing or placeholder 'N/A'
    df_clean = df_clean.dropna(subset=["question", "answer"])
    df_clean = df_clean[~df_clean["question"].str.upper().isin(["N/A", "NA", "NULL", "NONE"])]
    
    dropped = initial_len - len(df_clean)
    print(f"[Clean 1: Missing Values] Removed {dropped} incomplete/empty records. Remaining: {len(df_clean)}")
    return df_clean


def remove_duplicate_faqs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicate FAQ entries based on normalized question text.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    # Create temporary normalized key for accurate duplicate detection
    norm_key = df_clean["question"].astype(str).str.strip().str.lower()
    df_clean = df_clean.loc[~norm_key.duplicated(keep="first")]
    
    dropped = initial_len - len(df_clean)
    print(f"[Clean 2: Deduplication] Removed {dropped} duplicate questions. Remaining: {len(df_clean)}")
    return df_clean


def normalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes textual columns: unescapes HTML entities, trims whitespace,
    standardizes spacing, and formats sentence casing.
    """
    df_clean = df.copy()
    
    def _sanitize(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        # Unescape HTML entities (e.g. &amp; -> &)
        clean = html.unescape(text)
        # Remove HTML tags (e.g. <br>)
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Collapse multiple spaces, newlines, and tabs
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    for col in ["question", "answer", "category"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(_sanitize)
            
    print(f"[Clean 3: Text Normalization] Sanitized whitespace, HTML artifacts, and casing across columns.")
    return df_clean


def standardize_types_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts data types, standardizes categorical values, and resets clean sequential IDs.
    """
    df_clean = df.copy()
    df_clean["id"] = np.arange(1, len(df_clean) + 1, dtype=np.int32)
    df_clean["category"] = df_clean["category"].astype(str).str.title()
    df_clean["question"] = df_clean["question"].astype(str)
    df_clean["answer"] = df_clean["answer"].astype(str)
    
    print(f"[Clean 4: Type Conversion] Standardized IDs to int32 and categories to Title Case.")
    return df_clean


def filter_outlier_entries(df: pd.DataFrame, min_char_length: int = 15) -> pd.DataFrame:
    """
    Filters out outlier / low-quality entries where question or answer is too short to be meaningful.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    valid_mask = (df_clean["question"].str.len() >= min_char_length) & (df_clean["answer"].str.len() >= min_char_length)
    df_clean = df_clean[valid_mask].reset_index(drop=True)
    
    dropped = initial_len - len(df_clean)
    print(f"[Clean 5: Outlier Filter] Filtered {dropped} entries shorter than {min_char_length} chars. Final records: {len(df_clean)}")
    return df_clean


def run_data_cleaning_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full suite of cleaning functions sequentially.
    """
    print("\n--- Starting Data Cleaning Pipeline ---")
    df = clean_missing_values(df_raw)
    df = remove_duplicate_faqs(df)
    df = normalize_text_fields(df)
    df = filter_outlier_entries(df, min_char_length=15)
    df = standardize_types_and_ids(df)
    print("--- Data Cleaning Pipeline Completed Successfully ---\n")
    return df


# ==============================================================================
# 3. TEXT CHUNKING
# ==============================================================================

def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 40) -> List[str]:
    """
    Splits text into sliding semantic chunks of words with overlapping context.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_str = " ".join(words[start:end])
        chunks.append(chunk_str)
        if end >= len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def prepare_chunks_for_indexing(
    df: pd.DataFrame, chunk_size: int = 150, chunk_overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Creates individual indexed chunks from cleaned FAQs, combining question and answer
    context for optimal semantic matching while preserving source metadata.
    """
    documents = []
    for _, row in df.iterrows():
        full_text = f"Category: {row['category']} | Question: {row['question']} | Answer: {row['answer']}"
        chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk_content in enumerate(chunks):
            documents.append({
                "chunk_id": f"faq_{row['id']}_chk_{idx}",
                "faq_id": int(row["id"]),
                "category": row["category"],
                "question": row["question"],
                "text": chunk_content,
                "char_length": len(chunk_content),
                "word_count": len(chunk_content.split())
            })
    return documents


# ==============================================================================
# 4. NUMPY STATISTICAL PROCESSING & VECTOR OPERATIONS
# ==============================================================================

def compute_chunk_statistics(lengths: np.ndarray) -> Dict[str, float]:
    """
    Calculates detailed distribution statistics using NumPy operations:
    - np.mean: Central tendency
    - np.median: 50th percentile robust measure
    - np.std: Dispersion and variance
    - np.percentile: Quantiles (25th, 75th, 90th, 95th)
    - np.min / np.max: Boundary extremes
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    q25, q50, q75, q95 = np.percentile(lengths, [25, 50, 75, 95])
    
    stats = {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(q95),
        "iqr": float(q75 - q25)
    }
    return stats


def detect_outlier_chunks_numpy(lengths: np.ndarray, z_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies outlier chunk sizes using NumPy z-score calculations and np.where.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    mean = np.mean(lengths)
    std = np.std(lengths) if np.std(lengths) > 0 else 1.0
    
    # Vectorized z-score calculation
    z_scores = np.abs((lengths - mean) / std)
    
    # Locate outliers using np.where
    outlier_indices = np.where(z_scores > z_threshold)[0]
    normal_indices = np.where(z_scores <= z_threshold)[0]
    
    return outlier_indices, normal_indices


def compute_cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity calculation using NumPy:
    cosine_sim = dot(u, v) / (norm(u) * norm(v))
    """
    q_norm = np.linalg.norm(query_vec)
    d_norms = np.linalg.norm(doc_vecs, axis=1)
    
    # Guard against zero-division
    d_norms = np.where(d_norms == 0, 1e-10, d_norms)
    if q_norm == 0:
        q_norm = 1e-10
        
    dot_products = np.dot(doc_vecs, query_vec)
    similarities = dot_products / (d_norms * q_norm)
    return similarities


def filter_by_threshold_numpy(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Filters similarity scores above a given confidence threshold using np.where.
    """
    qualified_indices = np.where(scores >= threshold)[0]
    return qualified_indices


# ==============================================================================
# 5. CHROMADB VECTOR STORE & SEMANTIC SEARCH
# ==============================================================================

class MiniSemanticSearchEngine:
    """
    Semantic Search Engine wrapping ChromaDB for indexing and querying.
    """
    def __init__(self, collection_name: str = "aws_faq_knowledge_base", persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client (persistent or in-memory)
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()
            
        # Use Chroma's default sentence-transformers embedding function (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        # Delete existing collection if present to guarantee clean repeatable state
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "AWS Cloud and AI/ML FAQ Semantic Knowledge Base", "hnsw:space": "cosine"}
        )
        print(f"[ChromaDB] Initialized collection '{self.collection_name}' with Cosine metric.")

    def index_documents(self, chunk_records: List[Dict[str, Any]]) -> None:
        """
        Embeds and adds text chunks with rich metadata into ChromaDB.
        """
        ids = [rec["chunk_id"] for rec in chunk_records]
        documents = [rec["text"] for rec in chunk_records]
        metadatas = [
            {
                "faq_id": rec["faq_id"],
                "category": rec["category"],
                "question": rec["question"],
                "char_length": rec["char_length"],
                "word_count": rec["word_count"]
            }
            for rec in chunk_records
        ]
        
        print(f"[ChromaDB] Generating embeddings and indexing {len(documents)} chunks...")
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[ChromaDB] Indexing complete! Total items in collection: {self.collection.count()}")

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes a semantic vector query against ChromaDB and retrieves top_k matches.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def print_top_matches(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs query and prints beautifully formatted top 3 matches with scores.
        """
        raw_results = self.query(query_text, top_k=top_k)
        
        print("\n" + "=" * 80)
        print(f"🔍 QUERY: \"{query_text}\"")
        print("=" * 80)
        
        matched_items = []
        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]
        
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            # For cosine distance, similarity = 1 - distance
            similarity_score = max(0.0, 1.0 - float(dist))
            
            print(f"\n[Match #{rank}] Score: {similarity_score:.4f} (Cosine Distance: {dist:.4f})")
            print(f"  📁 Category: {meta['category']}")
            print(f"  ❓ FAQ Question: {meta['question']}")
            print(f"  📄 Content Snippet: {doc[:160]}...")
            
            matched_items.append({
                "rank": rank,
                "score": similarity_score,
                "distance": dist,
                "category": meta["category"],
                "question": meta["question"],
                "snippet": doc
            })
            
        print("-" * 80)
        return matched_items


# ==============================================================================
# 6. DEMONSTRATION & PIPELINE EXECUTION
# ==============================================================================

def main():
    print("================================================================================")
    print("      AWS AI/ML COHORT - WEEK 2: MINI SEMANTIC SEARCH & NUMPY LAB               ")
    print("================================================================================")
    
    # Step 1: Load raw dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "raw_dataset.csv")
    df_raw = load_faq_dataset(csv_path)
    
    # Step 2: Clean data using modular functions
    df_clean = run_data_cleaning_pipeline(df_raw)
    
    # Step 3: Chunk text
    chunks = prepare_chunks_for_indexing(df_clean, chunk_size=150, chunk_overlap=30)
    print(f"[Chunking] Generated {len(chunks)} semantic chunks from {len(df_clean)} FAQs.")
    
    # Step 4: Statistical analysis with NumPy
    chunk_lengths = np.array([chk["char_length"] for chk in chunks], dtype=np.float64)
    stats = compute_chunk_statistics(chunk_lengths)
    
    print("\n--- NumPy Computed Distribution Statistics for Text Chunks ---")
    print(f"  • Total Chunks Analyzed : {len(chunk_lengths)}")
    print(f"  • Mean Character Length : {stats['mean']:.2f}")
    print(f"  • Std Deviation         : {stats['std']:.2f}")
    print(f"  • Median Length         : {stats['median']:.2f}")
    print(f"  • 25th Percentile (Q1)  : {stats['p25']:.2f}")
    print(f"  • 75th Percentile (Q3)  : {stats['p75']:.2f}")
    print(f"  • 95th Percentile       : {stats['p95']:.2f}")
    print(f"  • Min / Max Length      : {stats['min']:.0f} / {stats['max']:.0f}")
    
    # Outlier detection with NumPy where
    outliers, normal = detect_outlier_chunks_numpy(chunk_lengths, z_threshold=2.0)
    print(f"  • Outlier Chunks (|z| > 2.0 via np.where): {len(outliers)} found")
    print("----------------------------------------------------------------\n")
    
    # Step 5: Index into ChromaDB
    chroma_dir = os.path.join(script_dir, "chroma_db")
    search_engine = MiniSemanticSearchEngine(
        collection_name="aws_faq_search",
        persist_directory=chroma_dir
    )
    search_engine.index_documents(chunks)
    
    # Step 6: Query and print Top 3 matches
    example_queries = [
        "How do I build and deploy machine learning models on AWS?",
        "What service provides serverless compute without managing servers?",
        "How does vector search and RAG work for generative AI applications?",
        "What is the best way to securely store objects and big data in the cloud?"
    ]
    
    print("\n================================================================================")
    print("                    RUNNING SEMANTIC SEARCH QUERIES                             ")
    print("================================================================================")
    
    for query in example_queries:
        search_engine.print_top_matches(query, top_k=3)
        
    print("\n[Complete] Mini Semantic Search pipeline executed successfully!")


if __name__ == "__main__":
    main()
