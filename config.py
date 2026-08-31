from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_NAME = str(
    BASE_DIR
    / "vector_db_qwen_750"
    / "vector_db_qwen"
)

COLLECTION_NAME = "annual_reports"
CANDIDATE_K = 40
FINAL_K = 10

MODEL = "gpt-4.1-mini-2025-04-14"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"

EMBEDDING_DEVICE = "cuda:0"
RERANKER_DEVICE = "cuda:1"
RERANKER_BATCH_SIZE = 20