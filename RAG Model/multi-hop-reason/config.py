import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# --- Model IDs ---
# DEFAULT_MODEL_ID = "gemini-2.5-flash-preview-04-17"
DEFAULT_MODEL_ID = "gemini-2.5-pro-preview-03-25"
DEFAULT_MODEL_ID = "gemini-2.0-flash"
DEFAULT_TEXT_EMBEDDING_MODEL = "text-embedding-004"

# --- Default Paths ---
DEFAULT_VECTOR_DB_PATH = "vector_db_cnt.csv"
DEFAULT_DOCUMENTS_PATH_PATTERN = "../../DataSets/TEST_SET/*" # "CNT_Papers/*" # Example: 
DEFAULT_LOG_FILE_PATH = "cnt_rag.log"
DEFAULT_FEEDBACK_DB_PATH = "cnt_feedback_history.csv"
DEFAULT_GRAPH_DIR = "cnt_rag_graphs"
DEFAULT_TEST_QUESTIONS_PATH = "test_questions.txt" # Path to test questions

# --- Default RAG Settings ---
DEFAULT_VECTOR_DB_TYPE = "inmemory" # Options: "csv", "chroma", "inmemory"
DEFAULT_CHUNK_STRATEGY = "recursive"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TOP_K = 20
DEFAULT_MAX_HOPS = 10

# --- Caching ---
DEFAULT_USE_EMBEDDING_CACHE = True
DEFAULT_USE_LLM_CACHE = True

# --- Generation Config ---
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    # "max_output_tokens": 600,
    "response_mime_type": "text/plain",
}

# --- Other ---
# Rough estimate for context management, adjust as needed
CHAR_TO_TOKEN_RATIO = 3.5
MAX_CONTEXT_TOKENS = 7000

# Threshold for simple sequence matcher deduplication
SIMILARITY_THRESHOLD = 0.9

# Minimum chunk length to keep after cleaning
MIN_CHUNK_LENGTH = 50
