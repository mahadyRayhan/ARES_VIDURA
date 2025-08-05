# Necessary imports (ensure these are installed: pip install ...)
import os
import glob
import re
import time
import logging
import numpy as np
import pandas as pd
import uuid
from graphviz import Digraph
from typing import Any, Dict, List, Tuple, Optional, Union
import PyPDF2
import docx # <-- Added for .docx support
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from difflib import SequenceMatcher # For simple deduplication

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

# Default Model IDs
DEFAULT_MODEL_ID = "gemini-1.5-flash" # Adjusted default
DEFAULT_TEXT_EMBEDDING_MODEL = "text-embedding-004"

# Default Paths for CNT use case
DEFAULT_VECTOR_DB_PATH = "vector_db_cnt.csv"
DEFAULT_DOCUMENTS_PATH_PATTERN = "CNT_Papers/*" # Example: Modify to your papers' location
DEFAULT_LOG_FILE_PATH = "cnt_rag.log"
DEFAULT_FEEDBACK_DB_PATH = "cnt_feedback_history.csv"
DEFAULT_GRAPH_DIR = "cnt_rag_graphs"

class CNTRagSystem: # Renamed Class
    """
    Enhanced RAG system adapted for querying Carbon Nanotube (CNT) research papers
    and experimental data. Handles resource loading, query processing, multi-hop
    reasoning, evaluation, feedback, and optimized vector storage options.
    """
    def __init__(self,
                 model_id: str = DEFAULT_MODEL_ID,
                 embedding_model_id: str = DEFAULT_TEXT_EMBEDDING_MODEL,
                 vector_db_type: str = "csv", # Options: "csv", "chroma", "inmemory"
                 vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
                 documents_path: str = DEFAULT_DOCUMENTS_PATH_PATTERN,
                 log_level: str = "INFO",
                 log_file: str = DEFAULT_LOG_FILE_PATH,
                 feedback_db_path: str = DEFAULT_FEEDBACK_DB_PATH,
                 graph_dir: str = DEFAULT_GRAPH_DIR, # Added graph dir config
                 chunk_strategy: str = "recursive", # Future: "semantic"
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 use_embedding_cache: bool = True,
                 use_llm_cache: bool = True):
        """
        Initialize the CNTRagSystem.

        Args:
            model_id: Generative model ID for text generation.
            embedding_model_id: Text embedding model ID.
            vector_db_type: Type of vector database ("csv", "chroma", "inmemory").
            vector_db_path: Path for CSV vector database file.
            documents_path: Glob pattern for source documents (PDF, DOCX).
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_file: Path for the log file.
            feedback_db_path: Path for the feedback CSV file.
            graph_dir: Directory to save reasoning graphs.
            chunk_strategy: Strategy for text chunking ("recursive").
            chunk_size: Target size for text chunks.
            chunk_overlap: Overlap between consecutive chunks.
            use_embedding_cache: Enable in-memory caching for embeddings.
            use_llm_cache: Enable in-memory caching for LLM responses.
        """
        self._setup_logging(log_level, log_file)
        self.logger.info(f"--- Initializing CNTRagSystem (PID: {os.getpid()}) ---")
        self.logger.info(f"Using Generative Model: {model_id}")
        self.logger.info(f"Using Embedding Model: {embedding_model_id}")
        self.logger.info(f"Vector DB Type: {vector_db_type}")
        self.logger.info(f"Chunk Strategy: {chunk_strategy} (Size: {chunk_size}, Overlap: {chunk_overlap})")

        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.vector_db_type = vector_db_type.lower()
        self.vector_db_path = vector_db_path
        self.documents_path_pattern = documents_path
        self.feedback_db_path = feedback_db_path
        self.graph_dir = graph_dir # Store graph dir
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Caching
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self.use_llm_cache = use_llm_cache
        self.llm_cache: Dict[str, str] = {}

        # Feedback
        self.feedback_history: List[Dict[str, Any]] = []

        try:
            self._initialize_clients()
            self._initialize_vector_db()
            self._load_feedback_history()
            self.logger.info("--- CNTRagSystem Initialized Successfully ---")
        except Exception as e:
            self.logger.exception(f"CRITICAL ERROR during CNTRagSystem Initialization: {e}")
            raise # Re-raise after logging

    # --- Logging Setup (Unchanged) ---
    def _setup_logging(self, log_level_str: str, log_file: str):
        """Set up proper logging instead of print statements."""
        numeric_level = getattr(logging, log_level_str.upper(), None)
        if not isinstance(numeric_level, int):
            logging.basicConfig(level=logging.INFO) # Default if level is invalid
            logging.warning(f'Invalid log level: {log_level_str}. Defaulting to INFO.')
            numeric_level = logging.INFO

        self.logger = logging.getLogger("CNTRAG") # Changed logger name slightly
        self.logger.setLevel(numeric_level)
        self.logger.propagate = False # Prevent duplicate logs in root logger

        # Clear existing handlers if re-initializing
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler for persistent logs
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            fh = logging.FileHandler(log_file, mode='a') # Append mode
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
            self.logger.error(f"Failed to set up file handler at {log_file}: {e}")

        self.logger.info(f"Logging initialized at level {log_level_str.upper()} to console and {log_file}")

    # --- Formatting and Visualization (Graph generation path updated) ---
    def _format_reasoning_trace(self, reasoning_trace: List[str]) -> str:
        # (Keep the existing formatting logic, it's general)
        """Formats the reasoning trace list into a more readable string."""
        formatted = ["**Reasoning Process:**\n" + "-"*20]
        hop_details = []
        current_hop = 0

        for step in reasoning_trace:
            step = step.strip()
            if step == "START":
                formatted.append("1. Initial Query Transformation/Expansion")
                continue
            elif step.startswith("--- Hop"):
                if hop_details: # Format previous hop before starting new one
                    formatted.append(f"\n**Hop {current_hop}:**")
                    formatted.extend([f"  - {d}" for d in hop_details])
                    hop_details = [] # Reset for next hop
                current_hop = int(step.split(" ")[2]) if len(step.split(" ")) > 2 else current_hop + 1
            elif step.startswith("Retrieving with query"):
                hop_details.append(f"Query: {step.split('->')[-1].strip()}")
            elif step.startswith("Retrieved"):
                hop_details.append(f"Retrieval: {step.split(':')[-1].strip()}")
            elif step.startswith("Added"):
                 hop_details.append(f"Context Mgmt: {step.split(':')[-1].strip()}")
            elif step.startswith("Reasoning result"):
                action_value = step.split('->')[-1].strip()
                hop_details.append(f"Reasoning Action: {action_value}")
            elif step.startswith("Reasoning ->"): # Summarizes the decision
                pass # Often redundant with "Reasoning Action", skip for cleaner output
            elif step == "--- Final Answer Generation ---":
                if hop_details: # Format last hop
                    formatted.append(f"\n**Hop {current_hop}:**")
                    formatted.extend([f"  - {d}" for d in hop_details])
                formatted.append("\n**Final Answer Generation**")
            elif step.startswith("Confidence Score:") or step.startswith("Evaluation Metrics:"):
                 formatted.append(f"  - {step}")
            elif step.startswith("Max hops"):
                 hop_details.append(step) # Add this as a note in the last hop
            else:
                 # Add other steps if they don't fit known patterns
                if current_hop > 0 and step not in ["START", "--- Final Answer Generation ---"]:
                    hop_details.append(f"Step: {step}")

        # Append details of the last hop if any remain
        if hop_details and (not formatted or not formatted[-1].startswith("**Hop")):
            formatted.append(f"\n**Hop {current_hop}:**")
            formatted.extend([f"  - {d}" for d in hop_details])

        formatted.append("-"*20)
        return "\n".join(formatted)

    # In class CNTRagSystem:

    def _generate_hop_graph(self, reasoning_trace: List[str], query_history: List[str]) -> Optional[str]:
        """Generates a Graphviz graph visualizing the RAG hops. (Revised for Readability)"""
        if not reasoning_trace:
            return None

        try:
            os.makedirs(self.graph_dir, exist_ok=True)
            graph_id = str(uuid.uuid4())[:8]
            filename = os.path.join(self.graph_dir, f"rag_hop_graph_{graph_id}")

            dot = Digraph(comment='RAG Multi-Hop Process', format='png')

            # --- Readability Improvements ---
            dot.attr(dpi='200') # Increase DPI (try 150, 200, or 300)
            dot.attr(rankdir='TB', size='12,12') # Slightly increase canvas size if needed
            dot.attr(label=f'RAG Process Flow (ID: {graph_id})', fontsize='24') # Increase title size

            # Increase default node and edge font sizes
            dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontsize='12') # Increased from 10
            dot.attr('edge', fontsize='10') # Increased from 8
            # --- End of Improvements ---


            node_counter = 0
            last_structural_node_name = f"n{node_counter}" # Tracks Start or Hop Query node

            # Start Node
            start_query = query_history[0] if query_history else "Initial Query (N/A)"
            start_label = f"Start\nQuery:\n{start_query[:150]}{'...' if len(start_query)>150 else ''}"
            # Apply increased font size specifically if needed, otherwise defaults apply
            dot.node(last_structural_node_name, start_label, shape='Mdiamond', fontsize='12')
            node_counter += 1

            hop_num = 0
            query_idx_for_hop = 0 # Index into query_history for current hop query
            current_hop_nodes = {} # Store nodes added within the current hop

            for i, step in enumerate(reasoning_trace):
                step = step.strip()

                # --- Hop Marker ---
                if step.startswith("--- Hop"):
                    hop_num = int(step.split(" ")[2]) if len(step.split(" ")) > 2 else hop_num + 1
                    current_hop_nodes = {} # Reset for new hop

                    if hop_num == 1:
                        query_idx_for_hop = 0
                    else:
                        prev_next_query = None
                        for j in range(i - 1, -1, -1):
                             prev_step = reasoning_trace[j].strip()
                             if prev_step.startswith("Hop") and "Reasoning -> NEXT_QUERY" in prev_step:
                                 match = re.search(r"NEXT_QUERY\s*=\s*'(.*?)'$", prev_step)
                                 if match:
                                     prev_next_query = match.group(1)
                                     break
                             elif prev_step.startswith("--- Hop"):
                                 break
                        found_idx = -1
                        start_search_idx = query_idx_for_hop + 1 if hop_num > 1 else 0
                        for q_idx in range(start_search_idx, len(query_history)):
                            if prev_next_query and query_history[q_idx] == prev_next_query:
                                found_idx = q_idx
                                break
                            elif not prev_next_query and q_idx > query_idx_for_hop:
                                found_idx = q_idx
                                break

                        if found_idx != -1:
                             query_idx_for_hop = found_idx
                        else:
                              self.logger.warning(f"Graph: Could not reliably determine query for Hop {hop_num}")


                    query_text = query_history[query_idx_for_hop] if query_idx_for_hop < len(query_history) else f"Query for Hop {hop_num} (N/A)"
                    query_label = f"Hop {hop_num}\nQuery:\n{query_text[:150]}{'...' if len(query_text)>150 else ''}"
                    query_node_name = f"n{node_counter}"

                    dot.node(query_node_name, query_label, fillcolor='lightgreen') # Fontsize defaults to 12
                    dot.edge(last_structural_node_name, query_node_name, label=f"Start Hop {hop_num}") # Fontsize defaults to 10
                    last_structural_node_name = query_node_name
                    current_hop_nodes['query'] = query_node_name
                    node_counter += 1

                # --- Reasoning Result Step ---
                elif step.startswith(f"Hop {hop_num}: Reasoning result"):
                    from_node = current_hop_nodes.get('query', last_structural_node_name)
                    reasoning_node_name = f"n{node_counter}"
                    action_match = re.search(r"Action='([^']*)'", step)
                    value_match = re.search(r"Value='([^']*)", step)

                    action = action_match.group(1) if action_match else "N/A"
                    value_raw = value_match.group(1).replace("'...", "") if value_match else "N/A"
                    value = value_raw[:70] + ('...' if len(value_raw) > 70 else '')

                    reasoning_label = f"Hop {hop_num} Reasoning\nAction: {action}\nValue: '{value}'"
                    dot.node(reasoning_node_name, reasoning_label, fillcolor='lightyellow') # Fontsize defaults to 12
                    dot.edge(from_node, reasoning_node_name, label="Process Context") # Fontsize defaults to 10
                    current_hop_nodes['reasoning'] = reasoning_node_name
                    node_counter += 1

                    if action == "ANSWER_COMPLETE" or action == "ERROR":
                        final_node_name = f"n{node_counter}"
                        # Use specific fontsize for final node if desired
                        dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange', fontsize='14')
                        dot.edge(reasoning_node_name, final_node_name, label=action) # Fontsize defaults to 10
                        last_structural_node_name = final_node_name
                        node_counter += 1


            # --- Final Connection Logic ---
            final_node_exists = any(dot.body[i].strip().startswith(f'"n{j}" [label="Final Answer Generation"') for i in range(len(dot.body)) for j in range(node_counter))

            if not final_node_exists:
                 connect_from_node = current_hop_nodes.get('reasoning', last_structural_node_name)
                 final_node_name = f"n{node_counter}"
                 dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange', fontsize='14')
                 label = "Max Hops Reached" if any("Max hops reached" in s for s in reasoning_trace) else "Proceed to Final"
                 if connect_from_node and connect_from_node.startswith("n"):
                     dot.edge(connect_from_node, final_node_name, label=label) # Fontsize defaults to 10
                 else:
                     self.logger.warning("Graph generation: Could not determine valid node to connect to final answer.")


            # Render the graph
            output_path = dot.render(filename=filename, view=False, cleanup=True)
            self.logger.info(f"Generated RAG hop graph (High Res): {output_path}")
            return output_path

        # Error handling remains the same
        except ImportError:
            self.logger.warning("Graphviz Python library not found. Cannot generate graph. Please run 'pip install graphviz'.")
            return None
        except FileNotFoundError:
             self.logger.error("Graphviz executable not found in PATH. Cannot generate graph. Please install Graphviz (see https://graphviz.org/download/).")
             return None
        except Exception as e:
            self.logger.exception(f"Failed to generate RAG hop graph: {e}")
            return None

    # --- Client Initialization (Unchanged) ---
    def _initialize_clients(self):
        """Initializes Google AI clients."""
        try:
            self.embedding_model_name = f"models/{self.embedding_model_id}"
            generation_config = {
                "temperature": 0.5, # Slightly lower temp for factual scientific answers
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            self.model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config=generation_config,
                # Add safety settings if needed, e.g., BLOCK_NONE for scientific text
                # safety_settings={'HARASSMENT':'BLOCK_NONE', 'HATE_SPEECH':'BLOCK_NONE', ...}
            )
            self.logger.info("Google AI Clients Initialized.")
        except Exception as e:
            self.logger.error(f"Error initializing Google AI clients: {e}")
            raise

    # --- Vector DB Initialization & Population (Largely Unchanged Logic, adapted paths/names) ---
    def _initialize_vector_db(self):
        """Initialize the appropriate vector database based on configuration."""
        self.logger.info(f"Initializing vector database (Type: {self.vector_db_type})...")
        if self.vector_db_type == "chroma":
            self._setup_chroma_db()
        elif self.vector_db_type == "csv":
            self._load_or_build_csv_db()
        elif self.vector_db_type == "inmemory":
            self._load_or_build_inmemory_db()
        else:
            self.logger.error(f"Unknown vector database type: {self.vector_db_type}")
            raise ValueError(f"Unknown vector database type: {self.vector_db_type}")
        self.logger.info("Vector database initialization complete.")

    def _setup_chroma_db(self):
        """Set up ChromaDB for vector storage."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Adjust collection name for CNT use case
            collection_name = "cnt_collection"
            self.logger.info(f"Setting up ChromaDB collection: '{collection_name}'")

            # Initialize ChromaDB client
            # Consider persistence if needed: persist_directory=".chromadb_cnt_cache"
            self.chroma_client = chromadb.Client(Settings(
                 # chroma_db_impl="duckdb+parquet", # Optional
                 # persist_directory=".chromadb_cnt_cache" # Optional persistence
            ))

            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                # metadata={"hnsw:space": "cosine"} # Use cosine distance
            )
            self.logger.info(f"ChromaDB client connected. Using collection: '{collection_name}'")

            # Check if we need to populate
            if self.collection.count() == 0:
                self.logger.info("ChromaDB collection is empty. Attempting to build/migrate...")
                 # Option 1: Migrate from existing CSV if it exists
                if os.path.exists(self.vector_db_path):
                    self.logger.info(f"Found existing CSV at '{self.vector_db_path}'. Migrating to ChromaDB...")
                    try:
                        csv_db = pd.read_csv(self.vector_db_path)
                        self._populate_chroma_from_dataframe(csv_db)
                    except Exception as migrate_err:
                        self.logger.error(f"Error migrating from CSV to ChromaDB: {migrate_err}. Trying to build from documents.")
                        self._build_index_and_populate_vector_db() # Fallback to building
                 # Option 2: Build directly from documents
                else:
                    self.logger.info(f"No existing CSV found at '{self.vector_db_path}'. Building ChromaDB index from documents...")
                    self._build_index_and_populate_vector_db()

            self.logger.info(f"ChromaDB ready. Collection '{collection_name}' contains {self.collection.count()} documents.")

        except ImportError:
            self.logger.error("ChromaDB library not installed. Please run 'pip install chromadb'")
            self.logger.warning("Falling back to CSV storage.")
            self.vector_db_type = "csv" # Change type
            self._load_or_build_csv_db() # Load/Build CSV instead
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during ChromaDB setup: {e}")
            raise

    def _populate_chroma_from_dataframe(self, df: pd.DataFrame):
        """Adds data from a Pandas DataFrame to the ChromaDB collection."""
        # (Logic is general, should work fine)
        if 'embeddings' not in df.columns or 'chunk_text' not in df.columns:
             raise ValueError("DataFrame must contain 'embeddings' and 'chunk_text' columns for ChromaDB population.")

         # Convert string embeddings if needed
        if not df.empty and isinstance(df['embeddings'].iloc[0], str):
            self.logger.info("Converting string embeddings from DataFrame...")
            df['embeddings'] = df['embeddings'].apply(lambda x: list(map(float, re.findall(r"-?\d+\.?\d*", x))) if isinstance(x, str) and x else []) # More robust parsing, handle empty strings
        elif not df.empty and isinstance(df['embeddings'].iloc[0], list):
            pass # Already in list format
        elif not df.empty and isinstance(df['embeddings'].iloc[0], np.ndarray):
            df['embeddings'] = df['embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])
        elif not df.empty:
            raise TypeError(f"Unsupported embedding type in DataFrame: {type(df['embeddings'].iloc[0])}")
        elif df.empty:
             self.logger.warning("DataFrame provided to _populate_chroma_from_dataframe is empty.")
             return # Nothing to populate

        # Prepare data for ChromaDB batch insertion
        # Filter out rows with empty embeddings or documents
        df_valid = df[df['embeddings'].apply(lambda x: isinstance(x, list) and len(x) > 0) & df['chunk_text'].apply(lambda x: isinstance(x, str) and len(x) > 0)].copy()

        if df_valid.empty:
            self.logger.warning("No valid rows (with embeddings and text) found in DataFrame for ChromaDB population.")
            return

        ids = [f"chunk_{i}" for i in df_valid.index] # Use original index if possible, or generate new ones
        embeddings = df_valid['embeddings'].tolist()
        documents = df_valid['chunk_text'].tolist()
        metadatas = [{
             "document_name": row.get('document_name', 'Unknown'),
             "page_number": int(row.get('page_number', -1)), # Ensure page is int
             "chunk_number": int(row.get('chunk_number', -1)), # Ensure chunk is int
             # Add other relevant metadata from scientific papers if extracted
             # e.g., "section_title": row.get('section_title', 'N/A')
         } for _, row in df_valid.iterrows()]

        # Add documents to ChromaDB in batches
        batch_size = 100 # ChromaDB recommends batches
        total_added = 0
        for i in range(0, len(df_valid), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_documents = documents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            if not batch_ids: continue # Skip empty batches

            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                self.logger.info(f"Added batch {i//batch_size + 1} ({len(batch_ids)} documents) to ChromaDB. Total added: {total_added}")
            except Exception as batch_err:
                # Log specific details about the batch that failed
                self.logger.error(f"Error adding batch {i//batch_size + 1} (IDs: {batch_ids[:5]}...) to ChromaDB: {batch_err}. Skipping batch.")
                # Optionally: Log sample data from the failed batch for debugging
                # self.logger.debug(f"Sample failed document: {batch_documents[0][:100] if batch_documents else 'N/A'}")
                # self.logger.debug(f"Sample failed embedding length: {len(batch_embeddings[0]) if batch_embeddings else 'N/A'}")

        self.logger.info(f"Finished populating ChromaDB from DataFrame. Added {total_added} valid documents.")


    def _load_or_build_csv_db(self):
        """Loads the vector database from CSV or builds it if not found."""
        # (Logic is general, should work fine - uses self.vector_db_path)
        self.vector_db_df: Optional[pd.DataFrame] = None # Store DataFrame here
        try:
            if os.path.exists(self.vector_db_path):
                self.logger.info(f"Loading existing vector database from {self.vector_db_path}...")
                self.vector_db_df = pd.read_csv(self.vector_db_path)
                if 'embeddings' in self.vector_db_df.columns and not self.vector_db_df.empty:
                     # Check type of first element to decide conversion
                    first_emb = self.vector_db_df['embeddings'].iloc[0]
                    if isinstance(first_emb, str):
                        self.logger.info("Converting string embeddings in CSV to numpy arrays...")
                         # Use a robust parsing method
                        self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(
                             lambda x: np.array(list(map(float, re.findall(r"-?\d+\.?\d*", x)))) if isinstance(x, str) and x else np.array([])
                         )
                    elif isinstance(first_emb, list):
                        self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(np.array)
                    elif isinstance(first_emb, np.ndarray):
                        pass # Already numpy arrays
                    else:
                        self.logger.warning(f"Embeddings column has unexpected type: {type(first_emb)}. Skipping conversion.")

                     # Basic validation
                    # Check if *any* embedding is valid after potential conversion
                    valid_embeddings = self.vector_db_df['embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
                    if valid_embeddings.any():
                         first_valid_emb = self.vector_db_df.loc[valid_embeddings.idxmax(), 'embeddings'] # Get first valid one
                         self.logger.info(f"CSV Vector database loaded successfully. Shape: {self.vector_db_df.shape}, Embedding Dim: {len(first_valid_emb)}")
                    else:
                         self.logger.warning(f"CSV Vector database loaded but seems empty or has invalid embeddings after conversion. Shape: {self.vector_db_df.shape}")
                         # Optionally treat as empty / trigger rebuild if essential
                         # self.vector_db_df = None

                elif self.vector_db_df.empty:
                     self.logger.warning(f"Loaded CSV file '{self.vector_db_path}' is empty.")
                     self.vector_db_df = None # Treat as not loaded
                else: # File exists but no 'embeddings' column
                    self.logger.error(f"CSV file '{self.vector_db_path}' is missing the 'embeddings' column.")
                    self.vector_db_df = None # Mark as failed load
            else:
                self.logger.info(f"CSV Vector Database '{self.vector_db_path}' not found. Building index...")
                self.vector_db_df = self._build_index_and_populate_vector_db() # Build and store in df
                if self.vector_db_df is not None and not self.vector_db_df.empty:
                    self._save_dataframe_to_csv(self.vector_db_df, self.vector_db_path)
                else:
                    self.logger.error("Failed to build index, CSV database remains unavailable.")

        except FileNotFoundError as fnf:
            self.logger.error(f"Error during CSV DB loading: {fnf}")
            self.vector_db_df = None
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during CSV DB loading/building: {e}")
            self.vector_db_df = None # Mark as failed

        if self.vector_db_df is None or self.vector_db_df.empty:
             self.logger.critical("CRITICAL ERROR: CSV Vector database could not be loaded or built.")
             # Depending on strictness, you might want to raise an error here
             # raise ValueError("Vector database could not be loaded or built.")


    def _load_or_build_inmemory_db(self):
        """Loads vector database into memory (from CSV if exists, else builds)."""
        # (Logic is general, should work fine)
        self.logger.info("Initializing in-memory vector database...")
         # Attempt to load from CSV first for speed if available
        if os.path.exists(self.vector_db_path):
            self.logger.info(f"Found existing CSV '{self.vector_db_path}', loading into memory.")
            self._load_or_build_csv_db() # This loads into self.vector_db_df
            if self.vector_db_df is not None and not self.vector_db_df.empty:
                self.logger.info("In-memory database populated from CSV.")
            else:
                self.logger.warning("Failed to load from CSV or CSV was invalid/empty. Attempting to build index directly into memory...")
                self.vector_db_df = self._build_index_and_populate_vector_db()
                if self.vector_db_df is not None and not self.vector_db_df.empty:
                    self.logger.info("In-memory database built successfully from documents.")
                     # Optionally save the newly built index to CSV for next time
                    self.logger.info("Saving newly built index to CSV for future use.")
                    self._save_dataframe_to_csv(self.vector_db_df, self.vector_db_path)
                else:
                    self.logger.error("Failed to build index for in-memory database.")
        else:
            # Build directly into memory
            self.logger.info(f"No CSV found at '{self.vector_db_path}'. Building index directly into memory...")
            self.vector_db_df = self._build_index_and_populate_vector_db()
            if self.vector_db_df is not None and not self.vector_db_df.empty:
                self.logger.info("In-memory database built successfully from documents.")
                 # Optionally save the newly built index to CSV for next time
                self.logger.info("Saving newly built index to CSV for future use.")
                self._save_dataframe_to_csv(self.vector_db_df, self.vector_db_path)
            else:
                self.logger.error("Failed to build index for in-memory database.")

        if self.vector_db_df is None or self.vector_db_df.empty:
             self.logger.critical("CRITICAL ERROR: In-memory Vector database could not be populated.")
             # raise ValueError("In-memory Vector database could not be populated.")


    def _build_index_and_populate_vector_db(self) -> Optional[pd.DataFrame]:
        """
        Builds the index from documents and populates the configured vector DB.
        If DB is CSV or InMemory, returns the DataFrame.
        If DB is ChromaDB, populates the collection directly.
        """
        # (Logic is general, relies on _parse_and_chunk_documents)
        self.logger.info(f"Starting index build process for vector DB type: {self.vector_db_type}")
         # --- 1. Document Parsing and Chunking ---
        all_chunks_info = self._parse_and_chunk_documents() # Uses the updated parsing logic
        if not all_chunks_info:
            self.logger.error("No chunks were extracted from the documents. Index building aborted.")
            return None

         # --- 2. Embedding Generation (Batched) ---
        chunk_texts = [info["chunk_text"] for info in all_chunks_info]
        self.logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        all_embeddings = self._batch_get_embeddings(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

         # --- 3. Combine Info and Embeddings ---
        processed_chunks_data = []
        failed_embeddings = 0
        embedding_dim = None
        for i, chunk_info in enumerate(all_chunks_info):
            if i < len(all_embeddings) and all_embeddings[i] is not None and len(all_embeddings[i]) > 0:
                embedding = all_embeddings[i]
                if embedding_dim is None: embedding_dim = len(embedding)
                processed_chunks_data.append({**chunk_info, "embeddings": embedding})
            else:
                failed_embeddings += 1
                self.logger.warning(f"Failed to get embedding for chunk {i} from {chunk_info.get('document_name', 'N/A')}. Skipping.")

        if not processed_chunks_data:
            self.logger.error("No chunks were successfully embedded. Index building failed.")
            return None

        self.logger.info(f"Successfully embedded {len(processed_chunks_data)} chunks.")
        if failed_embeddings > 0:
            self.logger.warning(f"Failed to embed {failed_embeddings} chunks.")

         # --- 4. Populate Vector DB ---
        df = pd.DataFrame(processed_chunks_data)
        # Ensure embeddings are numpy arrays for CSV/InMemory
        if not df.empty:
            df['embeddings'] = df['embeddings'].apply(np.array)

        if self.vector_db_type == "chroma":
            self.logger.info("Populating ChromaDB collection...")
            self._populate_chroma_from_dataframe(df)
            return None # ChromaDB handles its own storage
        else: # CSV or InMemory
            self.logger.info(f"Index data prepared for {self.vector_db_type} storage.")
            return df # Return the DataFrame

    # --- Document Parsing (MODIFIED for DOCX and general separators) ---
    def _parse_document_text(self, doc_path: str) -> Optional[List[Tuple[int, str]]]:
        """
        Extracts text.
        For PDF: Returns list of (page_num, page_text).
        For DOCX: Returns list containing [(1, full_doc_text)].
        """
        doc_name = os.path.basename(doc_path)
        content_list = []
        try:
            if doc_path.lower().endswith(".pdf"):
                # Keep PDF parsing page by page (or consider combining pages if needed)
                with open(doc_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and len(page_text.strip()) > 10:
                                content_list.append((page_num + 1, page_text))
                            else:
                                self.logger.debug(f"Skipping empty/short page {page_num + 1} in PDF {doc_name}")
                        except Exception as page_err:
                            self.logger.error(f"Error extracting text from PDF {doc_name}, page {page_num + 1}: {page_err}")
            elif doc_path.lower().endswith(".docx"):
                # --- MODIFICATION START ---
                # Combine all paragraphs into a single text block for DOCX
                document = docx.Document(doc_path)
                full_text = "\n".join([p.text for p in document.paragraphs if p.text.strip()])
                if full_text and len(full_text.strip()) > 10:
                    # Return the whole document text as item number 1
                    content_list.append((1, full_text))
                else:
                     self.logger.debug(f"Skipping empty/short DOCX {doc_name}")
                # --- MODIFICATION END ---
            else:
                self.logger.warning(f"Unsupported document format: {doc_path}. Skipping.")
                return None
            return content_list
        except FileNotFoundError:
            self.logger.error(f"Document not found during parsing: {doc_path}")
            return None
        except Exception as e:
            self.logger.exception(f"Error processing document {doc_path}: {e}")
            return None


    def _parse_and_chunk_documents(self) -> List[Dict[str, Any]]:
        """Parses PDFs/DOCX and splits them into chunks based on the configured strategy."""
        self.logger.info(f"Parsing documents using pattern: {self.documents_path_pattern}")
        document_files = glob.glob(self.documents_path_pattern)
        # Filter out directory matches if pattern is too broad, like '*'
        document_files = [f for f in document_files if os.path.isfile(f)]

        if not document_files:
            self.logger.error(f"No document files found matching pattern: {self.documents_path_pattern}")
            # It might be okay to proceed if DB already exists, so don't raise FileNotFoundError here.
            # The calling function (_build_index...) will handle the empty chunk list.
            return []
        self.logger.info(f"Found {len(document_files)} potential document files to process.")

        if self.chunk_strategy == "recursive":
             # Adjusted separators for general/scientific text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""], # Common separators
                length_function=len,
                is_separator_regex=False,
            )
        # Add other strategies here if needed
        else:
            self.logger.warning(f"Unknown chunk strategy '{self.chunk_strategy}'. Defaulting to 'recursive'.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
            )

        all_chunks_info = []
        total_chunks_processed = 0

        for doc_path in document_files:
            self.logger.debug(f"Processing document: {doc_path}")
            doc_name = os.path.basename(doc_path)
            # Use the helper to get text (list of tuples: (page/para_num, text))
            doc_content = self._parse_document_text(doc_path)

            if not doc_content:
                self.logger.debug(f"No content extracted from {doc_path}. Skipping.")
                continue

            for item_num, item_text in doc_content: # item_num is page or para num
                # Split item text into chunks
                try:
                    chunks = text_splitter.split_text(item_text)
                    for chunk_num, chunk_text in enumerate(chunks):
                        cleaned_chunk = chunk_text.strip()
                        # Filter out very small or potentially noisy chunks
                        if len(cleaned_chunk) >= 50: # Increased min length for scientific text
                            all_chunks_info.append({
                                "document_name": doc_name,
                                "page_number": item_num, # Represents page (PDF) or paragraph (DOCX)
                                "chunk_number": chunk_num,
                                "chunk_text": cleaned_chunk,
                                # Potential: Add metadata extracted during parsing here
                            })
                            total_chunks_processed += 1
                        else:
                            self.logger.debug(f"Skipping short/noisy chunk {chunk_num} from item {item_num} in {doc_name}")
                except Exception as split_err:
                    self.logger.error(f"Error splitting text for item {item_num} of {doc_name}: {split_err}")

        self.logger.info(f"Finished parsing. Extracted {len(all_chunks_info)} valid chunks from {len(document_files)} documents.")
        return all_chunks_info

    # --- CSV Saving (Unchanged) ---
    def _save_dataframe_to_csv(self, df: pd.DataFrame, file_path: str):
         """Saves the DataFrame (with embeddings as lists) to a CSV file."""
         # (Logic is general, should work fine)
         try:
             self.logger.info(f"Saving vector database DataFrame to {file_path}...")
             if df.empty:
                 self.logger.warning(f"Attempted to save an empty DataFrame to {file_path}. Skipping save.")
                 return

             df_copy = df.copy()
             # Convert numpy arrays to lists for CSV compatibility
             if 'embeddings' in df_copy.columns and isinstance(df_copy['embeddings'].iloc[0], np.ndarray):
                 df_copy['embeddings'] = df_copy['embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])
             elif 'embeddings' in df_copy.columns and isinstance(df_copy['embeddings'].iloc[0], list):
                 pass # Already lists
             elif 'embeddings' in df_copy.columns:
                  # Handle potential mixed types or unexpected types gracefully
                 self.logger.warning(f"Embeddings column has unexpected type {type(df_copy['embeddings'].iloc[0])} during CSV save. Attempting list conversion.")
                 df_copy['embeddings'] = df_copy['embeddings'].apply(lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else [])


             # Ensure directory exists before saving
             dir_name = os.path.dirname(file_path)
             if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

             df_copy.to_csv(file_path, index=False)
             self.logger.info(f"DataFrame successfully saved to {file_path}")
         except Exception as e:
             self.logger.exception(f"Error saving DataFrame to CSV at {file_path}: {e}")

    # --- Embedding Generation (Unchanged Logic) ---
    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, text: str, task_type="RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """Generates embeddings with retry logic and caching."""
        # (Logic is general)
        cache_key = f"{task_type}::{text}"
        if self.use_embedding_cache and cache_key in self.embedding_cache:
             self.logger.debug(f"Embedding cache HIT for task '{task_type}'")
             return self.embedding_cache[cache_key]

        # Avoid embedding empty strings
        if not text or not text.strip():
            self.logger.warning(f"Attempted to embed empty text for task '{task_type}'. Returning None.")
            return None

        self.logger.debug(f"Getting embedding for task '{task_type}' (text length: {len(text)})...")
        try:
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type,
            )
            embedding = response.get('embedding') # Use .get for safety
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                if self.use_embedding_cache:
                    self.embedding_cache[cache_key] = embedding
                return embedding
            else:
                self.logger.error(f"Embedding API returned invalid/empty embedding for task '{task_type}'. Response: {response}")
                return None
        except Exception as e:
            error_str = str(e).lower()
            if "resource has been exhausted" in error_str or "quota" in error_str or "429" in error_str:
                self.logger.warning(f"Rate limit or quota error during embedding: {e}. Retrying...")
                raise # Re-raise to trigger tenacity retry
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                self.logger.error(f"Authentication/Permission Error during embedding: {e}. NOT RETRYING.")
                return None # Don't retry auth errors
            elif "500" in error_str or "internal server error" in error_str:
                self.logger.warning(f"Server error during embedding: {e}. Retrying...")
                raise # Retry server errors
            elif "invalid content" in error_str or "400" in error_str:
                 # Log the problematic text snippet safely
                 log_snippet = text[:100].encode('unicode_escape').decode('utf-8')
                 self.logger.error(f"Invalid content error during embedding: {e}. Text snippet (escaped): '{log_snippet}...'. NOT RETRYING.")
                 return None
            else:
                self.logger.error(f"Non-retryable or unexpected error generating embeddings: {e}")
                return None # Return None for other errors after retries fail

    def _batch_get_embeddings(self, text_list: List[str], batch_size=10, task_type="RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """
        Process embeddings for a list of texts. Iterates and calls embedding function.
        """
        # (Logic is general)
        self.logger.info(f"Getting embeddings for {len(text_list)} texts (sequential calls, effective batch size ~1)")
        all_embeddings: List[Optional[List[float]]] = []
        rate_limit_delay = 0.5 # Adjust delay as needed (Google API often allows higher QPS)

        for i, text in enumerate(text_list):
            self.logger.debug(f"Processing embedding for item {i+1}/{len(text_list)}")
            emb = None
            try:
                emb = self._get_embeddings_with_retry(text, task_type=task_type)
                if emb is None:
                    self.logger.warning(f"Embedding failed for item {i+1} after retries.")
            except Exception as e: # Catch final retry failure from tenacity
                self.logger.error(f"Embedding failed permanently for item {i+1} after retries: {e}")
                 # emb remains None

            all_embeddings.append(emb)

            # Rate limiting: Sleep maybe only every N calls or if rate limit errors occur
            if (i + 1) % batch_size == 0: # Sleep every `batch_size` calls
                try:
                    time.sleep(rate_limit_delay)
                except KeyboardInterrupt:
                    self.logger.warning("Embedding process interrupted.")
                    break

            # Log progress periodically
            if (i + 1) % 100 == 0 or (i + 1) == len(text_list): # Log every 100
                successful_count = sum(1 for e in all_embeddings if e is not None)
                self.logger.info(f"Processed {i + 1}/{len(text_list)} embeddings ({successful_count} successful).")

        return all_embeddings


    # --- Retrieval (Unchanged Logic, DB specific methods handle data) ---
    def _get_relevant_chunks_csv(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from the CSV/InMemory DataFrame."""
        # (Logic is general)
        if self.vector_db_df is None or self.vector_db_df.empty or 'embeddings' not in self.vector_db_df.columns:
            self.logger.warning("CSV/InMemory vector database is not available or lacks embeddings.")
            return []

         # Ensure query embedding is a 2D numpy array
        query_embedding_2d = np.array([query_embedding])

         # Check if embeddings column contains valid numpy arrays and filter invalid ones
        is_valid_embedding = self.vector_db_df['embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
        valid_df = self.vector_db_df[is_valid_embedding]

        if valid_df.empty:
             self.logger.error("No valid numpy array embeddings found in DataFrame. Cannot perform similarity search.")
             return []

         # Stack database embeddings
        try:
            database_embeddings = np.stack(valid_df["embeddings"].to_numpy())
            if len(database_embeddings) == 0:
                self.logger.warning("Database embeddings stack is empty after filtering.")
                return []
             # Dimension check
            if query_embedding_2d.shape[1] != database_embeddings.shape[1]:
                self.logger.error(f"Embedding dimension mismatch: Query ({query_embedding_2d.shape[1]}) vs DB ({database_embeddings.shape[1]})")
                return []
        except Exception as stack_err:
            self.logger.error(f"Error stacking database embeddings: {stack_err}. Is the embedding data valid?")
            # Log a sample embedding for debugging
            if len(valid_df) > 0:
                 self.logger.debug(f"Sample embedding type: {type(valid_df['embeddings'].iloc[0])}, shape: {valid_df['embeddings'].iloc[0].shape}")
            return []

         # Calculate cosine similarities
        try:
            similarities = cosine_similarity(query_embedding_2d, database_embeddings)[0]
        except Exception as sim_err:
            self.logger.error(f"Error calculating cosine similarity: {sim_err}")
            return []

         # Get top_k indices
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k <= 0: return []

        # Get indices of top_k largest similarities
        top_indices_local = np.argsort(similarities)[-actual_top_k:][::-1] # Indices within the filtered 'database_embeddings'
        top_original_indices = valid_df.index[top_indices_local] # Map back to original DataFrame index

         # Prepare results
        results = []
        relevant_rows = self.vector_db_df.loc[top_original_indices] # Select using original indices
        for i, (original_idx, row) in enumerate(relevant_rows.iterrows()):
            similarity_score = similarities[top_indices_local[i]] # Get score using local index
            results.append({
                 "id": f"chunk_{original_idx}", # Use original index as ID for CSV
                 "text": row['chunk_text'],
                 "metadata": {
                     "document_name": row.get('document_name', 'Unknown'),
                     "page_number": int(row.get('page_number', -1)),
                     "chunk_number": int(row.get('chunk_number', -1)),
                 },
                 "score": float(similarity_score) # Ensure score is float
             })

        return results

    def _get_relevant_chunks_chroma(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
         """Retrieves relevant chunks from ChromaDB."""
         # (Logic is general)
         if not hasattr(self, 'collection') or self.collection is None:
             self.logger.error("ChromaDB collection is not initialized.")
             return []
         try:
             # Query ChromaDB collection
             results = self.collection.query(
                 query_embeddings=[query_embedding],
                 n_results=top_k,
                 include=["documents", "metadatas", "distances"] # Request distances
             )

             if not results or not results.get("ids") or not results["ids"][0]:
                 self.logger.debug("No relevant chunks found in ChromaDB for the query.")
                 return []

             # Format results consistently
             formatted_results = []
             for i, chunk_id in enumerate(results["ids"][0]):
                # Ensure all lists have the same length before accessing index i
                if i < len(results.get("distances", [[]])[0]) and \
                   i < len(results.get("documents", [[]])[0]) and \
                   i < len(results.get("metadatas", [[]])[0]):

                     distance = results["distances"][0][i]
                     # Convert distance to similarity score (assuming cosine distance)
                     similarity_score = 1.0 - distance # Adjust if Chroma uses a different metric

                     formatted_results.append({
                         "id": chunk_id,
                         "text": results["documents"][0][i],
                         "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                         "score": float(similarity_score) # Ensure score is float
                     })
                else:
                    self.logger.warning(f"Inconsistent result lengths from ChromaDB for query. Skipping partial result at index {i}.")


             return formatted_results

         except Exception as e:
             self.logger.exception(f"Error querying relevant chunks from ChromaDB: {e}")
             return []

    def _get_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
         """
         Retrieves relevant text chunks based on the query.
         """
         # (Logic is general)
         self.logger.info(f"Retrieving top {top_k} relevant chunks for query: '{query[:100]}...'")
         query_embedding = self._get_embeddings_with_retry(query, task_type="RETRIEVAL_QUERY")

         if query_embedding is None:
             self.logger.error("Query embedding failed. Cannot retrieve chunks.")
             return []

         results = []
         try:
             if self.vector_db_type == "chroma":
                 results = self._get_relevant_chunks_chroma(query_embedding, top_k)
             elif self.vector_db_type in ["csv", "inmemory"]:
                 results = self._get_relevant_chunks_csv(query_embedding, top_k)
             else:
                 self.logger.error(f"Unsupported vector_db_type '{self.vector_db_type}' in _get_relevant_chunks.")
                 return []

             if results:
                 self.logger.info(f"Retrieved {len(results)} chunks. Top score: {results[0]['score']:.4f}")
             else:
                 self.logger.info("No relevant chunks retrieved.")

             return results

         except Exception as e:
             self.logger.exception(f"An unexpected error occurred in _get_relevant_chunks: {e}")
             return []


    # --- Context Formatting (Unchanged) ---
    def _format_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a string for the LLM prompt."""
        # (Logic is general)
        if not chunks:
             return "No relevant context found."

        context_parts = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0.0)
            # Use page_number as primary locator info
            source_info = f"Source: Doc='{metadata.get('document_name', 'N/A')}', Loc={metadata.get('page_number', 'N/A')}, Score={score:.4f}"
            context_parts.append(f"[{source_info}]\n{chunk.get('text', '')}")

        return "\n\n---\n\n".join(context_parts)


    # --- LLM Response Generation (Unchanged Logic) ---
    def _generate_llm_response(self, prompt: str) -> str:
        """Helper to call the generative model with caching and error handling."""
        # (Logic is general)
        if self.use_llm_cache and prompt in self.llm_cache:
             self.logger.debug("LLM cache HIT")
             return self.llm_cache[prompt]

        self.logger.debug(f"Generating LLM response for prompt (length: {len(prompt)}): '{prompt[:200]}...'")
        start_time = time.time()

        try:
            # Use the synchronous version
            response = self.model.generate_content(prompt)

             # Refined error/block checking for synchronous response
            if not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = "Unknown"
                safety_ratings = []
                if feedback:
                     block_reason = getattr(feedback, 'block_reason', 'Unknown')
                     safety_ratings = getattr(feedback, 'safety_ratings', [])
                self.logger.warning(f"LLM Response was blocked or had no candidates. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
                error_message = f"LLM_ERROR: Response blocked (Reason: {block_reason})."
                if self.use_llm_cache: self.llm_cache[prompt] = error_message
                return error_message

             # Check if text attribute exists and is not empty
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                 generated_text = response.text
            else:
                # Fallback: Try to extract text from parts
                try:
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                         if not generated_text:
                             self.logger.warning("LLM Response generated but text content is empty even in parts.")
                             generated_text = "LLM_ERROR: Response content was empty."
                         # else:
                         #    self.logger.debug("LLM Response text extracted from parts.")
                     else:
                         self.logger.warning("LLM Response generated but text attribute and parts are missing/empty.")
                         generated_text = "LLM_ERROR: Response structure unexpected or empty."
                except (IndexError, AttributeError, Exception) as e:
                     self.logger.error(f"Error accessing response parts: {e}")
                     generated_text = "LLM_ERROR: Failed to access response content."

            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM generation took {elapsed_time:.2f}s. Response length: {len(generated_text)}")

            if self.use_llm_cache:
                 self.llm_cache[prompt] = generated_text

            return generated_text

        except Exception as e:
            self.logger.exception(f"Exception during LLM generation call: {e}")
            error_str = str(e).lower()
            # Check for specific API errors
            if "resource_exhausted" in error_str or "quota" in error_str or "429" in error_str:
                return "LLM_ERROR: API rate limit or quota exceeded."
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                return "LLM_ERROR: Invalid API Key or insufficient permissions."
            # Handle other potential exceptions from the API client
            return f"LLM_ERROR: An unexpected error occurred during text generation: {str(e)}"


    # --- Query Processing Steps ---

    # --- Query Transformation (MODIFIED - Placeholder/Removed Specific Logic) ---
    def _transform_query(self, query: str) -> str:
        """
        Transforms the query. Placeholder for CNT-specific transformations.
        Currently returns the query as is.
        """
        self.logger.debug(f"Applying query transformations (currently basic pass-through) to: '{query}'")
        transformed_query = query
        # --- Add CNT-specific transformations here if needed ---
        # Example: Normalize units (e.g., "10 nm" -> "10nm")
        # Example: Expand acronyms (e.g., "SWCNT" -> "Single-Walled Carbon Nanotube")
        # Example: Rephrase based on common scientific query patterns

        # Simple example: Lowercase
        transformed_query = transformed_query.lower()

        if transformed_query != query:
            self.logger.info(f"Query transformed: '{query}' -> '{transformed_query}'")
        return transformed_query

    # --- Query Expansion (MODIFIED prompt) ---
    def _expand_query(self, query: str, num_expansions: int = 2) -> List[str]:
        """Generate query variations using an LLM to improve retrieval recall for CNT research."""
        if num_expansions <= 0:
            return [query]

        self.logger.info(f"Expanding query '{query[:100]}...' into {num_expansions} variations for CNT context.")

        expansion_prompt = f"""You are an expert search query generator specializing in scientific literature about Carbon Nanotubes (CNTs).
        Generate {num_expansions} alternative search queries for the following original question.
        The goal is to capture the same information need using different phrasing, keywords, and potential concepts found in CNT research papers.

        Original Question:
        "{query}"

        Instructions:
        1.  Understand the core scientific intent of the original question.
        2.  Create {num_expansions} distinct alternative queries.
        3.  Use relevant scientific terms (e.g., 'CVD synthesis', 'catalyst', 'chirality', 'Raman spectroscopy', 'SWCNT', 'MWCNT', 'functionalization', specific chemical formulas if relevant).
        4.  Focus on experimental parameters, material properties, characterization techniques, or applications mentioned in CNT research.
        5.  Return ONLY the alternative queries, one per line. Do not include numbering, explanations, or the original query.

        Alternative Queries:"""

        response_text = self._generate_llm_response(expansion_prompt)

        if response_text.startswith("LLM_ERROR"):
            self.logger.warning(f"Query expansion failed due to LLM error: {response_text}")
            return [query] # Fallback to original query

        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]

        if len(expanded_queries) > num_expansions:
            self.logger.debug(f"LLM generated more expansions than requested ({len(expanded_queries)}), taking first {num_expansions}.")
            expanded_queries = expanded_queries[:num_expansions]
        elif len(expanded_queries) < num_expansions:
             self.logger.debug(f"LLM generated fewer expansions than requested ({len(expanded_queries)}).")

        # Always include the original query transformed (which is just lowercased now)
        transformed_original = self._transform_query(query)
        all_queries = [transformed_original] + expanded_queries

        # Deduplicate
        unique_queries_set = set()
        unique_queries_list = []
        for q in all_queries:
             if q not in unique_queries_set:
                  unique_queries_set.add(q)
                  unique_queries_list.append(q)


        self.logger.info(f"Generated {len(unique_queries_list)} unique queries for retrieval: {unique_queries_list}")
        return unique_queries_list

    # --- Reasoning (MODIFIED prompt context) ---
    def _reason_and_refine_query(self, original_query: str, accumulated_context: str, history: List[str], current_hop: int) -> Tuple[str, str]:
        """
        Uses the LLM to analyze context and decide the next step for multihop in a CNT context.
        """
        self.logger.info(f"--- Hop {current_hop} Reasoning ---")
        context_snippet = accumulated_context.strip()[:500]
        self.logger.debug(f"Reasoning based on History: {' -> '.join(history)}")
        self.logger.debug(f"Context Snippet Provided for Reasoning:\n{context_snippet if context_snippet else 'None'}...\n")

        prompt = f"""You are a precise reasoning engine for a RAG system answering questions about Carbon Nanotube (CNT) research papers and data.
Your SOLE task is to determine the *next action* based on the Original Question and the Accumulated Context retrieved so far.

Original Question:
"{original_query}"

Accumulated Context (from previous search steps, potentially including experimental results, paper excerpts, etc.):
--- START CONTEXT ---
{accumulated_context if accumulated_context.strip() else "No context retrieved yet."}
--- END CONTEXT ---

Reasoning History (Previous search queries used):
--- START HISTORY ---
{' -> '.join(history) if history else "This is the first reasoning step."}
--- END HISTORY ---

Instructions:
1. Analyze the Original Question and the Accumulated Context. Focus on whether the context *directly and completely* answers the Original Question (e.g., provides the specific parameter, explains the phenomenon, describes the result).
2. Consider the search queries used so far (Reasoning History).
3. Decide the single next action:
    a. If the Accumulated Context contains enough information to *fully answer* the Original Question: Output *exactly* this line and nothing else:
        ANSWER_COMPLETE
    b. If the Accumulated Context is insufficient or partially addresses the question, identify the *single most important missing piece of information* or the *next logical sub-question* needed to fully answer the Original Question. Formulate a *concise, specific search query* for this missing piece (e.g., "Raman D/G ratio for experiment #10", "effect of temperature on CNT diameter catalyst Fe 900C", "SWCNT functionalization methods using carboxyl groups"). Output *exactly* in this format (single line, nothing else):
        NEXT_QUERY: [Your concise scientific search query here]
    c. If there was an error in retrieving context or the context is clearly irrelevant, and previous attempts haven't helped, output:
         ANSWER_COMPLETE

CRITICAL:
- Your entire response MUST be only ONE line.
- It must be EITHER "ANSWER_COMPLETE" OR "NEXT_QUERY: [your query]".
- Do NOT add explanations, introductions, apologies, or any other text.
- Do NOT answer the Original Question yourself. Only decide the next action.

Decision:"""

        response_text = self._generate_llm_response(prompt)

        # --- Parsing Logic (Unchanged) ---
        action = "ERROR"
        value = f"LLM response parsing failed. Raw response: '{response_text}'"

        if response_text.startswith("LLM_ERROR"):
             self.logger.error(f"Reasoning step failed due to LLM error: {response_text}")
             value = response_text
             action = "ANSWER_COMPLETE" # Default to complete on error
             self.logger.warning("LLM error during reasoning. Assuming ANSWER_COMPLETE to proceed.")
        else:
            response_clean = response_text.strip()
            answer_complete_pattern = re.compile(r"^ANSWER_COMPLETE$", re.IGNORECASE)
            next_query_pattern = re.compile(r"^NEXT_QUERY:\s*(.+)$", re.IGNORECASE | re.DOTALL) # Allow multiline queries just in case

            match_complete = answer_complete_pattern.match(response_clean)
            match_next = next_query_pattern.match(response_clean)

            if match_complete:
                action = "ANSWER_COMPLETE"
                value = ""
                self.logger.info(">>> Reasoning Decision: ANSWER_COMPLETE")
            elif match_next:
                new_query = match_next.group(1).strip()
                if new_query:
                    action = "NEXT_QUERY"
                    value = new_query
                    self.logger.info(f">>> Reasoning Decision: NEXT_QUERY -> '{new_query}'")
                else:
                    action = "ANSWER_COMPLETE"
                    value = ""
                    self.logger.warning("LLM generated NEXT_QUERY but the query was empty. Treating as ANSWER_COMPLETE.")
            else:
                action = "ANSWER_COMPLETE" # Default to complete on format failure
                value = ""
                self.logger.warning(f"Unexpected reasoning response format: '{response_clean}'. Defaulting to ANSWER_COMPLETE.")

        return action, value

    # --- Context Management (Unchanged Logic, prompt updated) ---
    def _manage_context(self, accumulated_context_list: List[str], new_chunk_list: List[Dict[str, Any]],
                        max_tokens: int = 20000) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Manages the accumulated context list (deduplication, size control via summarization).
        """
        # (Deduplication logic is general)
        if not new_chunk_list:
            self.logger.debug("No new chunks to manage.")
            return accumulated_context_list, []

        # --- Deduplication ---
        unique_new_chunks_by_text: Dict[str, Dict[str, Any]] = {}
        for chunk in new_chunk_list:
            text = chunk.get('text', '')
            if not text: continue # Skip chunks with no text
            if text not in unique_new_chunks_by_text:
                 unique_new_chunks_by_text[text] = chunk
            else:
                 if chunk.get('score', 0) > unique_new_chunks_by_text[text].get('score', 0):
                     unique_new_chunks_by_text[text] = chunk
        deduped_new_chunks = list(unique_new_chunks_by_text.values())
        if len(deduped_new_chunks) < len(new_chunk_list):
            self.logger.debug(f"Removed {len(new_chunk_list) - len(deduped_new_chunks)} duplicate chunks from current hop.")

        # --- Deduplicate against existing context ---
        existing_texts = set()
        for ctx_str in accumulated_context_list:
             # Extract text content, handling potential formatting like "[Source: ...] Text" or "[SUMMARIZED...]:\nText"
             text_content = ctx_str
             match_source = re.search(r'\]\s*(.*)', ctx_str, re.DOTALL | re.IGNORECASE)
             match_summary = re.search(r'\]:\s*\n?(.*)', ctx_str, re.DOTALL | re.IGNORECASE)
             if match_summary:
                 text_content = match_summary.group(1)
             elif match_source:
                 text_content = match_source.group(1)
             existing_texts.add(text_content.strip())

        truly_new_chunks = []
        similarity_threshold = 0.9 # Threshold for considering chunks 'similar'
        for chunk in deduped_new_chunks:
            text = chunk.get('text', '')
            is_similar = False
            for existing_text in existing_texts:
                 # Optimization: Quick check for exact match first
                 if text == existing_text:
                     is_similar = True
                     self.logger.debug("Found new chunk identical to existing context. Skipping.")
                     break
                 # If not identical, check similarity
                 similarity = SequenceMatcher(None, text, existing_text).ratio()
                 if similarity > similarity_threshold:
                     is_similar = True
                     self.logger.debug(f"Found new chunk similar to existing context (Score: {similarity:.2f}). Skipping.")
                     break
            if not is_similar:
                 truly_new_chunks.append(chunk)
                 existing_texts.add(text) # Add its text to avoid adding another similar one later

        if len(truly_new_chunks) < len(deduped_new_chunks):
            self.logger.info(f"Removed {len(deduped_new_chunks) - len(truly_new_chunks)} chunks similar to existing context.")

        if not truly_new_chunks:
            self.logger.info("No truly new, unique context chunks found in this hop.")
            return accumulated_context_list, []

        # Format the truly new chunks into a single context string for this hop
        new_context_string = self._format_context_from_chunks(truly_new_chunks)
        current_context_list = accumulated_context_list + [new_context_string]

        # --- Size Management (Summarization) ---
        # Rough token count estimation (adjust multiplier if needed for scientific text)
        char_to_token_ratio = 3.5
        total_chars = sum(len(ctx) for ctx in current_context_list)
        estimated_tokens = total_chars / char_to_token_ratio

        self.logger.debug(f"Context size before potential summarization: ~{estimated_tokens:.0f} tokens")

        while estimated_tokens > max_tokens and len(current_context_list) > 1:
             self.logger.info(f"Context too large ({estimated_tokens:.0f} > {max_tokens} tokens), summarizing oldest content...")
             oldest_context = current_context_list.pop(0) # Remove oldest

             if oldest_context.strip().startswith("[SUMMARIZED CONTEXT]:"):
                 self.logger.warning("Oldest context is already summarized. Cannot summarize further. Keeping it.")
                 current_context_list.insert(0, oldest_context)
                 break # Avoid infinite loop

             # --- Updated Summarization Prompt ---
             summary_prompt = f"""Summarize the following context section from scientific literature or experimental data regarding Carbon Nanotubes (CNTs).
             Preserve all key scientific findings, experimental parameters (temperature, pressure, catalyst, etc.), numerical results, material properties, characterization data (e.g., Raman shifts, diameters), and specific CNT types (SWCNT, MWCNT, chirality) mentioned.
             Be concise but retain the essential scientific information needed to answer questions based on this section.

             Context Section to Summarize:
             ---
             {oldest_context}
             ---

             Concise Scientific Summary:"""

             summarized = self._generate_llm_response(summary_prompt)

             if summarized.startswith("LLM_ERROR"):
                 self.logger.error(f"Failed to summarize context: {summarized}. Keeping original.")
                 current_context_list.insert(0, oldest_context)
                 break # Avoid retrying summarization if LLM is failing
             else:
                 # Add the summary to the beginning of the list
                 current_context_list.insert(0, f"[SUMMARIZED CONTEXT]:\n{summarized}")
                 self.logger.info("Summarization complete. Added summarized context.")

             # Recalculate token count
             total_chars = sum(len(ctx) for ctx in current_context_list)
             estimated_tokens = total_chars / char_to_token_ratio
             self.logger.info(f"Context size after summarization: ~{estimated_tokens:.0f} tokens")

        if estimated_tokens > max_tokens:
            self.logger.warning(f"Context size (~{estimated_tokens:.0f}) still exceeds max_tokens ({max_tokens}) even after attempting summarization.")

        return current_context_list, truly_new_chunks


    # --- Final Answer Generation (MODIFIED prompt context) ---
    def _generate_final_answer(self, original_query: str, accumulated_context_list: List[str]) -> str:
        """
        Generates the final answer using the LLM, prompted for synthesizing scientific context.
        """
        self.logger.info("Generating final answer from accumulated scientific context...")
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_context_str or final_context_str.strip() == "No relevant context found.":
            self.logger.warning("No context available for final answer generation.")
            # Provide a more informative message for scientific context
            return "Based on the retrieved scientific literature and data, I could not find the specific information required to answer your question."

        # --- Updated Final Answer Prompt ---
        prompt = f"""You are an expert scientific assistant knowledgeable about Carbon Nanotubes (CNTs).
Your task is to synthesize a final, comprehensive answer to the user's *Original Question* using *only* the information found within the *Accumulated Context* provided below. The context comes from potentially multiple search steps over scientific papers/data and may include summaries.

Instructions:
1.  Carefully read the *Original Question*.
2.  Carefully read *all sections* within the Accumulated Context, noting experimental details, results, conclusions, and any summaries indicated by "[SUMMARIZED CONTEXT]:".
3.  Synthesize a single, coherent answer that directly and fully addresses the *Original Question* with scientific accuracy. Combine relevant findings, data points, and explanations.
4.  Base your answer *strictly* on the text present in the Accumulated Context. Do *not* add external information, interpretations, or scientific knowledge not explicitly stated in the context. Do not make assumptions or extrapolate beyond the provided data.
5.  If the Accumulated Context *definitively lacks* the specific data, parameters, or explanations needed to fully answer the Original Question (after reviewing all sections), state clearly: "Based on the provided context, the specific information regarding [missing aspect, e.g., 'the effect of pressure on chirality'] could not be found." Do not apologize excessively.
6.  Structure the answer logically, using clear scientific language. Present data or findings as accurately as possible based on the context. Start directly with the answer.
7.  Do NOT refer to the search process (e.g., "Based on the retrieved context..."). Just provide the synthesized scientific answer based *on* the context.
8.  Do NOT include the source markers (like `[Source: ...]`) in your final answer.

Original Question:
"{original_query}"

Accumulated Context:
--- START CONTEXT ---
{final_context_str}
--- END CONTEXT ---

Final Synthesized Answer (provide a direct scientific answer based ONLY on the context):"""

        final_response = self._generate_llm_response(prompt)

        if final_response.startswith("LLM_ERROR"):
             self.logger.error(f"Final answer generation failed due to LLM error: {final_response}")
             return f"I encountered an error while trying to generate the final answer ({final_response}). Please check the logs or try again."
        else:
             self.logger.info(f"Final answer generated (length: {len(final_response)}). Snippet: {final_response[:200]}...")
             # Basic post-processing: remove potential leftover instruction phrases if the LLM included them
             final_response = re.sub(r"^\s*Final Synthesized Answer:?\s*", "", final_response, flags=re.IGNORECASE).strip()
             return final_response


    # --- Evaluation and Feedback (MODIFIED prompt contexts) ---

    def evaluate_response(self, question: str, answer: str, retrieved_context_str: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer using LLM-based metrics in a scientific context.
        """
        self.logger.info("Evaluating generated scientific response...")
        metrics = {}

        # --- Relevance Prompt Updated ---
        try:
            relevance_prompt = f"""Rate the relevance of the Answer to the scientific Question on a scale of 1 (Not relevant) to 5 (Highly relevant and directly addresses the scientific query).

            Question: "{question}"
            Answer: "{answer}"

            Output format (JSON): {{"relevance_rating": [1-5], "relevance_explanation": "[Brief explanation of relevance to the scientific query]"}}
            """
            relevance_response = self._generate_llm_response(relevance_prompt)
            # Use more robust JSON parsing if possible, fallback to regex
            try:
                 import json
                 parsed_json = json.loads(relevance_response)
                 metrics['relevance_rating'] = int(parsed_json.get('relevance_rating'))
                 metrics['relevance_explanation'] = str(parsed_json.get('relevance_explanation', ''))
            except (json.JSONDecodeError, TypeError, ValueError):
                 self.logger.warning(f"Could not parse JSON relevance rating, trying regex. Response: {relevance_response}")
                 match = re.search(r'{\s*"relevance_rating"\s*:\s*(\d+)\s*,\s*"relevance_explanation"\s*:\s*"([^"]*)"\s*}', relevance_response, re.IGNORECASE | re.DOTALL)
                 if match:
                     metrics['relevance_rating'] = int(match.group(1))
                     metrics['relevance_explanation'] = match.group(2).strip()
                 else:
                     self.logger.warning(f"Could not parse relevance rating from response: {relevance_response}")
                     metrics['relevance_rating'] = None
                     metrics['relevance_explanation'] = f"Parsing Failed: {relevance_response}"
        except Exception as e:
            self.logger.error(f"Error during relevance evaluation: {e}")
            metrics['relevance_rating'] = None
            metrics['relevance_explanation'] = f"Error: {e}"

        # --- Faithfulness Prompt Updated ---
        try:
            faithfulness_prompt = f"""Rate how faithful the scientific Answer is to the provided Context (excerpts from papers/data) on a scale of 1 (Contradicts context or introduces external info) to 5 (Fully supported by context).
            The answer should only contain information, data, and conclusions present in the context.

            Context:
            ---
            {retrieved_context_str[:6000]}...
            ---
            Answer: "{answer}"

            Output format (JSON): {{"faithfulness_rating": [1-5], "faithfulness_explanation": "[Brief explanation highlighting support or contradiction/external info]"}}
            """
            faithfulness_response = self._generate_llm_response(faithfulness_prompt)
            # Use more robust JSON parsing if possible, fallback to regex
            try:
                 import json
                 parsed_json = json.loads(faithfulness_response)
                 metrics['faithfulness_rating'] = int(parsed_json.get('faithfulness_rating'))
                 metrics['faithfulness_explanation'] = str(parsed_json.get('faithfulness_explanation', ''))
            except (json.JSONDecodeError, TypeError, ValueError):
                 self.logger.warning(f"Could not parse JSON faithfulness rating, trying regex. Response: {faithfulness_response}")
                 match = re.search(r'{\s*"faithfulness_rating"\s*:\s*(\d+)\s*,\s*"faithfulness_explanation"\s*:\s*"([^"]*)"\s*}', faithfulness_response, re.IGNORECASE | re.DOTALL)
                 if match:
                     metrics['faithfulness_rating'] = int(match.group(1))
                     metrics['faithfulness_explanation'] = match.group(2).strip()
                 else:
                      self.logger.warning(f"Could not parse faithfulness rating from response: {faithfulness_response}")
                      metrics['faithfulness_rating'] = None
                      metrics['faithfulness_explanation'] = f"Parsing Failed: {faithfulness_response}"
        except Exception as e:
            self.logger.error(f"Error during faithfulness evaluation: {e}")
            metrics['faithfulness_rating'] = None
            metrics['faithfulness_explanation'] = f"Error: {e}"

        # --- Ground Truth Comparison (Unchanged Logic, prompt updated slightly) ---
        if ground_truth:
            try:
                gt_prompt = f"""Compare the Generated Answer to the Ground Truth Answer for the given scientific Question. Rate the similarity/correctness on a scale of 1 (Very different/Incorrect) to 5 (Highly similar/Correct).

                Question: "{question}"
                Ground Truth Answer: "{ground_truth}"
                Generated Answer: "{answer}"

                Output format (JSON): {{"ground_truth_similarity_rating": [1-5], "ground_truth_similarity_explanation": "[Brief explanation of similarity/differences in scientific content]"}}
                """
                gt_response = self._generate_llm_response(gt_prompt)
                # Use more robust JSON parsing if possible, fallback to regex
                try:
                    import json
                    parsed_json = json.loads(gt_response)
                    metrics['ground_truth_similarity_rating'] = int(parsed_json.get('ground_truth_similarity_rating'))
                    metrics['ground_truth_similarity_explanation'] = str(parsed_json.get('ground_truth_similarity_explanation', ''))
                except (json.JSONDecodeError, TypeError, ValueError):
                    self.logger.warning(f"Could not parse JSON ground truth rating, trying regex. Response: {gt_response}")
                    match = re.search(r'{\s*"ground_truth_similarity_rating"\s*:\s*(\d+)\s*,\s*"ground_truth_similarity_explanation"\s*:\s*"([^"]*)"\s*}', gt_response, re.IGNORECASE | re.DOTALL)
                    if match:
                         metrics['ground_truth_similarity_rating'] = int(match.group(1))
                         metrics['ground_truth_similarity_explanation'] = match.group(2).strip()
                    else:
                         self.logger.warning(f"Could not parse ground truth similarity rating from response: {gt_response}")
                         metrics['ground_truth_similarity_rating'] = None
                         metrics['ground_truth_similarity_explanation'] = f"Parsing Failed: {gt_response}"
            except Exception as e:
                self.logger.error(f"Error during ground truth comparison: {e}")
                metrics['ground_truth_similarity_rating'] = None
                metrics['ground_truth_similarity_explanation'] = f"Error: {e}"

        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def _assess_answer_confidence(self, answer: str, context: str, query: str) -> Optional[float]:
        """Estimate confidence (0.0 to 1.0) in the answer based on scientific context support."""
        # (Logic Unchanged, prompt updated)
        self.logger.debug("Assessing answer confidence...")
        prompt = f"""Assess how well the scientific Answer is supported *solely* by the provided Context (research excerpts/data) in relation to the Query.
        Provide a confidence score between 0.0 (no support/contradicted) and 1.0 (fully and explicitly supported).
        Consider only the information, data, and findings explicitly present in the Context.

        Query: {query}
        Context:
        ---
        {context[:6000]}...
        ---
        Answer: {answer}

        Confidence Score (Return ONLY the numerical score between 0.0 and 1.0, e.g., 0.85):"""

        response = self._generate_llm_response(prompt)

        if response.startswith("LLM_ERROR"):
            self.logger.warning(f"Confidence assessment failed due to LLM error: {response}")
            return None

        try:
            # Extract float, handling potential surrounding text or integer responses
            match = re.search(r"(\d(?:[.,]\d+)?)", response) # Match integer or float
            if match:
                confidence_str = match.group(1).replace(',', '.') # Ensure decimal point
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence)) # Clamp between 0 and 1
                self.logger.debug(f"Assessed Confidence Score: {confidence:.2f}")
                return confidence
            else:
                self.logger.warning(f"Could not parse confidence score from response: '{response}'. Returning moderate confidence.")
                return 0.5 # Default moderate confidence
        except ValueError:
            self.logger.warning(f"Could not convert confidence score response to float: '{response}'. Returning moderate confidence.")
            return 0.5
        except Exception as e:
            self.logger.error(f"Error parsing confidence score: {e}")
            return None


    # --- Feedback Loading/Saving/Recording (Unchanged Logic) ---
    def _load_feedback_history(self):
        """Load feedback history from CSV if it exists."""
        # (Logic is general)
        if os.path.exists(self.feedback_db_path):
             try:
                 feedback_df = pd.read_csv(self.feedback_db_path)
                 # Convert columns back if necessary (e.g., eval metrics if stored as strings)
                 # Example: Convert string dicts back to dicts
                 # for col in ['evaluation_metrics']: # Add other dict columns if needed
                 #    if col in feedback_df.columns:
                 #         feedback_df[col] = feedback_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                 self.feedback_history = feedback_df.to_dict('records')
                 self.logger.info(f"Loaded {len(self.feedback_history)} feedback records from {self.feedback_db_path}.")
             except Exception as e:
                 self.logger.warning(f"Could not load feedback history from {self.feedback_db_path}: {e}")
                 self.feedback_history = []
        else:
             self.logger.info(f"Feedback history file not found at {self.feedback_db_path}. Starting fresh.")
             self.feedback_history = []

    def _save_feedback_history(self):
        """Save the current feedback history to CSV."""
        # (Logic is general)
        if not self.feedback_history:
            self.logger.debug("No feedback history to save.")
            return

        try:
             # Ensure directory exists before saving
             dir_name = os.path.dirname(self.feedback_db_path)
             if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

             feedback_df = pd.DataFrame(self.feedback_history)
             # Convert dicts/lists to strings for CSV storage if needed
             # for col in ['evaluation_metrics', 'reasoning_trace', 'queries_used']: # Add relevant list/dict cols
             #     if col in feedback_df.columns:
             #         feedback_df[col] = feedback_df[col].astype(str)

             feedback_df.to_csv(self.feedback_db_path, index=False)
             self.logger.info(f"Saved {len(self.feedback_history)} feedback records to {self.feedback_db_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save feedback history to {self.feedback_db_path}: {e}")

    def record_feedback(self, query: str, answer: str, hop_count: int,
                         final_context: str, reasoning_trace: List[str], search_query_history: List[str], # Added query history
                         user_rating: Optional[int] = None,
                         user_comment: Optional[str] = None,
                         evaluation_metrics: Optional[Dict[str, Any]] = None):
        """Record user feedback and internal metrics for a query-answer interaction."""
        # (Logic is general, added query history)
        feedback_record = {
             "timestamp": pd.Timestamp.now().isoformat(),
             "query": query,
             "answer": answer,
             "hop_count": hop_count,
             "final_context_length": len(final_context),
             "reasoning_trace": " -> ".join(reasoning_trace), # Store trace as string
             "search_queries": " | ".join(search_query_history), # Store queries used
             "user_rating": user_rating,
             "user_comment": user_comment,
             **(evaluation_metrics or {}) # Add automated metrics
         }
        # Ensure all values are suitable for CSV (e.g., no complex objects without string conversion)
        for key, value in feedback_record.items():
            if isinstance(value, (dict, list)):
                 feedback_record[key] = str(value) # Simple string conversion for CSV


        self.feedback_history.append(feedback_record)
        self.logger.info(f"Recorded feedback for query: '{query[:50]}...' (User Rating: {user_rating})")
        self._save_feedback_history() # Save immediately


    def _analyze_feedback_patterns(self):
        """Placeholder for analyzing feedback."""
        # (Logic is general)
        self.logger.info("Analyzing feedback patterns (Placeholder)...")
        if not self.feedback_history: return

        try:
            feedback_df = pd.DataFrame(self.feedback_history)
            # Example analysis: Count low-rated responses
            if 'user_rating' in feedback_df.columns:
                 low_rated = feedback_df[pd.to_numeric(feedback_df['user_rating'], errors='coerce') <= 2]
                 if not low_rated.empty:
                     self.logger.info(f"Found {len(low_rated)} low-rated (<=2) feedback entries.")
                     # Further analysis could go here... find common queries, keywords in comments etc.
            # Example: Analyze average confidence/evaluation scores
            # Example: Correlate hop count with ratings
        except Exception as e:
            self.logger.error(f"Error during basic feedback analysis: {e}")


    # --- Main Query Processing (Largely Unchanged Flow, uses updated components) ---

    def process_query(self, question: str, top_k: int = 5, max_hops: int = 3,
                      use_query_expansion: bool = False,
                      request_evaluation: bool = True,
                      generate_graph: bool = True,
                      record_user_feedback: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """
        Processes a user query using the enhanced multihop RAG pipeline for CNTs.
        """
        # (Flow is general, relies on the modified methods above)
        process_start_time = time.time()
        self.logger.info(f"\n--- Starting CNT Query Process ---")
        self.logger.info(f"Original Question: '{question}'")
        self.logger.info(f"Config: top_k={top_k}, max_hops={max_hops}, expansion={use_query_expansion}, eval={request_evaluation}, graph={generate_graph}")

        # --- Initialization Check ---
        db_ready = False
        if self.vector_db_type == 'chroma' and hasattr(self, 'collection') and self.collection is not None:
            # Check count only if client is likely initialized
            try:
                 if self.collection.count() >= 0: # count() might return 0 legitimately
                     db_ready = True
            except Exception as chroma_err:
                 self.logger.warning(f"Could not confirm ChromaDB status: {chroma_err}")
        elif self.vector_db_type in ['csv', 'inmemory'] and self.vector_db_df is not None and not self.vector_db_df.empty:
             db_ready = True

        if not db_ready:
             self.logger.critical("Vector database is not initialized or empty. Cannot process query.")
             # Return structure consistent with success case but with error message
             return {
                 "final_answer": "Error: The knowledge base (Vector DB) is not available. Please check configuration and data processing.",
                 "source_info": "Initialization Error",
                 "reasoning_trace": ["Vector DB not ready"],
                 "formatted_reasoning": "Vector DB not ready",
                 "confidence_score": None,
                 "evaluation_metrics": None,
                 "debug_info": {"processing_time_s": time.time() - process_start_time, "hops_taken": 0, "final_context_length": 0, "graph_filename": None}
             }

        original_query = question
        current_query = self._transform_query(original_query) # Apply basic transformations

        accumulated_context_list = []
        reasoning_trace = ["START"]
        search_query_history = [current_query] # Start history with the initial transformed query
        hops_taken = 0
        all_retrieved_chunks_this_session = [] # Keep track of unique chunks retrieved

        # --- Initial Query Expansion (Optional) ---
        # If expansion is used, the first query for the loop might be different
        # or the reasoning step might consider the expanded queries.
        # Simple approach: Expand and use the first one, let reasoning generate subsequent ones.
        if use_query_expansion:
             self.logger.info("Applying query expansion for the initial retrieval...")
             expanded_queries = self._expand_query(current_query, num_expansions=2)
             if expanded_queries:
                 current_query = expanded_queries[0] # Use the first (potentially transformed original) for the first hop
                 search_query_history = expanded_queries # Log all expansions tried initially
                 reasoning_trace.append(f"Initial Expanded Queries: {expanded_queries}")
             # If expansion fails, current_query remains the transformed original

        # --- Multihop Loop ---
        for hop in range(max_hops):
            hops_taken = hop + 1
            self.logger.info(f"\n--- Hop {hops_taken}/{max_hops} ---")
            reasoning_trace.append(f"--- Hop {hops_taken} ---")

            if not current_query:
                 self.logger.warning(f"Hop {hops_taken}: Current query is empty. Breaking loop.")
                 reasoning_trace.append(f"Hop {hops_taken}: Error - Query is empty.")
                 break

            # Only add subsequent queries decided by reasoning to history here
            if hop > 0: # Add queries from hop 2 onwards
                 search_query_history.append(current_query)

            self.logger.info(f"Retrieving context for query: '{current_query}'")
            reasoning_trace.append(f"Hop {hops_taken}: Retrieving with query -> '{current_query}'")
            retrieved_chunks = self._get_relevant_chunks(current_query, top_k=top_k)

            if not retrieved_chunks:
                 self.logger.warning(f"Retrieval returned no relevant chunks for hop {hops_taken}.")
                 reasoning_trace.append(f"Hop {hops_taken}: Retrieval yielded no results.")
                 # Don't necessarily break if retrieval fails mid-way, reasoning might still work with previous context
            else:
                 self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks for hop {hops_taken}.")
                 reasoning_trace.append(f"Hop {hops_taken}: Retrieved {len(retrieved_chunks)} chunk(s). Top score: {retrieved_chunks[0]['score']:.4f}")
                 all_retrieved_chunks_this_session.extend(retrieved_chunks)


            # --- Context Management ---
            accumulated_context_list, added_unique_chunks = self._manage_context(
                 accumulated_context_list,
                 retrieved_chunks,
                 max_tokens=20000
            )
            if added_unique_chunks:
                 reasoning_trace.append(f"Hop {hops_taken}: Added {len(added_unique_chunks)} unique chunks to context.")
            elif retrieved_chunks: # Only log 'no new chunks' if we actually retrieved something
                 reasoning_trace.append(f"Hop {hops_taken}: No new unique chunks added to context (all duplicates/similar).")


            # Combine context for reasoning
            full_accumulated_text_for_reasoning = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

            # --- Reasoning ---
            action, value = self._reason_and_refine_query(
                 original_query,
                 full_accumulated_text_for_reasoning,
                 search_query_history,
                 hops_taken
            )
            reasoning_trace.append(f"Hop {hops_taken}: Reasoning result -> Action='{action}', Value='{value[:100]}...'")

            # --- Process Reasoning Outcome ---
            if action == "ANSWER_COMPLETE":
                 self.logger.info("Reasoning concluded: Sufficient context likely found or cannot improve.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> ANSWER_COMPLETE. Proceeding to final answer.")
                 break
            elif action == "NEXT_QUERY":
                 # Update query for the *next* iteration
                 next_raw_query = value
                 current_query = self._transform_query(next_raw_query) # Apply transformations to the new query too
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> NEXT_QUERY = '{current_query}'")
                 # Loop continues...
            elif action == "ERROR":
                 self.logger.error(f"Error during reasoning step: {value}. Stopping multihop process.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> ERROR: {value}. Proceeding to final answer with current context.")
                 break
            else:
                 self.logger.warning(f"Unexpected action '{action}' from reasoning step. Stopping.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> UNEXPECTED ACTION '{action}'. Proceeding to final answer.")
                 break

        # --- End of Loop ---
        if hops_taken == max_hops and action != "ANSWER_COMPLETE":
            self.logger.info("Max hops reached. Proceeding to final answer generation.")
            reasoning_trace.append(f"Max hops ({max_hops}) reached.")

        reasoning_trace.append("--- Final Answer Generation ---")

        # --- Final Answer Generation ---
        final_answer = self._generate_final_answer(original_query, accumulated_context_list)

        # --- Confidence & Evaluation ---
        confidence_score = None
        evaluation_metrics = None
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_answer.startswith("LLM_ERROR:") and final_context_str:
            confidence_score = self._assess_answer_confidence(final_answer, final_context_str, original_query)
            reasoning_trace.append(f"Confidence Score: {confidence_score:.2f}" if confidence_score is not None else "Confidence Score: N/A")

            if request_evaluation:
                 evaluation_metrics = self.evaluate_response(original_query, final_answer, final_context_str)
                 # Avoid overly long traces by summarizing metrics dict
                 eval_summary = {k: v for k, v in evaluation_metrics.items() if 'rating' in k}
                 reasoning_trace.append(f"Evaluation Metrics: {eval_summary}") # Log summary


        # --- Format Trace & Generate Graph ---
        formatted_reasoning = self._format_reasoning_trace(reasoning_trace)
        graph_filename = None
        if generate_graph:
            # Pass the actual unique queries used for retrieval steps
            unique_search_queries = list(dict.fromkeys(search_query_history))
            graph_filename = self._generate_hop_graph(reasoning_trace, unique_search_queries)


        # --- Wrap Up ---
        process_end_time = time.time()
        processing_time = process_end_time - process_start_time
        self.logger.info(f"--- CNT Query Process Finished ---")
        self.logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Final Answer Snippet: {final_answer[:200]}...")

        debug_info = {
            "processing_time_s": round(processing_time, 2),
            "hops_taken": hops_taken,
            "queries_used": list(dict.fromkeys(search_query_history)), # Unique queries used
            "final_context_length": len(final_context_str),
            "vector_db_type": self.vector_db_type,
            # Simple cache hit counting based on debug logs (can be improved)
            "cache_hits_embedding": self.llm_cache.get("embedding_hits", 0), # Requires updating cache logic to track hits
            "cache_hits_llm": self.llm_cache.get("llm_hits", 0), # Requires updating cache logic to track hits
            "graph_filename": graph_filename
        }

        # --- Record Feedback ---
        user_rating = record_user_feedback.get('rating') if record_user_feedback else None
        user_comment = record_user_feedback.get('comment') if record_user_feedback else None
        self.record_feedback(
            query=original_query,
            answer=final_answer,
            hop_count=hops_taken,
            final_context=final_context_str, # Maybe truncate context for feedback log?
            reasoning_trace=reasoning_trace, # Pass raw trace
            search_query_history=list(dict.fromkeys(search_query_history)), # Pass unique queries
            user_rating=user_rating,
            user_comment=user_comment,
            evaluation_metrics=evaluation_metrics # Pass full metrics dict
        )

        return {
            "final_answer": final_answer,
            "source_info": f"Synthesized from context retrieved over {hops_taken} hop(s) using {self.vector_db_type} DB.",
            "reasoning_trace": reasoning_trace, # Raw trace
            "formatted_reasoning": formatted_reasoning, # Formatted trace
            "confidence_score": confidence_score,
            "evaluation_metrics": evaluation_metrics,
            "debug_info": debug_info
        }

# --- End of CNTRagSystem Class ---

# Example Usage (adjust paths as needed)
if __name__ == "__main__":
    # Load API key from .env file if not already set
    # load_dotenv()
    # if not os.getenv("GOOGLE_API_KEY"):
    #    raise ValueError("GOOGLE_API_KEY not found.")

    print("Setting up CNT RAG System...")
    # --- Instantiate the RAG system ---
    # Make sure the documents_path points to your CNT papers/data
    cnt_rag_system = CNTRagSystem(
        documents_path="../../DataSets/Experimental Dataset/*", # Use the uploaded folder path pattern
        vector_db_type="inmemory",       # Use inmemory for quick testing
        vector_db_path="vector_db_cnt_test.csv", # Path if saving/loading csv/inmemory
        feedback_db_path="cnt_feedback_test.csv",
        log_file="cnt_rag_test.log",
        log_level="INFO",
        graph_dir="cnt_rag_graphs_test",
        use_embedding_cache=True,
        use_llm_cache=True
    )
    print("System Initialized.")

    # --- Process a query relevant to the uploaded data ---
    # Example query based on the provided docx files
    question = "What was the final height achieved in the experiment with temperature 963?" #[cite: 1, 3]

    print(f"\nProcessing Query: {question}")
    results = cnt_rag_system.process_query(
        question=question,
        top_k=3,
        max_hops=5, # Reduce hops for potentially simpler scientific Q&A
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    # --- Display Results ---
    print("\n--- Final Answer ---")
    print(results["final_answer"])
    # print("\n--- Source Info ---")
    # print(results["source_info"])
    print("\n--- Confidence Score ---")
    print(f"{results['confidence_score']:.2f}" if results['confidence_score'] is not None else "N/A")
    print("\n--- Evaluation Metrics ---")
    print(results["evaluation_metrics"])
    print("\n--- Formatted Reasoning Trace ---")
    print(results["formatted_reasoning"])
    print("\n--- Debug Info ---")
    print(results["debug_info"])
    if results["debug_info"].get("graph_filename"):
        print(f"\nReasoning graph saved to: {results['debug_info']['graph_filename']}")

    # Example 2: More complex query potentially requiring reasoning/context combination
    question_2 = "Compare the highest growth rate for experiments run at temperatures above 900." #[cite: 1, 19, 37]

    print(f"\nProcessing Query 2: {question_2}")
    # results_2 = cnt_rag_system.process_query(question=question_2, max_hops=3)
    results_2 = cnt_rag_system.process_query(
        question=question_2,
        top_k=20,
        max_hops=10, # Reduce hops for potentially simpler scientific Q&A
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    print("\n--- Final Answer (Query 2) ---")
    print(results_2["final_answer"])
    print("\n--- Debug Info (Query 2) ---")
    print(results_2["debug_info"])
    
    # Example 3: More complex query potentially requiring reasoning/context combination
    question_3 = "Which temperature is better for achieving the highest growth rate for CNT?" #[cite: 1, 19, 37]

    print(f"\nProcessing Query 2: {question_2}")
    # results_2 = cnt_rag_system.process_query(question=question_2, max_hops=3)
    results_3 = cnt_rag_system.process_query(
        question=question_3,
        top_k=20,
        max_hops=10, # Reduce hops for potentially simpler scientific Q&A
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    print("\n--- Final Answer (Query 3) ---")
    print(results_3["final_answer"])
    print("\n--- Debug Info (Query 3) ---")
    print(results_3["debug_info"])
    
    # Example 4: More complex query potentially requiring reasoning/context combination
    question_4 = "Among the experiments with catalyst thickness between 1.4 and 1.5, which Input Conditions achieved the maximum Final height?"

    print(f"\nProcessing Query 4: {question_4}")
    # results_2 = cnt_rag_system.process_query(question=question_2, max_hops=3)
    results_4 = cnt_rag_system.process_query(
        question=question_4,
        top_k=20,
        max_hops=10, # Reduce hops for potentially simpler scientific Q&A
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    print("\n--- Final Answer (Query 4) ---")
    print(results_4["final_answer"])
    print("\n--- Debug Info (Query 4) ---")
    print(results_4["debug_info"])
    
    # Example 4: More complex query potentially requiring reasoning/context combination
    question_5 = "For experiments conducted at temperatures above 800°C, was there a consistent trend observed between the catalyst thickness and the final height achieved? Describe the findings based on the available data."

    print(f"\nProcessing Query 5: {question_5}")
    # results_2 = cnt_rag_system.process_query(question=question_2, max_hops=3)
    results_5 = cnt_rag_system.process_query(
        question=question_5,
        top_k=20,
        max_hops=10, # Reduce hops for potentially simpler scientific Q&A
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    print("\n--- Final Answer (Query 5) ---")
    print(results_5["final_answer"])
    print("\n--- Debug Info (Query 5) ---")
    print(results_5["debug_info"])
    
    