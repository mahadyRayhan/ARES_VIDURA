import os
import re
import logging
import chromadb
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Assuming llm_interface.py contains LLMInterface class
from llm_interface import LLMInterface
# Assuming data_processing.py contains parse_and_chunk_documents
from data_processing import parse_and_chunk_documents
import config # Import config for defaults
import json

from graph_db import Neo4jGraphDB # Import the graph DB class

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._is_ready = False

    @abstractmethod
    def load_or_build(self,
                      documents_path_pattern: str,
                      chunk_settings: Dict,
                      embedding_interface: LLMInterface,
                      graph_db: Neo4jGraphDB) -> bool: # Add graph_db parameter
        """Loads existing vector data or builds it from documents."""
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the vector store for relevant chunks."""
        pass

    def is_ready(self) -> bool:
        """Checks if the vector store is initialized and populated."""
        return self._is_ready

    def _build_index_internal(self,
                              documents_path_pattern: str,
                              chunk_settings: Dict,
                              embedding_interface: LLMInterface,
                              graph_db: Neo4jGraphDB) -> Optional[List[Dict[str, Any]]]:
        """Internal helper to parse, chunk, embed documents, and populate KG."""
        self.logger.info("Starting internal index build process with smart chunking...")

        # This now calls the simplified PyPDF2-based parser
        all_chunks_info = parse_and_chunk_documents(
            documents_path_pattern=documents_path_pattern,
            graph_db=graph_db,
            embedding_interface=embedding_interface,
            chunk_size=chunk_settings.get('size', config.DEFAULT_CHUNK_SIZE),
            chunk_overlap=chunk_settings.get('overlap', config.DEFAULT_CHUNK_OVERLAP),
            logger_parent=self.logger
        )

        if not all_chunks_info:
            self.logger.error("No chunks extracted from documents. Index build failed.")
            return None

        chunk_texts = [info["chunk_text"] for info in all_chunks_info]
        self.logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        all_embeddings = embedding_interface.batch_get_embeddings(
            chunk_texts, task_type="RETRIEVAL_DOCUMENT"
        )

        processed_chunks_data = []
        failed_embeddings = 0

        for i, chunk_info in enumerate(all_chunks_info):
            if i < len(all_embeddings) and all_embeddings[i] is not None and len(all_embeddings[i]) > 0:
                processed_chunks_data.append({**chunk_info, "embeddings": all_embeddings[i]})
            else:
                failed_embeddings += 1
                metadata = chunk_info.get('metadata', {})
                self.logger.warning(f"Failed to get valid embedding for chunk {metadata.get('chunk_id', 'N/A')}. Skipping.")

        if not processed_chunks_data:
            self.logger.error("No chunks were successfully embedded. Index build failed.")
            return None

        self.logger.info(f"Successfully embedded {len(processed_chunks_data)} chunks.")
        if failed_embeddings > 0:
            self.logger.warning(f"Failed to embed {failed_embeddings} chunks.")

        return processed_chunks_data

# --- CSV/InMemory Implementation ---
class PandasVectorStore(VectorStore):
    """Vector store using Pandas DataFrame (for CSV or In-Memory)."""

    def __init__(self, db_path: Optional[str], in_memory: bool, logger: logging.Logger):
        super().__init__(logger)
        self.db_path = db_path
        self.in_memory = in_memory
        self.vector_db_df: Optional[pd.DataFrame] = None
        if not in_memory and not db_path:
             raise ValueError("db_path must be provided for CSVVectorStore")

    def _save_dataframe_to_csv(self):
        if self.in_memory or not self.db_path or self.vector_db_df is None or self.vector_db_df.empty:
            return
        try:
            self.logger.info(f"Saving vector database DataFrame to {self.db_path}...")
            df_to_save = self.vector_db_df.copy()

            if 'embeddings' in df_to_save.columns and not df_to_save.empty and isinstance(df_to_save['embeddings'].iloc[0], np.ndarray):
                df_to_save['embeddings_list_str'] = df_to_save['embeddings'].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else json.dumps([]))
            elif 'embeddings' in df_to_save.columns: # Fallback if not ndarray but present
                 df_to_save['embeddings_list_str'] = df_to_save['embeddings'].apply(lambda x: json.dumps(list(x)) if hasattr(x, '__iter__') else json.dumps([]))
            else:
                df_to_save['embeddings_list_str'] = pd.Series([json.dumps([]) for _ in range(len(df_to_save))])


            if 'metadata' in df_to_save.columns and not df_to_save.empty:
                df_to_save['metadata_json_str'] = df_to_save['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else json.dumps({}))
            else:
                df_to_save['metadata_json_str'] = pd.Series([json.dumps({}) for _ in range(len(df_to_save))])


            columns_to_save = ['chunk_text', 'embeddings_list_str', 'metadata_json_str']
            final_columns = [col for col in columns_to_save if col in df_to_save.columns]
            if 'chunk_text' not in final_columns:
                self.logger.error("Cannot save DataFrame: 'chunk_text' column is missing.")
                return

            dir_name = os.path.dirname(self.db_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            df_to_save[final_columns].to_csv(self.db_path, index=False)
            self.logger.info(f"DataFrame successfully saved to {self.db_path} with columns: {final_columns}")
        except Exception as e:
            self.logger.exception(f"Error saving DataFrame to CSV at {self.db_path}: {e}")

    def _load_dataframe_from_csv(self) -> bool:
        if not self.db_path or not os.path.exists(self.db_path):
            return False
        try:
            self.logger.info(f"Loading vector database from {self.db_path}...")
            self.vector_db_df = pd.read_csv(self.db_path)
            
            # Convert strings back to numpy arrays and dicts
            self.vector_db_df['embeddings'] = self.vector_db_df['embeddings_list_str'].apply(lambda x: np.array(json.loads(x)))
            self.vector_db_df['metadata'] = self.vector_db_df['metadata_json_str'].apply(lambda x: json.loads(x))
            self.vector_db_df.drop(columns=['embeddings_list_str', 'metadata_json_str'], inplace=True)
            
            self.logger.info(f"Vector database loaded successfully. Shape: {self.vector_db_df.shape}")
            self._is_ready = True
            return True
        except Exception as e:
            self.logger.exception(f"Unexpected error during CSV DB loading: {e}"); self.vector_db_df = None; return False

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single, complete chunk dictionary by its unique chunk_id."""
        if self.vector_db_df is None or self.vector_db_df.empty:
            self.logger.warning("Cannot get chunk by ID, DataFrame is not loaded.")
            return None
        
        try:
            # Find the row where the chunk_id in the metadata dictionary matches.
            # This is not the most performant way for huge dataframes, but effective.
            # A better long-term solution would be to set the chunk_id as the DataFrame index.
            mask = self.vector_db_df['metadata'].apply(lambda meta: isinstance(meta, dict) and meta.get('chunk_id') == chunk_id)
            result_df = self.vector_db_df.loc[mask]

            if not result_df.empty:
                # Return the first match as a dictionary
                return result_df.iloc[0].to_dict()
            else:
                self.logger.warning(f"No chunk found with ID: {chunk_id}")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving chunk by ID '{chunk_id}': {e}")
            return None

    def load_or_build(self,
                      documents_path_pattern: str,
                      chunk_settings: Dict,
                      embedding_interface: LLMInterface,
                      graph_db: Neo4jGraphDB) -> bool:
        loaded = False
        if not self.in_memory: loaded = self._load_dataframe_from_csv()
        if loaded: self.logger.info("Successfully loaded existing vector data."); self._is_ready = True; return True
        
        if self.in_memory and self.db_path and os.path.exists(self.db_path):
             if self._load_dataframe_from_csv(): self.logger.info("Successfully loaded vector data into memory from CSV."); self._is_ready = True; return True
             else: self.logger.info("Failed to load from CSV for in-memory store, proceeding to build.")

        self.logger.info("Building new vector index...")
        # _build_index_internal now returns List[{"chunk_text":..., "metadata":..., "embeddings":...}]
        processed_chunks_with_embeddings = self._build_index_internal(documents_path_pattern, chunk_settings, embedding_interface, graph_db)

        if processed_chunks_with_embeddings:
            df_data_list = []
            for chunk in processed_chunks_with_embeddings:
                # Ensure all keys are present, provide defaults if not (though they should be)
                df_data_list.append({
                    "chunk_text": chunk.get("chunk_text", ""),
                    "embeddings": np.array(chunk.get("embeddings", [])), # Ensure numpy array
                    "metadata": chunk.get("metadata", {}) # This is the rich dict
                })
            
            self.vector_db_df = pd.DataFrame(df_data_list)
            if self.vector_db_df.empty and processed_chunks_with_embeddings: # Should not happen if data was processed
                self.logger.error("DataFrame is empty after processing chunks. Check data integrity.")
                self._is_ready = False; return False

            self.logger.info(f"Index built successfully. DataFrame shape: {self.vector_db_df.shape}")
            self._is_ready = True
            if not self.in_memory or (self.in_memory and self.db_path): # Save if CSV type OR if in-memory but path provided (for caching)
                self._save_dataframe_to_csv()
            return True
        else:
            self.logger.error("Failed to build index (no processed chunks with embeddings).")
            self._is_ready = False
            return False

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Queries the Pandas DataFrame for top_k most relevant chunks.
        """
        if not self.is_ready() or self.vector_db_df is None or self.vector_db_df.empty:
            self.logger.warning("Pandas vector store is not ready or is empty. Returning no results.")
            return []
        
        required_cols = ['embeddings', 'metadata', 'chunk_text']
        if not all(col in self.vector_db_df.columns for col in required_cols):
            self.logger.error(f"DataFrame is missing one or more required columns {required_cols}. Cannot query.")
            return []

        # --- Vectorized Similarity Calculation for Performance ---
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        
        # Filter out rows with invalid embeddings to prevent errors
        valid_embeddings_mask = self.vector_db_df['embeddings'].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == query_embedding_np.shape[1]
        )
        valid_df = self.vector_db_df[valid_embeddings_mask]

        if valid_df.empty:
            self.logger.warning("No valid/matching dimension embeddings found in DataFrame for querying.")
            return []
            
        try:
            # Stack all valid embeddings into a single numpy matrix
            database_embeddings = np.stack(valid_df["embeddings"].to_numpy())
        except Exception as stack_err:
            self.logger.error(f"Error stacking database embeddings, likely due to inconsistent shapes: {stack_err}")
            return []
            
        # Calculate cosine similarities in one go
        similarities = cosine_similarity(query_embedding_np, database_embeddings)[0]

        # Get top_k indices from the *similarities* array (which corresponds to valid_df)
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k <= 0:
            return []
        
        top_indices_in_filtered_df = np.argsort(similarities)[-actual_top_k:][::-1] # Most similar first

        # Get the corresponding rows from the valid DataFrame
        relevant_rows = valid_df.iloc[top_indices_in_filtered_df]
        
        # --- Format Results ---
        results = []
        for i, (_, row) in enumerate(relevant_rows.iterrows()):
            # The score is from the sorted similarities array
            score = similarities[top_indices_in_filtered_df[i]]
            
            chunk_metadata = row.get('metadata', {})
            # Ensure metadata is a dict, not a string, if loaded from CSV incorrectly
            if isinstance(chunk_metadata, str):
                try: chunk_metadata = json.loads(chunk_metadata)
                except json.JSONDecodeError: chunk_metadata = {}
            
            results.append({
                 "id": chunk_metadata.get("chunk_id", f"df_idx_{row.name}"), # Use df index as fallback id
                 "text": row.get('chunk_text', ""),
                 "metadata": chunk_metadata, # This is the detailed dict
                 "score": float(score)
            })
            
        return results
# --- ChromaDB Implementation ---
class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""
    def __init__(self, collection_name: str = "cnt_collection", persist_directory: Optional[str] = None, logger: logging.Logger = logging.getLogger("CNTRAG")):
        super().__init__(logger)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_client = None
        self.collection = None
        self._initialize_chroma_client()

    def _initialize_chroma_client(self):
        try:
            # from chromadb.config import Settings # Older versions might need this
            client_settings = {}
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                # For newer chromadb versions, persistence is handled by passing path to client
                self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
                self.logger.info(f"Initializing ChromaDB with persistence at: {self.persist_directory}")
            else:
                 self.chroma_client = chromadb.Client() # In-memory client
                 self.logger.info("Initializing ChromaDB in-memory.")

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"ChromaDB client connected. Using collection: '{self.collection_name}'")
            if self.collection.count() > 0: self.logger.info(f"Chroma collection already contains {self.collection.count()} documents."); self._is_ready = True
        except ImportError: self.logger.error("ChromaDB library not installed. pip install chromadb"); raise
        except Exception as e: self.logger.exception(f"ChromaDB setup error: {e}"); raise

    def _populate_chroma(self, processed_chunks_with_embeddings: List[Dict[str, Any]]):
        if not self.collection or not processed_chunks_with_embeddings: self.logger.warning("Chroma collection not init or no chunks."); return False
        
        ids, embeddings_list, documents_list, metadatas_list = [], [], [], []
        embedding_dim = None

        for i, chunk_data in enumerate(processed_chunks_with_embeddings):
            current_embedding = chunk_data.get('embeddings')
            if current_embedding and chunk_data.get('chunk_text') and isinstance(current_embedding, list) and len(current_embedding) > 0:
                if embedding_dim is None: embedding_dim = len(current_embedding)
                elif len(current_embedding) != embedding_dim:
                    self.logger.warning(f"Inconsistent embedding dimension in batch for Chroma: expected {embedding_dim}, got {len(current_embedding)} for chunk_id {chunk_data.get('metadata', {}).get('chunk_id')}. Skipping.")
                    continue

                ids.append(chunk_data.get('metadata', {}).get('chunk_id', f"auto_id_{i}"))
                embeddings_list.append(current_embedding) # Should be list of floats
                documents_list.append(chunk_data['chunk_text'])
                metadatas_list.append(chunk_data['metadata']) # Pass the full metadata dict
            else:
                self.logger.warning(f"Skipping chunk for Chroma due to missing text, embedding, or invalid embedding: id {chunk_data.get('metadata', {}).get('chunk_id')}")


        if not ids: self.logger.warning("No valid chunks found to add to ChromaDB."); return False
        
        batch_size = 100; total_added = 0
        try:
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                if not batch_ids: continue
                self.collection.add(
                    ids=batch_ids,
                    embeddings=embeddings_list[i:i+batch_size],
                    documents=documents_list[i:i+batch_size],
                    metadatas=metadatas_list[i:i+batch_size]
                )
                total_added += len(batch_ids)
                self.logger.info(f"Added batch {i//batch_size + 1} ({len(batch_ids)} docs) to ChromaDB. Total: {total_added}")
            self.logger.info(f"Finished populating ChromaDB. Added {total_added} documents.")
            return True
        except Exception as batch_err: self.logger.exception(f"Error adding batch to ChromaDB: {batch_err}"); return False

    def load_or_build(self,
                    documents_path_pattern: str,
                    chunk_settings: Dict,
                    embedding_interface: LLMInterface,
                    graph_db: Neo4jGraphDB) -> bool: # Add graph_db
        if not self.collection: self.logger.error("ChromaDB collection not initialized."); return False
        if self.collection.count() > 0: self.logger.info(f"ChromaDB collection already has {self.collection.count()} docs."); self._is_ready = True; return True

        self.logger.info(f"ChromaDB collection '{self.collection_name}' empty. Building index...")
        processed_chunks_with_embeddings = self._build_index_internal(documents_path_pattern, chunk_settings, embedding_interface, graph_db)
        if processed_chunks_with_embeddings:
             populated = self._populate_chroma(processed_chunks_with_embeddings)
             self._is_ready = populated
             return populated
        else: self.logger.error("Failed to build index (no processed chunks)."); self._is_ready = False; return False

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the ChromaDB collection for top_k most relevant chunks."""
        if not self.is_ready() or not self.collection:
            self.logger.warning("ChromaDB not ready or collection not initialized.")
            return []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=top_k, 
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                self.logger.debug("No chunks returned from ChromaDB query.")
                return []
            
            # --- Format Results Consistently ---
            formatted_results = []
            res_ids = results["ids"][0]
            res_distances = results.get("distances", [[]])[0]
            res_documents = results.get("documents", [[]])[0]
            res_metadatas = results.get("metadatas", [[]])[0]
            
            for i, chunk_id in enumerate(res_ids):
                # Ensure all result lists are long enough to avoid IndexError
                if i < len(res_documents) and i < len(res_metadatas) and i < len(res_distances):
                    # For cosine distance: similarity = 1 - distance
                    similarity_score = 1.0 - res_distances[i]
                    
                    metadata_dict = res_metadatas[i] or {}
                    
                    formatted_results.append({
                        "id": chunk_id,
                        "text": res_documents[i],
                        "metadata": metadata_dict,
                        "score": float(similarity_score)
                    })
                else:
                    self.logger.warning(f"Inconsistent result lengths from ChromaDB at index {i}. Skipping this result.")
            
            return formatted_results
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred while querying ChromaDB: {e}")
            return []


# --- Factory Function ---
def get_vector_store(vector_db_type: str, vector_db_path: Optional[str], logger: logging.Logger) -> VectorStore:
    db_type = vector_db_type.lower()
    if db_type == "csv":
        logger.info("Creating CSV Vector Store")
        if not vector_db_path: raise ValueError("vector_db_path is required for CSV vector store.")
        return PandasVectorStore(db_path=vector_db_path, in_memory=False, logger=logger)
    elif db_type == "inmemory":
        logger.info("Creating InMemory Vector Store")
        return PandasVectorStore(db_path=vector_db_path, in_memory=True, logger=logger)
    elif db_type == "chroma":
        logger.info("Creating ChromaDB Vector Store")
        persist_dir = vector_db_path if vector_db_path else None
        return ChromaVectorStore(persist_directory=persist_dir, logger=logger)
    else:
        raise ValueError(f"Unknown vector database type: {vector_db_type}")
