# main.py
import os
import time
import logging
from typing import List

# Import configuration and components
import config # Ensure this imports your updated config.py
from utils import setup_logging
from llm_interface import LLMInterface # Ensure this imports your updated llm_interface.py
from vector_store import get_vector_store, VectorStore
from evaluation import load_feedback_history
from rag_core import CNTRagSystem # Assuming rag_core.py is also updated if needed
from graph_db import Neo4jGraphDB # Import the new class

def load_test_questions(filepath: str) -> List[str]:
    """Loads questions from a text file, one question per line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return questions
    except FileNotFoundError:
        print(f"Error: Test questions file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading test questions file: {e}")
        return []

def run_test_query(rag_system: CNTRagSystem, question: str, index: int, user_type: str):
    """Runs a single test query and prints the results with detailed sources."""
    print(f"\n--- Processing Query {index+1} for '{user_type}' user ---")
    print(f"Question: {question}")

    results = rag_system.process_query(
        question=question,
        top_k=config.DEFAULT_TOP_K,
        max_hops=config.DEFAULT_MAX_HOPS,
        use_query_expansion=True,
        request_evaluation=True,
        generate_graph=True
    )

    print("\n--- Final Answer ---")
    print(results.get("final_answer", "N/A"))
    
    # --- NEW: Print Proactive Suggestions ---
    suggestions = results.get("proactive_suggestions")
    if suggestions:
        print(suggestions) # The suggestions string is already formatted

    print("\n--- Detailed Sources Used for Final Context ---")
    retrieved_sources = results.get("retrieved_sources")
    if retrieved_sources and isinstance(retrieved_sources, list):
        if not retrieved_sources:
            print("No specific sources were successfully retrieved or used for the final context.")
        else:
            for i, src_info in enumerate(retrieved_sources):
                score = src_info.get('retrieval_score', 'N/A')
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                print(
                    f"Source {i+1}: \n"
                    f"  - Document: '{src_info.get('document_name', 'N/A')}'\n"
                    f"  - Page: {src_info.get('page_number', 'N/A')}\n"
                    f"  - Chunk ID: {src_info.get('chunk_id', 'N/A')}\n"
                    f"  - Retrieval Score: {score_str}\n"
                    f"  - Snippet: \"{src_info.get('text_snippet', '')}\""
                )
    elif retrieved_sources:
        print(f"Source Information: {retrieved_sources}")

    confidence = results.get("confidence_score")
    print("\n--- Confidence Score ---")
    print(f"{confidence:.2f}" if confidence is not None else "N/A")

    print("\n--- Debug Info ---")
    debug_info = results.get("debug_info", {})
    print(f"  Processing Time: {debug_info.get('processing_time_s', 'N/A')}s")
    print(f"  Hops Taken: {debug_info.get('hops_taken', 'N/A')}")
    print(f"  Vector Store: {debug_info.get('vector_db_type', 'N/A')}")
    print(f"  Queries Used: {debug_info.get('queries_used', 'N/A')}")
    print(f"  LLM Cache Hits: {debug_info.get('llm_cache_hits', 'N/A')}")
    print(f"  Embedding Cache Hits: {debug_info.get('embedding_cache_hits', 'N/A')}")
    graph_file = debug_info.get("graph_filename")
    if graph_file:
        print(f"  Reasoning graph saved to: {graph_file}")
    print("-" * 30)


if __name__ == "__main__":
    # --- Setup ---
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Starting CNT RAG Application ---")
    logger.info(f"CONFIG: Using LLM Provider: {config.DEFAULT_GENERATIVE_LLM_PROVIDER.upper()}")
    if config.DEFAULT_GENERATIVE_LLM_PROVIDER == "google":
        logger.info(f"CONFIG: Google Model: {config.DEFAULT_GOOGLE_MODEL_ID}")
        logger.info(f"CONFIG: Google Embedding Model: {config.DEFAULT_GOOGLE_EMBEDDING_MODEL}")
    elif config.DEFAULT_GENERATIVE_LLM_PROVIDER == "openai":
        logger.info(f"CONFIG: OpenAI Chat Model: {config.DEFAULT_OPENAI_CHAT_MODEL}")
        logger.info(f"CONFIG: OpenAI Embedding Model: {config.DEFAULT_OPENAI_EMBEDDING_MODEL}")


    # --- Determine User Type (Example: from config or fixed for now) ---
    # For simplicity, let's assume 'user_type' might be used to adjust prompts inside CNTRagSystem or its components.
    # If your CNTRagSystem takes `user_type` in `__init__`, ensure it's passed.
    # Your previous main.py had a `user_type_input_str` but it wasn't clear how CNTRagSystem used it.
    # Let's assume CNTRagSystem might use it to tailor prompts.
    # If not, this can be simplified.
    user_type_setting = "advanced" # Or load from config: config.DEFAULT_USER_TYPE
    logger.info(f"Setting user type context to: {user_type_setting}")


    # --- Initialize Components ---
    try:
        # LLMInterface initialization now handles provider selection internally based on config
        llm_interface = LLMInterface(
            llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER,
            google_api_key=config.GOOGLE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
            # Google specific args
            google_model_id=config.DEFAULT_GOOGLE_MODEL_ID,
            google_embedding_model_id=config.DEFAULT_GOOGLE_EMBEDDING_MODEL,
            google_generation_config=config.DEFAULT_GOOGLE_GENERATION_CONFIG,
            # OpenAI specific args
            openai_chat_model=config.DEFAULT_OPENAI_CHAT_MODEL,
            openai_embedding_model=config.DEFAULT_OPENAI_EMBEDDING_MODEL,
            openai_generation_config=config.DEFAULT_OPENAI_GENERATION_CONFIG,
            # Common
            use_embedding_cache=config.DEFAULT_USE_EMBEDDING_CACHE,
            use_llm_cache=config.DEFAULT_USE_LLM_CACHE,
            logger=logger
        )
        
        # Initialize Neo4j Graph Database
        logger.info("Initializing Knowledge Graph connection...")
        graph_db = Neo4jGraphDB(logger=logger)

        vector_store = get_vector_store(
            vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
            vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
            logger=logger
        )

        chunk_settings = {
            'strategy': config.DEFAULT_CHUNK_STRATEGY,
            'size': config.DEFAULT_CHUNK_SIZE,
            'overlap': config.DEFAULT_CHUNK_OVERLAP,
        }

        logger.info("Loading or building vector store AND populating knowledge graph...")
        # The build process now handles both KG and Vector DB
        build_success = vector_store.load_or_build(
            documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
            chunk_settings=chunk_settings,
            embedding_interface=llm_interface,
            graph_db=graph_db # Pass the graph_db instance here
        )

        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed. Exiting.")
             exit(1)
        logger.info("Vector store and Knowledge Graph are ready.")

        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)

        # Assuming CNTRagSystem's __init__ might take user_type if it customizes behavior
        # If your CNTRagSystem __init__ does not take `user_type`, remove it from the call.
        cnt_rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            graph_db=graph_db,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history,
            # user_type=user_type_setting, # Pass if CNTRagSystem uses it in __init__
            graph_dir=config.DEFAULT_GRAPH_DIR,
            max_context_tokens=config.MAX_CONTEXT_TOKENS,
            char_to_token_ratio=config.CHAR_TO_TOKEN_RATIO,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )
        logger.info(f"CNTRagSystem Initialized (using LLM: {cnt_rag_system.llm_interface.llm_provider.upper()}).") # User type from CNTRagSystem if it stores it

    except Exception as e:
        logger.exception(f"Fatal error during RAG system initialization: {e}")
        # Make sure to close the DB connection on error
        if 'graph_db' in locals() and graph_db:
            graph_db.close()
        exit(1)

    # --- Load Test Questions ---
    test_questions = load_test_questions(config.DEFAULT_TEST_QUESTIONS_PATH)

    # --- Run Test Queries ---
    if not test_questions:
        logger.warning("No test questions loaded. Exiting.")
    else:
        logger.info(f"Loaded {len(test_questions)} test questions. Starting processing...")
        overall_start_time = time.time()
        for i, question in enumerate(test_questions):
            try:
                # Pass user_type_setting to run_test_query for display purposes
                run_test_query(cnt_rag_system, question, i, user_type_setting)
            except Exception as query_err:
                 logger.exception(f"Error processing query {i+1} ('{question[:50]}...'): {query_err}")
                 print(f"\n--- Error processing query {i+1}. Check logs. ---")
        overall_end_time = time.time()
        logger.info(f"Finished processing all {len(test_questions)} questions in {overall_end_time - overall_start_time:.2f} seconds.")

    if 'graph_db' in locals() and graph_db:
        graph_db.close()
    logger.info("--- CNT RAG Application Finished ---")