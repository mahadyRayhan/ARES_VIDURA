# ablation_runner.py

import os
import csv
import time
import logging
from typing import List, Dict, Any

# Import your existing components
import config
from utils import setup_logging
from llm_interface import LLMInterface
from vector_store import get_vector_store
from rag_core import CNTRagSystem
from graph_db import Neo4jGraphDB

def load_test_questions(filepath: str) -> List[str]:
    """Loads questions from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: Test questions file not found at {filepath}")
        return []

def initialize_system(logger: logging.Logger) -> CNTRagSystem:
    """A helper function to initialize all components of the RAG system."""
    llm_interface = LLMInterface(logger=logger) # Simplified for brevity
    graph_db = Neo4jGraphDB(logger=logger)
    vector_store = get_vector_store(
        vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
        vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
        logger=logger
    )
    vector_store.load_or_build(
        documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
        chunk_settings={'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP},
        embedding_interface=llm_interface,
        graph_db=graph_db
    )
    # Use an empty list for feedback history during tests to avoid contamination
    return CNTRagSystem(
        llm_interface=llm_interface,
        vector_store=vector_store,
        graph_db=graph_db,
        logger=logger,
        feedback_db_path="ablation_feedback_temp.csv", # Use a temp file
        feedback_history=[]
    )

def main():
    logger = setup_logging("INFO", "ablation_study.log")
    logger.info("--- Starting Ablation Study ---")

    # --- 1. Initialize shared components once ---
    logger.info("Initializing shared components (LLM, DBs)...")
    llm_interface = LLMInterface(logger=logger)
    graph_db = Neo4jGraphDB(logger=logger)
    vector_store = get_vector_store(
        vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
        vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
        logger=logger
    )
    # Build the knowledge base once. All RAG instances will use this.
    vector_store.load_or_build(
        documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
        chunk_settings={'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP},
        embedding_interface=llm_interface,
        graph_db=graph_db
    )
    test_questions = load_test_questions(config.DEFAULT_TEST_QUESTIONS_PATH)
    if not test_questions:
        logger.critical("No test questions found. Aborting study.")
        graph_db.close()
        return

    # --- 2. Define Ablation Configurations ---
    # Each dictionary key corresponds to a CNTRagSystem constructor argument.
    configurations = {
        "C1_Full_System":     {"use_knowledge_graph": True,  "use_multi_hop": True,  "use_source_tagging": True,  "use_proactive_suggestions": True,  "use_llm_evaluation": True,  "generate_graph": True},
        "C2_No_KG":           {"use_knowledge_graph": False, "use_multi_hop": True,  "use_source_tagging": True,  "use_proactive_suggestions": True,  "use_llm_evaluation": True,  "generate_graph": False},
        "C3_No_MultiHop":     {"use_knowledge_graph": True,  "use_multi_hop": False, "use_source_tagging": True,  "use_proactive_suggestions": True,  "use_llm_evaluation": True,  "generate_graph": False},
        "C4_No_Suggestions":  {"use_knowledge_graph": True,  "use_multi_hop": True,  "use_source_tagging": True,  "use_proactive_suggestions": False, "use_llm_evaluation": True,  "generate_graph": False},
        "C5_No_LLMEval":      {"use_knowledge_graph": True,  "use_multi_hop": True,  "use_source_tagging": True,  "use_proactive_suggestions": True,  "use_llm_evaluation": False, "generate_graph": False},
        "C6_No_SrcTagging":   {"use_knowledge_graph": True,  "use_multi_hop": True,  "use_source_tagging": False, "use_proactive_suggestions": True,  "use_llm_evaluation": True,  "generate_graph": False},
        "C7_No_Graph":        {"use_knowledge_graph": True,  "use_multi_hop": True,  "use_source_tagging": True,  "use_proactive_suggestions": True,  "use_llm_evaluation": True,  "generate_graph": False},
    }

    # --- 3. Setup Results CSV ---
    results_filepath = "ablation_study_results.csv"
    fieldnames = [
        "config_name", "question_id", "question", "execution_time_s", "hops_taken",
        "confidence_score", "relevance_rating", "faithfulness_rating", "answer_length",
        "num_sources"
    ]
    with open(results_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    logger.info(f"Results will be saved to {results_filepath}")

    # --- 4. Run the Experiments ---
    for config_name, settings in configurations.items():
        logger.info(f"\n{'='*25} RUNNING CONFIGURATION: {config_name} {'='*25}")
        
        # Extract the graph setting, as it's passed to process_query, not the constructor
        generate_graph_setting = settings.pop("generate_graph", True)

        # Initialize a new RAG system instance for EACH configuration with its specific settings
        rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            graph_db=graph_db,
            logger=logger,
            feedback_db_path="ablation_feedback_temp.csv", # Use a temp, non-persistent feedback log
            feedback_history=[],
            **settings  # Unpack the configuration settings into the constructor
        )

        for i, question in enumerate(test_questions):
            logger.info(f"Processing Question {i+1}/{len(test_questions)} for '{config_name}': '{question[:50]}...'")
            
            results = rag_system.process_query(
                question=question,
                generate_graph=generate_graph_setting # Pass the specific setting here
            )
            
            # --- 5. Collect and Save Metrics ---
            metrics = results.get("evaluation_metrics", {}) or {}
            debug_info = results.get("debug_info", {})
            
            row_data = {
                "config_name": config_name,
                "question_id": i + 1,
                "question": question,
                "execution_time_s": debug_info.get("processing_time_s", 0.0),
                "hops_taken": debug_info.get("hops_taken", 0),
                "confidence_score": results.get("confidence_score"),
                "relevance_rating": metrics.get("relevance_rating"),
                "faithfulness_rating": metrics.get("faithfulness_rating"),
                "answer_length": len(results.get("final_answer", "")),
                "num_sources": len(results.get("retrieved_sources", []))
            }
            
            with open(results_filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_data)
            
            logger.info(f"Finished Q{i+1} for {config_name}. Time: {debug_info.get('processing_time_s', 'N/A')}s")
            
    logger.info("--- Ablation Study Finished ---")
    graph_db.close()
    logger.info("Neo4j connection closed.")

if __name__ == "__main__":
    main()