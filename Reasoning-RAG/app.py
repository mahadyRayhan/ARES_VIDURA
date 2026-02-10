import os
import json
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

from flask import Flask, render_template, request, Response, jsonify, current_app, Blueprint

# --- Local Imports ---
import config
import config_ablation as settings
from utils import setup_logging, prepare_interactive_graph_data
from llm_interface import LLMInterface
from vector_store import get_vector_store
from graph_db import Neo4jGraphDB
from rag_core import CNTRagSystem
from evaluation import load_feedback_history

# --- Feedback Manager Class ---
class FeedbackManager:
    """Handles writing UI feedback to the CSV file."""
    
    COLUMNS = [
        "timestamp", "session_id", "message_id", "vote",
        "rating_0_10", "comment", "query", "answer"
    ]

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def log_feedback(self, data: Dict[str, Any]) -> None:
        is_new = not self.filepath.exists()
        
        if self.filepath.parent and str(self.filepath.parent) != ".":
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Normalize data
        row = {col: data.get(col, "") for col in self.COLUMNS}

        with self.filepath.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

# --- RAG System Initializer ---
def init_rag_system(app_logger: logging.Logger) -> CNTRagSystem:
    """Initializes the components of the RAG system."""
    app_logger.info("--- Initializing RAG System Components ---")

    try:
        llm_interface = LLMInterface(
            llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER,
            google_api_key=config.GOOGLE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
            logger=app_logger
        )
        graph_db = Neo4jGraphDB(logger=app_logger)
        vector_store = get_vector_store(
            vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
            vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
            logger=app_logger
        )
        
        chunk_settings = {'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP}
        
        app_logger.info("Loading knowledge base...")
        build_success = vector_store.load_or_build(
            documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
            chunk_settings=chunk_settings,
            embedding_interface=llm_interface,
            graph_db=graph_db
        )
        
        if not build_success or not vector_store.is_ready():
             raise RuntimeError("Fatal Error: The knowledge base (vector store) could not be loaded.")

        app_logger.info("Vector store and Knowledge Graph are ready.")
        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, app_logger)
        
        system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            graph_db=graph_db,
            logger=app_logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history,
            user_type="advanced",
            use_knowledge_graph=settings.USE_KNOWLEDGE_GRAPH,
            use_multi_hop=settings.USE_MULTI_HOP,
            use_source_tagging=settings.USE_SOURCE_TAGGING,
            use_proactive_suggestions=settings.USE_PROACTIVE_SUGGESTIONS,
            use_llm_evaluation=settings.USE_LLM_EVALUATION
        )
        
        app_logger.info("--- CNTRagSystem Initialized Successfully ---")
        return system
        
    except Exception as e:
        app_logger.critical(f"RAG Initialization Failed: {e}")
        raise e

# --- Blueprint Definitions ---
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/chat', methods=['POST'])
def chat():
    rag: CNTRagSystem = current_app.extensions['rag_system']
    logger: logging.Logger = current_app.extensions['rag_logger']
    
    if not rag:
        return {"error": "RAG system is not initialized."}, 500

    try:
        data = request.json
        user_message = data.get('message')
        chat_history = data.get('history', [])

        if not user_message:
            return {"error": "No message provided."}, 400

        def generate_chat_response() -> Generator[str, None, None]:
            live_trace = []
            
            try:
                contextual_query = rag.generate_contextual_query(
                    chat_history=chat_history, 
                    new_question=user_message
                )
                
                # Stream the query process
                for event in rag.stream_query_process(
                    question=contextual_query,
                    max_hops=4,
                    use_query_expansion=True
                ):
                    if event.get("event") == "trace":
                        live_trace.append(event.get("data"))
                    yield json.dumps(event) + '\n'
                
                # Generate and send graph data
                if settings.GENERATE_FLOW_GRAPH:
                    logger.info("Stream finished. Generating flow graph data.")
                    query_history = getattr(rag, 'search_query_history', [])
                    graph_data = prepare_interactive_graph_data(live_trace, query_history)
                    
                    if graph_data:
                        yield json.dumps({"event": "graph_data", "data": graph_data}) + '\n'
                        logger.info("Graph data sent.")

            except Exception as e:
                logger.exception(f"Error during stream generation: {e}")
                yield json.dumps({"event": "error", "data": str(e)}) + '\n'

        return Response(generate_chat_response(), mimetype='application/x-ndjson')

    except Exception as e:
        logger.exception(f"Error in /chat route: {e}")
        return {"error": f"Internal server error: {str(e)}"}, 500

@main_bp.route('/get_chunk/<chunk_id>')
def get_chunk(chunk_id):
    rag: CNTRagSystem = current_app.extensions['rag_system']
    
    try:
        chunk_data = rag.get_chunk_by_id(chunk_id)
        if chunk_data and 'chunk_text' in chunk_data:
            return jsonify(chunk_data)
        return jsonify({"error": "Chunk not found."}), 404
    except Exception as e:
        current_app.extensions['rag_logger'].error(f"Error in /get_chunk: {e}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    feedback_manager: FeedbackManager = current_app.extensions['feedback_manager']
    logger: logging.Logger = current_app.extensions['rag_logger']
    
    try:
        data = request.json or {}
        
        # Validation
        vote = (data.get("vote") or "").strip().lower()
        if vote not in {"up", "down"}:
            return jsonify({"error": "Invalid vote. Use 'up' or 'down'."}), 400

        rating = data.get("rating_0_10", "")
        if rating not in [None, ""]:
            try:
                rating_int = int(rating)
                if not (0 <= rating_int <= 10):
                    raise ValueError
                data["rating_0_10"] = rating_int
            except ValueError:
                return jsonify({"error": "rating_0_10 must be integer 0-10."}), 400

        # Construct feedback object
        feedback_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": (data.get("session_id") or "").strip(),
            "message_id": (data.get("message_id") or "").strip(),
            "vote": vote,
            "rating_0_10": data.get("rating_0_10", ""),
            "comment": (data.get("comment") or "").strip(),
            "query": (data.get("query") or "").strip(),
            "answer": (data.get("answer") or "").strip(),
        }

        feedback_manager.log_feedback(feedback_data)
        logger.info(f"UI feedback recorded: {vote}")
        return jsonify({"ok": True})

    except Exception as e:
        logger.exception(f"Error in /feedback: {e}")
        return jsonify({"error": str(e)}), 500


# --- Application Factory ---
def create_app() -> Flask:
    """Factory function to create the Flask application."""
    app = Flask(__name__, template_folder='templates')
    
    # Setup Logging
    log_dir = os.path.dirname(config.DEFAULT_LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # We initialize the logger and attach it to the app extensions
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    
    # Initialize Core Systems
    try:
        rag_system = init_rag_system(logger)
    except Exception as e:
        print(f"FATAL: Failed to initialize RAG system: {e}")
        # Depending on requirements, you might want to exit or let the app start in a broken state
        raise e

    # Initialize Feedback Manager
    feedback_path = getattr(config, "DEFAULT_UI_FEEDBACK_DB_PATH", "cnt_ui_feedback_log.csv")
    feedback_manager = FeedbackManager(feedback_path)

    # Attach extensions to the app instance (Context-safe storage)
    app.extensions['rag_system'] = rag_system
    app.extensions['rag_logger'] = logger
    app.extensions['feedback_manager'] = feedback_manager

    # Register Blueprints
    app.register_blueprint(main_bp)

    return app

# --- Entry Point ---
if __name__ == '__main__':
    # When running locally
    try:
        app = create_app()
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start application: {e}")