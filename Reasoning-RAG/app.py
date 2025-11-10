import os
import json
from flask import Flask, render_template, request, Response, jsonify
from typing import List, Dict, Any

# --- Imports from your original app.py ---
# Ensure all these files (config.py, utils.py, etc.) are in the same directory
import config
import config_ablation as settings
from utils import setup_logging, format_reasoning_trace, prepare_interactive_graph_data
from llm_interface import LLMInterface
from vector_store import get_vector_store
from graph_db import Neo4jGraphDB
from rag_core import CNTRagSystem
from evaluation import load_feedback_history
# --- End original imports ---

# Initialize Flask App
# The 'template_folder' argument tells Flask to look for index.html in a folder named 'templates'
app = Flask(__name__, template_folder='templates')

# --- Global RAG System Initialization ---
# This code runs ONCE when the server starts, replacing Streamlit's @st.cache_resource
logger = None
rag_system = None

def load_rag_system_on_startup():
    """
    Initializes and returns the CNTRagSystem.
    This is based on your original load_rag_system() function,
    but with Streamlit-specific code (st.spinner, st.error) removed.
    """
    global logger # Make logger global so routes can use it
    
    log_dir = os.path.dirname(config.DEFAULT_LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Initializing Flask App and RAG System ---")
    
    try:
        llm_interface = LLMInterface(
            llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER,
            google_api_key=config.GOOGLE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
            logger=logger
        )
        graph_db = Neo4jGraphDB(logger=logger)
        vector_store = get_vector_store(
            vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
            vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
            logger=logger
        )
        chunk_settings = {'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP}
        
        logger.info("Loading knowledge base...")
        build_success = vector_store.load_or_build(
            documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
            chunk_settings=chunk_settings,
            embedding_interface=llm_interface,
            graph_db=graph_db
        )
        
        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed.")
             raise RuntimeError("Fatal Error: The knowledge base (vector store) could not be loaded.")

        logger.info("Vector store and Knowledge Graph are ready.")
        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)
        
        system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            graph_db=graph_db,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history,
            user_type="advanced",
            use_knowledge_graph=settings.USE_KNOWLEDGE_GRAPH,
            use_multi_hop=settings.USE_MULTI_HOP,
            use_source_tagging=settings.USE_SOURCE_TAGGING,
            use_proactive_suggestions=settings.USE_PROACTIVE_SUGGESTIONS,
            use_llm_evaluation=settings.USE_LLM_EVALUATION
        )
        
        logger.info("--- CNTRagSystem Initialized Successfully for Flask ---")
        return system
        
    except Exception as e:
        if logger:
            logger.exception(f"An error occurred during RAG system initialization: {e}")
        else:
            print(f"A critical error occurred before logger was initialized: {e}")
        # We raise the exception to stop the server from starting
        # in a broken state.
        raise e

try:
    rag_system = load_rag_system_on_startup()
except Exception as e:
    print(f"FATAL: Could not initialize RAG system. Server cannot start. Error: {e}")
    # In a real production environment, you might want to exit
    # os._exit(1) 


# --- Flask Routes ---

@app.route('/')
def index():
    """
    Serves the main index.html file from the 'templates' folder.
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    The main chat API endpoint. It receives a JSON payload with 'message'
    and 'history', and streams back a response.
    """
    if not rag_system:
        return {"error": "RAG system is not initialized."}, 500

    try:
        data = request.json
        user_message = data.get('message')
        chat_history = data.get('history', [])

        if not user_message:
            return {"error": "No message provided."}, 400

        def generate_chat_response():
            """
            A generator function that yields RAG events as JSON strings.
            This allows for a streaming response.
            """
            live_trace = [] # Store trace events for graph generation
            
            try:
                # 1. Generate contextual query
                contextual_query = rag_system.generate_contextual_query(
                    chat_history=chat_history, 
                    new_question=user_message
                )
                
                # 2. Stream the query process
                # We yield each event as a JSON string followed by a newline
                # This is called "newline-delimited JSON" (ndjson)
                for event in rag_system.stream_query_process(
                    question=contextual_query,
                    max_hops=4,
                    use_query_expansion=True
                ):
                    # Store trace for graph
                    if event.get("event") == "trace":
                        live_trace.append(event.get("data"))
                        
                    yield json.dumps(event) + '\n'
                
                # --- NEW: After stream, generate and send graph data ---
                if settings.GENERATE_FLOW_GRAPH:
                    logger.info("Stream finished. Generating flow graph data.")
                    query_history = getattr(rag_system, 'search_query_history', [])
                    graph_data = prepare_interactive_graph_data(live_trace, query_history)
                    
                    if graph_data:
                        graph_event = {
                            "event": "graph_data",
                            "data": graph_data
                        }
                        yield json.dumps(graph_event) + '\n'
                        logger.info("Successfully sent graph data to frontend.")
                # --- End new section ---

            except Exception as e:
                logger.exception(f"Error during stream generation: {e}")
                error_event = {"event": "error", "data": str(e)}
                yield json.dumps(error_event) + '\n'

        # Return a streaming response
        return Response(generate_chat_response(), mimetype='application/x-ndjson')

    except Exception as e:
        logger.exception(f"Error in /chat route: {e}")
        return {"error": f"An internal server error occurred: {e}"}, 500
# --- NEW: Route to fetch chunk text ---
@app.route('/get_chunk/<chunk_id>')
def get_chunk(chunk_id):
    """
    Gets the full text for a specific chunk_id.
    """
    if not rag_system:
        return jsonify({"error": "RAG system is not initialized."}), 500
        
    try:
        chunk_data = rag_system.get_chunk_by_id(chunk_id)
        if chunk_data and 'chunk_text' in chunk_data:
            return jsonify(chunk_data)
        else:
            return jsonify({"error": "Chunk not found."}), 404
            
    except Exception as e:
        logger.exception(f"Error in /get_chunk/{chunk_id}: {e}")
        return jsonify({"error": str(e)}), 500
# --- End new route ---


# --- Run the App ---
if __name__ == '__main__':
    # Note: debug=True is great for development but should be False in production
    app.run(debug=True, port=5000)

