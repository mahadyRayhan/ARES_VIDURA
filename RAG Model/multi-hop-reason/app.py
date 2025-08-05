# app.py (Final Polished Version with UI bug fix)
import streamlit as st
import os
import re
import textwrap
from typing import List, Dict, Any

from streamlit_agraph import agraph, Node, Edge, Config

import config
from utils import setup_logging, format_reasoning_trace, prepare_interactive_graph_data
from llm_interface import LLMInterface
from vector_store import get_vector_store
from graph_db import Neo4jGraphDB
from rag_core import CNTRagSystem
from evaluation import load_feedback_history

# --- App Configuration & Title ---
st.set_page_config(page_title="CNT Research Assistant", layout="wide")
st.title("🔬 CNT Research Assistant")
st.info("Ask a question about Carbon Nanotubes. Expand the 'Details' section under each answer to verify sources and reasoning.")

# --- Initialization & Caching ---
@st.cache_resource
def load_rag_system():
    # ... (This entire function remains the same as your working version)
    # ... (No changes needed here)
    log_dir = os.path.dirname(config.DEFAULT_LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Initializing Streamlit App and RAG System ---")
    try:
        llm_interface = LLMInterface(llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER, google_api_key=config.GOOGLE_API_KEY, openai_api_key=config.OPENAI_API_KEY, logger=logger)
        graph_db = Neo4jGraphDB(logger=logger)
        vector_store = get_vector_store(vector_db_type=config.DEFAULT_VECTOR_DB_TYPE, vector_db_path=config.DEFAULT_VECTOR_DB_PATH, logger=logger)
        chunk_settings = {'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP}
        with st.spinner("Loading knowledge base... This may take a moment on first run."):
            build_success = vector_store.load_or_build(documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN, chunk_settings=chunk_settings, embedding_interface=llm_interface, graph_db=graph_db)
        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed.")
             st.error("Fatal Error: The knowledge base could not be loaded. Please check the logs.")
             return None
        logger.info("Vector store and Knowledge Graph are ready.")
        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)
        rag_system = CNTRagSystem(llm_interface=llm_interface, vector_store=vector_store, graph_db=graph_db, logger=logger, feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH, feedback_history=feedback_history, user_type="advanced")
        logger.info("--- CNTRagSystem Initialized Successfully for Streamlit ---")
        return rag_system
    except Exception as e:
        logger.exception(f"An error occurred during RAG system initialization: {e}")
        st.error(f"An error occurred during initialization: {e}")
        st.exception(e)
        return None

rag_system = load_rag_system()

# --- UI Helper Functions ---
def display_sources(sources, message_key):
    # ... (This function remains unchanged)
    if not sources:
        st.caption("No specific sources were cited for this response.")
        return
    for i, source in enumerate(sources):
        doc_name = source.get('document_name', 'N/A')
        page_num = source.get('page_number', 'N/A')
        chunk_id = source.get('chunk_id', f'unknown_{i}')
        button_key = f"btn_{message_key}_{i}"
        col1, col2 = st.columns([4, 1])
        with col1: st.info(f"**Source {i+1}:** {doc_name} (Page: {page_num})")
        with col2:
            if st.button("Show Text", key=button_key, use_container_width=True):
                st.session_state[f'show_chunk_{button_key}'] = not st.session_state.get(f'show_chunk_{button_key}', False)
        if st.session_state.get(f'show_chunk_{button_key}', False):
            with st.spinner("Fetching chunk text..."):
                chunk_data = rag_system.get_chunk_by_id(chunk_id)
                if chunk_data and 'chunk_text' in chunk_data:
                    st.text_area("Full Chunk Text", value=chunk_data['chunk_text'], height=200, disabled=True, key=f'text_{button_key}')
                else: st.error("Could not retrieve the full text for this chunk.")

@st.dialog("Interactive Reasoning Flow")
def show_graph_dialog(graph_data):
    # ... (This function remains unchanged)
    st.info("You can drag nodes to rearrange the graph and use your mouse wheel to zoom.")
    if graph_data:
        config = Config(width=800, height=600, directed=True, physics=False, hierarchical={"enabled": True, "sortMethod": "directed", "shakeTowards": "roots"}, node={'size': 25, 'font': {'size': 12}}, edge={'font': {'align': 'top'}})
        agraph(nodes=graph_data["nodes"], edges=graph_data["edges"], config=config)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?", "reasoning": "Initial greeting."}]

# Display all historical messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Show Details"):
                source_tab, reasoning_tab, graph_tab = st.tabs(["🔬 Sources", "🧠 Reasoning", "📈 Flow"])
                with source_tab:
                    display_sources(message.get("sources", []), f"msg_{i}")
                with reasoning_tab:
                    st.text(message.get("reasoning", "No trace available."))
                with graph_tab:
                    if message.get("graph_data"):
                        if st.button("Launch Graph", key=f"graph_btn_{i}"):
                            show_graph_dialog(message.get("graph_data"))

# --- Main Chat Logic ---
if rag_system:
    if prompt := st.chat_input("Ask your question about CNTs..."):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream the assistant's response
        with st.chat_message("assistant"):
            # Main placeholder for the answer
            answer_placeholder = st.empty()
            
            # --- THE FIX: We create the final expander and tabs from the start ---
            with st.expander("Show Details (Live)", expanded=True) as details_expander:
                source_tab, reasoning_tab, graph_tab = st.tabs(["🔬 Sources", "🧠 Reasoning Text", "📈 Reasoning Flow"])
                
                # Placeholders within each tab
                with source_tab:
                    source_placeholder = st.empty()
                with reasoning_tab:
                    reasoning_placeholder = st.empty()
                with graph_tab:
                    graph_placeholder = st.empty()

            # Initialize variables
            final_answer, suggestions, sources = "", "", []
            live_trace = []
            
            try:
                contextual_query = rag_system.generate_contextual_query(
                    chat_history=st.session_state.messages[:-1], 
                    new_question=prompt
                )
                
                # Stream events from the RAG system
                for event in rag_system.stream_query_process(question=contextual_query):
                    event_type, data = event.get("event"), event.get("data")

                    if event_type == "trace":
                        live_trace.append(data)
                        reasoning_placeholder.text(format_reasoning_trace(live_trace))
                    
                    elif event_type == "final_answer":
                        final_answer = data
                        answer_placeholder.markdown(final_answer)
                    
                    elif event_type == "suggestions":
                        suggestions = data
                        answer_placeholder.markdown(final_answer + suggestions)
                    
                    elif event_type == "sources":
                        sources = data
                        with source_placeholder.container():
                            # Use a unique key for the live message's sources
                            display_sources(sources, f"msg_live_{len(st.session_state.messages)}")
                    
                    elif event_type == "done":
                        break
                
                # After the loop, prepare the final data to be stored
                full_response_content = final_answer + suggestions
                final_formatted_trace = format_reasoning_trace(live_trace)
                
                query_history = getattr(rag_system, 'search_query_history', [])
                graph_data = prepare_interactive_graph_data(live_trace, query_history) if live_trace and query_history else None

                # Display the final graph button
                with graph_placeholder.container():
                    if graph_data:
                        if st.button("Launch Interactive Graph", key=f"graph_btn_live_{len(st.session_state.messages)}"):
                            show_graph_dialog(graph_data)
                    else:
                        st.caption("Could not generate graph data.")

                # Store the final, complete message in the session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_content,
                    "sources": sources,
                    "reasoning": final_formatted_trace,
                    "graph_data": graph_data
                })

            except Exception as e:
                logger.exception(f"An error occurred during query processing: {e}")
                st.error(f"An error occurred: {e}")
else:
    st.warning("RAG system is offline. Please check the logs.")