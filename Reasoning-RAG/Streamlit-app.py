# app.py (Final Version with Optimal Configuration and Modern LIGHT UI - CORRECTED CONTRAST & EXPANDER TEXT)
import streamlit as st
import os
import re
import textwrap
from typing import List, Dict, Any

# Streamlit-Agraph is retained for graph visualization
from streamlit_agraph import agraph, Node, Edge, Config

import config
import config_ablation as settings # <-- 1. IMPORT YOUR ABLATION SETTINGS
from utils import setup_logging, format_reasoning_trace, prepare_interactive_graph_data
from llm_interface import LLMInterface
from vector_store import get_vector_store
from graph_db import Neo4jGraphDB
from rag_core import CNTRagSystem
from evaluation import load_feedback_history

# --- CUSTOM UI/CSS FOR MODERN, LIGHT, PROFESSIONAL DESIGN (FINAL CONTRAST CORRECTION) ---
def inject_custom_css():
    """Injects custom CSS for a modern, light, chatbot-style design with high contrast."""
    st.markdown("""
    <style>
    /* Global Light Theme Overrides */
    .stApp {
        background-color: #FFFFFF; /* Pure white background */
        color: #1F2937; /* Deep charcoal text for high contrast */
        font-family: 'Segoe UI', 'Arial', sans-serif; 
    }
    
    /* Custom Header Styling */
    h1 {
        color: #0077B6 !important; /* Professional Blue accent color (Slightly adjusted) */
        font-size: 2.2em;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    /* Info Box Styling */
    .stAlert {
        border-left: 5px solid #0077B6 !important;
        background-color: #F8F9FA !important; 
        color: #1F2937 !important; 
        border-radius: 8px;
    }

    /* Chat Input Styling */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #1F2937; 
        border: 1px solid #CED4DA; 
        border-radius: 25px; 
        padding: 10px 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); 
    }
    
    /* General Button Styling (Primary Accent) */
    .stButton>button {
        background-color: #0077B6; /* Primary deep blue button */
        color: white;
        border-radius: 8px; 
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        font-weight: 600;
        padding: 8px 15px;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #005A8A; /* Darker hover */
        color: white;
    }

    /* Assistant Message Bubble Styling */
    .stChatMessage.assistant > div:first-child {
        background-color: #E6F0FF; /* Soft light blue background for assistant */
        color: #1F2937; 
        border-radius: 15px 15px 15px 5px; 
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* User Message Bubble Styling */
    .stChatMessage.user > div:first-child {
        background-color: #F1F3F5; 
        color: #1F2937; 
        border-radius: 15px 15px 5px 15px; 
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-left: auto; 
    }
    
    /* >>>>>> CRITICAL FIX: EXPANDER HEADER TEXT COLOR <<<<<< */
    .streamlit-expanderHeader > div > p {
        color: #1F2937 !important; /* Forces dark text color on the header */
        font-weight: 600;
        margin: 0;
    }
    
    .streamlit-expanderHeader {
        background-color: #F8F9FA; 
        border-radius: 8px;
        /* The color setting here was previously only affecting the icon, not the text */
        padding: 10px;
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF;
        border-radius: 0 0 8px 8px;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Source Info Box Styling */
    .stNotification > div {
        background-color: #F8F9FA !important; 
        border-left: 3px solid #6C757D !important; 
        color: #1F2937 !important; 
    }
    
    /* Text Area/Code Block Styling */
    textarea[disabled] {
        background-color: #F1F3F5 !important;
        color: #1F2937 !important; 
        border: 1px solid #CED4DA !important;
    }
    
    pre {
        background-color: #F8F9FA !important; 
        color: #1F2937 !important; 
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
    }

    </style>
    """, unsafe_allow_html=True)
# --- END CUSTOM UI/CSS ---

# --- App Configuration & Title ---
st.set_page_config(page_title="CNT Research Assistant", layout="wide")
inject_custom_css() # Inject the custom styles
st.title("🔬 CNT Research Assistant")
st.info("Ask a question about Carbon Nanotubes. Expand the 'Details' section under each answer to verify sources and reasoning.")

# --- Initialization & Caching (NO CHANGE) ---
@st.cache_resource
def load_rag_system():
    log_dir = os.path.dirname(config.DEFAULT_LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Initializing Streamlit App and RAG System with Optimal Config ---")
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
        
        # <-- 2. INITIALIZE CNTRagSystem WITH THE BEST SETTINGS FROM config_ablation.py -->
        rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            graph_db=graph_db,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history,
            user_type="advanced",
            # Pass the flags from config_ablation.py
            use_knowledge_graph=settings.USE_KNOWLEDGE_GRAPH,
            use_multi_hop=settings.USE_MULTI_HOP,
            use_source_tagging=settings.USE_SOURCE_TAGGING,
            use_proactive_suggestions=settings.USE_PROACTIVE_SUGGESTIONS,
            use_llm_evaluation=settings.USE_LLM_EVALUATION
        )
        
        logger.info("--- CNTRagSystem Initialized Successfully for Streamlit ---")
        return rag_system
        
    except Exception as e:
        logger.exception(f"An error occurred during RAG system initialization: {e}")
        st.error(f"An error occurred during initialization: {e}")
        st.exception(e)
        return None

rag_system = load_rag_system()

# --- UI Helper Functions (NO CHANGE) ---
def display_sources(sources, message_key):
    """Displays sources with a cleaner, modern look."""
    if not sources:
        st.caption("No specific sources were cited for this response.")
        return
    
    with st.container(border=True): 
        st.markdown("**Cited Sources:**")
        for i, source in enumerate(sources):
            doc_name = source.get('document_name', 'N/A')
            page_num = source.get('page_number', 'N/A')
            chunk_id = source.get('chunk_id', f'unknown_{i}')
            button_key = f"btn_{message_key}_{i}"
            
            col1, col2 = st.columns([5, 1])
            with col1: 
                st.markdown(f"**Source {i+1}:** *{doc_name}* (Page: {page_num})")
            with col2:
                if st.button("Text", key=button_key, use_container_width=True):
                    st.session_state[f'show_chunk_{button_key}'] = not st.session_state.get(f'show_chunk_{button_key}', False)
            
            if st.session_state.get(f'show_chunk_{button_key}', False):
                with st.spinner("Fetching chunk text..."):
                    chunk_data = rag_system.get_chunk_by_id(chunk_id)
                    if chunk_data and 'chunk_text' in chunk_data:
                        st.text_area(f"Full Chunk Text (Source {i+1})", 
                                     value=chunk_data['chunk_text'], 
                                     height=150, 
                                     disabled=True, 
                                     key=f'text_{button_key}')
                    else: st.error("Could not retrieve the full text for this chunk.")

@st.dialog("Interactive Reasoning Flow")
def show_graph_dialog(graph_data):
    """Displays the interactive AGraph in a styled dialog."""
    st.markdown("### 📈 RAG Flow Visualization")
    st.info("Drag nodes to rearrange the graph and use your mouse wheel to zoom.")
    if graph_data:
        graph_config = Config(
            width=800, 
            height=600, 
            directed=True, 
            physics=False, 
            hierarchical={
                "enabled": True, 
                "sortMethod": "directed", 
                "shakeTowards": "roots"
            }, 
            node={'size': 30, 'font': {'size': 12, 'color': '#FFFFFF'}, 'color': '#0077B6'}, 
            edge={'font': {'align': 'top', 'color': '#1F2937'}, 'color': '#6C757D'} 
        )
        agraph(nodes=graph_data["nodes"], edges=graph_data["edges"], config=graph_config)

# --- Chat History Management (NO CHANGE) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?", "reasoning": "Initial greeting."}]

# Display all historical messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            with st.expander("Show RAG Details (Sources, Reasoning, Flow)"): 
                source_tab, reasoning_tab, graph_tab = st.tabs(["🔬 Sources", "🧠 Reasoning", "📈 Flow"])
                
                with source_tab:
                    display_sources(message.get("sources", []), f"msg_{i}")
                
                with reasoning_tab:
                    st.code(message.get("reasoning", "No trace available."), language='text')
                
                with graph_tab:
                    if message.get("graph_data"):
                        if st.button("Launch Interactive Graph 🚀", key=f"graph_btn_{i}"):
                            show_graph_dialog(message.get("graph_data"))
                    else:
                        st.caption("No flow graph generated for this query.")


# --- Main Chat Logic (NO CHANGE) ---
if rag_system:
    if prompt := st.chat_input("Ask your question about CNTs...", key="chat_input_main"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            
            with st.expander("Show RAG Details (Live) ⚙️", expanded=True) as details_expander:
                source_tab, reasoning_tab, graph_tab = st.tabs(["🔬 Sources", "🧠 Reasoning", "📈 Flow"])
                
                with source_tab: source_placeholder = st.empty()
                with reasoning_tab: reasoning_placeholder = st.empty()
                with graph_tab: graph_placeholder = st.empty()

            final_answer, suggestions, sources = "", "", []
            live_trace = []
            
            try:
                contextual_query = rag_system.generate_contextual_query(
                    chat_history=st.session_state.messages[:-1], 
                    new_question=prompt
                )
                
                # <-- 3. CALL STREAMING FUNCTION WITH OPTIMAL SETTINGS -->
                for event in rag_system.stream_query_process(
                    question=contextual_query,
                    max_hops=4,
                    use_query_expansion=True
                ):
                    event_type, data = event.get("event"), event.get("data")

                    if event_type == "trace":
                        live_trace.append(data)
                        reasoning_placeholder.code(format_reasoning_trace(live_trace), language='text')
                    
                    elif event_type == "final_answer":
                        final_answer = data
                        answer_placeholder.markdown(final_answer)
                    
                    elif event_type == "suggestions":
                        suggestions = data
                        answer_placeholder.markdown(final_answer + suggestions) 
                    
                    elif event_type == "sources":
                        sources = data
                        with source_placeholder.container():
                            display_sources(sources, f"msg_live_{len(st.session_state.messages)}")
                    
                    elif event_type == "done":
                        break
                
                full_response_content = final_answer + suggestions
                final_formatted_trace = format_reasoning_trace(live_trace)
                
                query_history = getattr(rag_system, 'search_query_history', [])
                graph_data = prepare_interactive_graph_data(live_trace, query_history) if live_trace and query_history else None

                with graph_tab:
                    if settings.GENERATE_FLOW_GRAPH and graph_data:
                        if st.button("Launch Interactive Graph 🚀", key=f"graph_btn_live_{len(st.session_state.messages)}"):
                            show_graph_dialog(graph_data)
                    else:
                        st.caption("Graph generation is disabled in the current configuration or no complex flow was detected.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_content,
                    "sources": sources,
                    "reasoning": final_formatted_trace,
                    "graph_data": graph_data
                })

            except Exception as e:
                answer_placeholder.error(f"An error occurred during query processing: {e}")
                logger.exception(f"An error occurred during query processing: {e}")
                st.exception(e)
else:
    st.error("RAG system is offline. Please check the logs and configuration.")