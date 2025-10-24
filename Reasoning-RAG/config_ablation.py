# config_ablation.py
# This file controls the features of the RAG system for a single run.
# Set to True to enable a feature, False to disable it.

# C1: Full System (All True)
# C2: No Knowledge Graph -> Set USE_KNOWLEDGE_GRAPH to False
# C3: No Multi-Hop -> Set USE_MULTI_HOP to False
# C4: No Suggestions -> Set USE_PROACTIVE_SUGGESTIONS to False
# C5: No LLM Eval -> Set USE_LLM_EVALUATION to False
# C6: No Source Tagging -> Set USE_SOURCE_TAGGING to False
# C7: No Flow Graph -> Set GENERATE_FLOW_GRAPH to False

# --- Core Retrieval & Reasoning ---
USE_KNOWLEDGE_GRAPH = True  # If False, forces pure vector search (ablates KG).
USE_MULTI_HOP = True        # If False, forces a single retrieval hop.
USE_SOURCE_TAGGING = True   # If False, LLM is not prompted to add inline citations.

# --- Ancillary Features ---
USE_PROACTIVE_SUGGESTIONS = True # If False, disables the "You might also be interested in" feature.
USE_LLM_EVALUATION = True        # If False, disables LLM-based evaluation and confidence scoring.
GENERATE_FLOW_GRAPH = True       # If False, disables the generation of the reasoning graph visualization.