import os
import logging
import re
import uuid
from typing import List, Optional
from streamlit_agraph import Node, Edge
import textwrap


# --- Logging Setup ---
def setup_logging(log_level_str: str, log_file: str) -> logging.Logger:
    """Sets up logging to console and file."""
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        logging.basicConfig(level=logging.INFO)
        logging.warning(f'Invalid log level: {log_level_str}. Defaulting to INFO.')
        numeric_level = logging.INFO

    logger = logging.getLogger("CNTRAG")
    logger.setLevel(numeric_level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.error(f"Failed to set up file handler at {log_file}: {e}")

    logger.info(f"Logging initialized at level {log_level_str.upper()} to console and {log_file}")
    return logger

# --- Formatting and Visualization ---
def format_reasoning_trace(reasoning_trace: List[str]) -> str:
    """Formats the reasoning trace list into a more readable string."""
    # (Keep the existing formatting logic from the original script)
    formatted = ["**Reasoning Process:**\n" + "-"*20]
    hop_details = []
    current_hop = 0

    for step in reasoning_trace:
        step = step.strip()
        if step == "START":
            formatted.append("1. Initial Query Transformation/Expansion")
            continue
        elif step.startswith("--- Hop"):
            if hop_details:
                formatted.append(f"\n**Hop {current_hop}:**")
                formatted.extend([f"  - {d}" for d in hop_details])
                hop_details = []
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
        elif step.startswith("Reasoning ->"):
            pass
        elif step == "--- Final Answer Generation ---":
            if hop_details:
                formatted.append(f"\n**Hop {current_hop}:**")
                formatted.extend([f"  - {d}" for d in hop_details])
            formatted.append("\n**Final Answer Generation**")
        elif step.startswith("Confidence Score:") or step.startswith("Evaluation Metrics:"):
             formatted.append(f"  - {step}")
        elif step.startswith("Max hops"):
             hop_details.append(step)
        else:
            if current_hop > 0 and step not in ["START", "--- Final Answer Generation ---"]:
                hop_details.append(f"Step: {step}")

    if hop_details and (not formatted or not formatted[-1].startswith("**Hop")):
        formatted.append(f"\n**Hop {current_hop}:**")
        formatted.extend([f"  - {d}" for d in hop_details])

    formatted.append("-"*20)
    return "\n".join(formatted)


# def generate_hop_graph(reasoning_trace: List[str], query_history: List[str], graph_dir: str, logger: logging.Logger) -> Optional[str]:
#     """Generates a Graphviz graph visualizing the RAG hops. (Revised for Readability)"""
#     # (Keep the existing graph generation logic from the original script)
#     # (Make sure graphviz is installed: pip install graphviz)
#     try:
#         from graphviz import Digraph
#     except ImportError:
#         logger.warning("Graphviz Python library not found. Cannot generate graph. Please run 'pip install graphviz'.")
#         return None

#     if not reasoning_trace:
#         return None

#     try:
#         os.makedirs(graph_dir, exist_ok=True)
#         graph_id = str(uuid.uuid4())[:8]
#         filename = os.path.join(graph_dir, f"rag_hop_graph_{graph_id}")

#         dot = Digraph(comment='RAG Multi-Hop Process', format='png')

#         # --- Readability Improvements ---
#         dot.attr(dpi='200')
#         dot.attr(rankdir='TB', size='12,12')
#         dot.attr(label=f'RAG Process Flow (ID: {graph_id})', fontsize='24')
#         dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontsize='12')
#         dot.attr('edge', fontsize='10')

#         node_counter = 0
#         last_structural_node_name = f"n{node_counter}"

#         start_query = query_history[0] if query_history else "Initial Query (N/A)"
#         start_label = f"Start\nQuery:\n{start_query[:150]}{'...' if len(start_query)>150 else ''}"
#         dot.node(last_structural_node_name, start_label, shape='Mdiamond', fontsize='12')
#         node_counter += 1

#         hop_num = 0
#         query_idx_for_hop = 0
#         current_hop_nodes = {}

#         for i, step in enumerate(reasoning_trace):
#             step = step.strip()

#             if step.startswith("--- Hop"):
#                 hop_num = int(step.split(" ")[2]) if len(step.split(" ")) > 2 else hop_num + 1
#                 current_hop_nodes = {}

#                 if hop_num > 1:
#                     # Simplified logic for finding the query for the hop
#                     # Assumes the query triggering this hop is the last 'NEXT_QUERY' value logged
#                     # or the next query in the history if no explicit 'NEXT_QUERY' was logged before this hop marker
#                     prev_next_query_value = None
#                     for j in range(i - 1, -1, -1):
#                         prev_step = reasoning_trace[j].strip()
#                         if prev_step.startswith(f"Hop {hop_num-1}: Reasoning -> NEXT_QUERY"):
#                              match = re.search(r"NEXT_QUERY\s*=\s*'(.*?)'", prev_step) # Look for the logged query value
#                              if match:
#                                  prev_next_query_value = match.group(1).strip()
#                                  break
#                         elif prev_step.startswith("--- Hop"): # Stop search at previous hop marker
#                              break

#                     found_idx = -1
#                     start_search_idx = query_idx_for_hop + 1 # Start search from the query *after* the previous hop's query
#                     for q_idx in range(start_search_idx, len(query_history)):
#                          # If we found an explicit NEXT_QUERY log, match it
#                          if prev_next_query_value and query_history[q_idx] == prev_next_query_value:
#                              found_idx = q_idx
#                              break
#                          # If no explicit log, assume the next query in sequence corresponds to this hop
#                          elif not prev_next_query_value:
#                              found_idx = q_idx
#                              break
#                     if found_idx != -1:
#                          query_idx_for_hop = found_idx
#                     else:
#                         # Fallback: If matching fails, increment index, but log a warning
#                         logger.warning(f"Graph: Could not reliably determine specific query for Hop {hop_num} based on trace/history matching. Using next sequential query.")
#                         query_idx_for_hop = min(query_idx_for_hop + 1, len(query_history) - 1)


#                 query_text = query_history[query_idx_for_hop] if query_idx_for_hop < len(query_history) else f"Query for Hop {hop_num} (N/A)"
#                 query_label = f"Hop {hop_num}\nQuery:\n{query_text[:150]}{'...' if len(query_text)>150 else ''}"
#                 query_node_name = f"n{node_counter}"

#                 dot.node(query_node_name, query_label, fillcolor='lightgreen')
#                 dot.edge(last_structural_node_name, query_node_name, label=f"Start Hop {hop_num}")
#                 last_structural_node_name = query_node_name
#                 current_hop_nodes['query'] = query_node_name
#                 node_counter += 1

#             elif step.startswith(f"Hop {hop_num}: Reasoning result"):
#                 from_node = current_hop_nodes.get('query', last_structural_node_name)
#                 reasoning_node_name = f"n{node_counter}"
#                 action_match = re.search(r"Action='([^']*)'", step)
#                 value_match = re.search(r"Value='([^']*)", step)

#                 action = action_match.group(1) if action_match else "N/A"
#                 value_raw = value_match.group(1).replace("'...", "") if value_match else "N/A"
#                 value = value_raw[:70] + ('...' if len(value_raw) > 70 else '')

#                 reasoning_label = f"Hop {hop_num} Reasoning\nAction: {action}\nValue: '{value}'"
#                 dot.node(reasoning_node_name, reasoning_label, fillcolor='lightyellow')
#                 dot.edge(from_node, reasoning_node_name, label="Process Context")
#                 current_hop_nodes['reasoning'] = reasoning_node_name
#                 node_counter += 1

#                 if action == "ANSWER_COMPLETE" or action == "ERROR":
#                     final_node_name = f"n{node_counter}"
#                     dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange', fontsize='14')
#                     dot.edge(reasoning_node_name, final_node_name, label=action)
#                     last_structural_node_name = final_node_name
#                     node_counter += 1

#         final_node_exists = any(f'"n{j}" [label="Final Answer Generation"' in node_def for node_def in dot.body for j in range(node_counter))

#         if not final_node_exists:
#              connect_from_node = current_hop_nodes.get('reasoning', last_structural_node_name)
#              final_node_name = f"n{node_counter}"
#              dot.node(final_node_name, "Final Answer Generation", shape='ellipse', fillcolor='orange', fontsize='14')
#              label = "Max Hops Reached" if any("Max hops reached" in s for s in reasoning_trace) else "Proceed to Final"
#              if connect_from_node and connect_from_node.startswith("n"):
#                  dot.edge(connect_from_node, final_node_name, label=label)
#              else:
#                  logger.warning("Graph generation: Could not determine valid node to connect to final answer.")


#         # Render the graph
#         output_path = dot.render(filename=filename, view=False, cleanup=True)
#         logger.info(f"Generated RAG hop graph (High Res): {output_path}")
#         return output_path

#     except FileNotFoundError:
#          logger.error("Graphviz executable not found in PATH. Cannot generate graph. Please install Graphviz (see https://graphviz.org/download/).")
#          return None
#     except Exception as e:
#         logger.exception(f"Failed to generate RAG hop graph: {e}")
#         return None
    
def generate_hop_graph(reasoning_trace: List[str], query_history: List[str], graph_dir: str, logger: logging.Logger) -> Optional[str]:
    """Generates a more readable Graphviz graph visualizing the RAG hops."""
    try:
        from graphviz import Digraph
    except ImportError:
        logger.warning("Graphviz library not found. Cannot generate graph. 'pip install graphviz'")
        return None

    if not reasoning_trace:
        return None

    # Helper to wrap long text for better display in nodes
    def wrap_text(text, width=40):
        return '\n'.join(textwrap.wrap(text, width=width))

    try:
        os.makedirs(graph_dir, exist_ok=True)
        graph_id = str(uuid.uuid4())[:8]
        filename = os.path.join(graph_dir, f"rag_hop_graph_{graph_id}")

        dot = Digraph(comment='RAG Multi-Hop Process', format='png')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='12', fontname='Helvetica')
        dot.attr('edge', fontsize='10', fontname='Helvetica')

        # Start Node
        start_query = query_history[0] if query_history else "Initial Query"
        start_label = f"Start\nQuery: {wrap_text(start_query)}"
        dot.node('start', start_label, shape='Mdiamond', fillcolor='lightblue')
        last_node = 'start'

        # Process Hops
        hop_num = 0
        query_idx = 0
        for step in reasoning_trace:
            if step.startswith("--- Hop"):
                hop_num = int(re.search(r'\d+', step).group())
                
                # Create a Query node for the hop
                query_text = query_history[query_idx] if query_idx < len(query_history) else f"Query for Hop {hop_num}"
                query_node_name = f"query_{hop_num}"
                query_label = f"Hop {hop_num} Query\n{wrap_text(query_text)}"
                dot.node(query_node_name, query_label, fillcolor='lightgreen')
                dot.edge(last_node, query_node_name, label=f"Refine Query (Hop {hop_num})")
                
                last_node = query_node_name
                if any("NEXT_QUERY" in s for s in reasoning_trace[:reasoning_trace.index(step)]):
                    query_idx += 1
            
            elif "Reasoning result" in step:
                # Create a Reasoning node
                reasoning_node_name = f"reasoning_{hop_num}"
                action_match = re.search(r"Action='([^']*)'", step)
                value_match = re.search(r"Value='([^']*)", step)
                action = action_match.group(1) if action_match else "N/A"
                value = value_match.group(1).replace("'...", "") if value_match else "N/A"
                
                reasoning_label = f"Hop {hop_num} Reasoning\nAction: {action}\nValue: {wrap_text(value)}"
                dot.node(reasoning_node_name, reasoning_label, fillcolor='lightyellow', shape='ellipse')
                dot.edge(last_node, reasoning_node_name, label="Process Context")
                last_node = reasoning_node_name
        
        # Final Answer Node
        dot.node('final_answer', "Final Answer Generation", shape='octagon', fillcolor='orange')
        dot.edge(last_node, 'final_answer', label="Proceed to Final")

        output_path = dot.render(filename=filename, view=False, cleanup=True)
        logger.info(f"Generated readable RAG hop graph: {output_path}")
        return output_path

    except Exception as e:
        logger.exception(f"Failed to generate readable RAG hop graph: {e}")
        return None
    
def prepare_interactive_graph_data(reasoning_trace: List[str], query_history: List[str]) -> dict:
    """Prepares node and edge lists for streamlit-agraph with readable labels."""
    
    nodes = []
    edges = []

    def wrap_text(text, width=30):
        return '\n'.join(textwrap.wrap(text, width=width))

    # Start Node
    start_query = query_history[0] if query_history else "Initial Query"
    start_label = f"Start\n{wrap_text(start_query)}"
    nodes.append(Node(id='start', label=start_label, shape='diamond', color='#ADD8E6'))
    last_node_id = 'start'

    hop_num = 0
    query_idx = 0
    for step in reasoning_trace:
        if step.startswith("--- Hop"):
            hop_num = int(re.search(r'\d+', step).group())
            
            query_text = query_history[query_idx] if query_idx < len(query_history) else f"Query for Hop {hop_num}"
            query_node_id = f"query_{hop_num}"
            query_label = f"Hop {hop_num} Query\n{wrap_text(query_text)}"
            nodes.append(Node(id=query_node_id, label=query_label, color='#90EE90'))
            edges.append(Edge(source=last_node_id, target=query_node_id, label=f"Refine (Hop {hop_num})"))
            
            last_node_id = query_node_id
            # A more robust way to track the query index
            if "NEXT_QUERY" in step or "Reasoning result" in step:
                 if any(f"Hop {hop_num}: Reasoning -> NEXT_QUERY" in s for s in reasoning_trace if s.startswith(f"Hop {hop_num}")):
                     query_idx = min(query_idx + 1, len(query_history) - 1)
        
        elif "Reasoning result" in step and f"reasoning_{hop_num}" not in [n.id for n in nodes]:
            reasoning_node_id = f"reasoning_{hop_num}"
            action_match = re.search(r"Action='([^']*)'", step)
            value_match = re.search(r"Value='([^']*)", step)
            action = action_match.group(1) if action_match else "N/A"
            value = value_match.group(1).replace("'...", "") if value_match else "N/A"
            
            reasoning_label = f"Hop {hop_num} Reasoning\nAction: {action}\nValue: {wrap_text(value)}"
            nodes.append(Node(id=reasoning_node_id, label=reasoning_label, color='#FFFFE0', shape='ellipse'))
            edges.append(Edge(source=last_node_id, target=reasoning_node_id, label="Process Context"))
            last_node_id = reasoning_node_id
            
    # Final Answer Node
    nodes.append(Node(id='final_answer', label='Final Answer Generation', shape='octagon', color='#FFA500'))
    edges.append(Edge(source=last_node_id, target='final_answer', label="Proceed to Final"))
    
    return {"nodes": nodes, "edges": edges}