import config
from utils import setup_logging
from llm_interface import LLMInterface
from vector_store import get_vector_store
import os
import re

logger = setup_logging("INFO", "fls_qa_test.log")

llm = LLMInterface(
    api_key=config.API_KEY,
    model_id=config.DEFAULT_MODEL_ID,
    embedding_model_id=config.DEFAULT_TEXT_EMBEDDING_MODEL,
    generation_config=config.DEFAULT_GENERATION_CONFIG,
    use_embedding_cache=False,
    use_llm_cache=False,
    logger=logger
)

vector_store = get_vector_store(
    vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
    vector_db_path=config.DEFAULT_VECTOR_DB_PATH,
    logger=logger
)

build_success = vector_store.load_or_build(
    documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
    chunk_settings={
        'strategy': config.DEFAULT_CHUNK_STRATEGY,
        'size': config.DEFAULT_CHUNK_SIZE,
        'overlap': config.DEFAULT_CHUNK_OVERLAP
    },
    embedding_interface=llm
)

if not build_success:
    logger.error("Failed to build vector store.")
    exit(1)


# Load FLS Summary
fls_file_path = "/home/vs4ky/ARES_VIDURA/DataSets/FLS/cnt_FLS.txt"

if not os.path.exists(fls_file_path):
    logger.error(f"FLS summary file not found at: {fls_file_path}")
    exit(1)

with open(fls_file_path, 'r', encoding='utf-8') as f:
    retrieved_fls_statements = f.read().strip()

# User Query
user_query = input("Enter your CNT question: ").strip()
query_embedding = llm.get_embedding(user_query)

# Retrieve domain knowledge
top_k = 5
retrieved_chunks = vector_store.query(query_embedding, top_k=top_k)
retrieved_domain_knowledge = "\n\n".join([c['text'] for c in retrieved_chunks])

# 5. parser - for the dg with ret. prompt
def extract_conditions(text):
    param_match = re.findall(r'\b(temperature|thickness|pressure|sigma_rate|catalyst)\b', text, re.IGNORECASE)
    prop_match = re.findall(r'\b(height|growth rate|chirality|diameter|defects|quality|conductance)\b', text, re.IGNORECASE)
    return list(set(param_match)), list(set(prop_match))

parameters, target_properties = extract_conditions(user_query)
parameter_conditions = "N/A"  # Optional: Use regex to extract numeric filters
target_property_conditions = "N/A"
objective = "Explain why..." if "why" in user_query.lower() else "Summarize..."

# prompt
structured_prompt = f"""
You are an expert in material science and Fuzzy Linguistic Summaries (FLS).  
Your task is to carefully synthesize an explanation by using the provided FLS statements and domain knowledge, following these instructions carefully.

# User Query:
{user_query}

# Parsed Information:
- Parameters: {parameters}
- Parameter Conditions: {parameter_conditions}
- Target Properties: {target_properties}
- Target Property Conditions: {target_property_conditions}
- Objective: {objective}  

# Retrieved FLS Statements:
{retrieved_fls_statements}

# Retrieved Domain Knowledge:
{retrieved_domain_knowledge}

# Instructions:
1. Analyze the FLS Statements:
   - Identify any overlapping or potentially conflicting statements.
   - If contradictions exist, explicitly mention them.
   - Highlight statements that support or contradict the user query.

2. Align with Physics/Domain Knowledge:
   - Check if the physics supports or conflicts with the FLS statements.
   - Use physics to explain the patterns or contradictions found in the FLS.

3. Synthesize a Reasoned Answer:
   - Combine the insights from FLS and physics.
   - Provide a reasoned, evidence-supported answer.
   - Clearly state if the answer is uncertain or if multiple interpretations are possible due to conflicting FLS statements.

4. Transparency and Explanation Structure:
   - Always provide a concluding paragraph that summarizes the reasoning and confidence level.
   - Use phrases like "Based on both FLS and domain knowledge...", "While the FLS shows conflicting patterns...", "Physics suggests that...".

# Important Notes:
- Do not assume any facts outside the provided FLS and physics knowledge.
- If the FLS is contradictory or incomplete, state this explicitly and reason carefully using physics.
"""

response = llm.generate_response(structured_prompt)

# print("\n" + "="*80)
# print(f"[PROMPT SENT]:\n{prompt}")
# print("="*80)
# print(f"[LLM RESPONSE]:\n{response}")
# print("="*80)


print("\n" + "="*100)
print("[PROMPT SENT]:\n")
print(structured_prompt)
print("="*100)
print("[LLM RESPONSE]:\n")
print(response)
print("="*100)