import os
import json
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
import matplotlib.pyplot as plt

# --- NLP & Evaluation Libraries ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from bert_score import score as bert_score_calc
import torch
from dotenv import load_dotenv

# Note: Adjust the path if your .env file is in a different location
load_dotenv('../../.env')

# ==============================================================================
# --- 1. SETUP & ONE-TIME LOADING ---
# ==============================================================================
print("Downloading NLTK 'punkt' for tokenization...")
nltk.download('punkt', quiet=True)
print("Loading evaluation models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BI_ENCODER = SentenceTransformer('all-MiniLM-L6-v2', device=device)
CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Evaluation models loaded successfully.")

# ==============================================================================
# --- 2. METRIC COMPUTATION & NEW HELPER FUNCTION ---
# ==============================================================================
def clean_llm_answer(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text); text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\n+', ' ', text).strip()
    return re.sub(r'\s{2,}', ' ', text)

# --- NEW HELPER FUNCTION TO PREVENT JSON ERRORS ---
def sanitize_for_json_prompt(text: str) -> str:
    """Escapes characters that can break JSON when inserted into a prompt."""
    if not isinstance(text, str): return ""
    return text.replace('\\', '\\\\') \
               .replace('"', '\\"') \
               .replace('\n', ' ') \
               .replace('\r', '') \
               .replace('\t', ' ')

# (The rest of the compute functions are unchanged and correct)
def compute_bleu(r,c): return sentence_bleu([nltk.word_tokenize(r.lower())],nltk.word_tokenize(c.lower()),smoothing_function=SmoothingFunction().method1)
def compute_rouge(r,c): s=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True).score(r,c); return {"rouge-1":s['rouge1'].fmeasure,"rouge-2":s['rouge2'].fmeasure,"rouge-L":s['rougeL'].fmeasure}
def compute_tfidf_cosine_similarity(r,c):
    try:
        if not r.strip() or not c.strip(): return 0.0
        v=TfidfVectorizer().fit_transform([r,c]); return float(sklearn_cosine_similarity(v[0:1],v[1:2])[0][0])
    except: return 0.0
def compute_bert_score(r,c):
    try: P,R,F1=bert_score_calc([c],[r],lang="en",verbose=False); return {"precision":P.mean().item(),"recall":R.mean().item(),"f1":F1.mean().item()}
    except: return {"precision":0.0,"recall":0.0,"f1":0.0}
def compute_st_scores(r,c):
    try:
        embeddings = BI_ENCODER.encode([r, c], convert_to_tensor=True)
        r_e, c_e = embeddings[0], embeddings[1]
        return {"cosine_similarity":util.cos_sim(r_e,c_e).item(),"dot_product":util.dot_score(r_e,c_e).item()}
    except: return {"cosine_similarity":0.0,"dot_product":0.0}
def compute_sas(q,a):
    try: return float(CROSS_ENCODER.predict([(q,a)],show_progress_bar=False)[0])
    except: return 0.0

# ==============================================================================
# --- 3. MAIN EVALUATION WORKFLOWS (WITH FIX) ---
# ==============================================================================
def run_quantitative_evaluation(config, eval_data):
    all_results = []
    for i, item in enumerate(eval_data):
        # NOTE: Ensure your JSON file has the key 'RAG' or change this to 'LLM'
        question, gt_clean, gen_clean = item['question'], clean_llm_answer(item['answer']), clean_llm_answer(item['RAG'])
        st_scores = compute_st_scores(gt_clean, gen_clean)
        scores = {
            'BLEU': compute_bleu(gt_clean, gen_clean), 'ROUGE': compute_rouge(gt_clean, gen_clean),
            'TF-IDF_Cosine_Sim': compute_tfidf_cosine_similarity(gt_clean, gen_clean),
            'BERTScore': compute_bert_score(gt_clean, gen_clean),
            'ST_Cosine_Sim': st_scores['cosine_similarity'], 'ST_Dot_Product': st_scores['dot_product'],
            'SAS_Score': compute_sas(question, gen_clean)
        }
        all_results.append({ "question": question, "ground_truth": gt_clean, "generated_answer": gen_clean, "scores": scores })
    return all_results

def run_llm_grader_evaluation(config, eval_data, grader_model):
    """Uses an LLM to 'grade' each generated answer against the ground truth."""
    print("\n--- Running LLM-as-a-Grader Evaluation ---")
    grader_results = []
    for i, item in enumerate(eval_data):
        print(f"  Grader is evaluating item {i+1}/{len(eval_data)}...")
        
        # --- FIX IS APPLIED HERE ---
        # 1. Sanitize the inputs before placing them in the prompt
        question = sanitize_for_json_prompt(item['question'])
        ground_truth = sanitize_for_json_prompt(item['answer'])
        # NOTE: Ensure your JSON file has the key 'RAG' or change this to 'LLM'
        generated_answer = sanitize_for_json_prompt(item['RAG'])
        
        # 2. Use the sanitized variables in the prompt
        prompt = f"""You are a meticulous and impartial AI evaluator for a scientific research context. Your task is to grade a generated answer based on a ground truth answer.
**Evaluation Criteria (Score 1-5):**
1.  **Correctness:** How factually accurate is the generated answer when compared to the ground truth? Does it contain any contradictions? (1=Completely incorrect, 5=Perfectly correct)
2.  **Completeness:** Does the generated answer cover all the key points and important information present in the ground truth answer? (1=Misses almost all key points, 5=Covers all key points)
3.  **Conciseness:** Is the generated answer free of redundant, irrelevant, or verbose information compared to the ground truth? (1=Very verbose and irrelevant, 5=Perfectly concise)
**The Question:** {question}
**Ground Truth (The perfect answer key):** {ground_truth}
**Generated Answer (The one to be graded):** {generated_answer}
**Your Task:** Provide a score (1-5) for the Generated Answer on each criterion. Provide a brief justification for your scores. Output ONLY a JSON object:
{{ "scores": {{ "correctness": <score>, "completeness": <score>, "conciseness": <score> }}, "justification": "<Your brief analysis here>" }}"""
        
        try:
            raw_grade = grader_model.generate_content(prompt).text
            json_match = re.search(r'\{.*\}', raw_grade, re.DOTALL)
            if json_match:
                # The extracted JSON string is now much more likely to be valid
                grade_json = json.loads(json_match.group(0))
                grader_results.append({ "question": item['question'], **grade_json }) # Use original question for report
            else:
                raise ValueError("No JSON object found in grader response.")
        except Exception as e:
            print(f"    An error occurred while grading Q{i+1}: {e}")
            
    return grader_results

# ==============================================================================
# --- 4. VISUALIZATION FUNCTION ---
# ==============================================================================
def generate_visualizations(quantitative_results, grader_results, config):
    # This function is correct
    print("\n--- Generating Visualizations ---")
    data_to_plot, q_scores = {}, [res['scores'] for res in quantitative_results]
    data_to_plot['BERTScore F1'] = [s['BERTScore']['f1'] for s in q_scores]
    data_to_plot['ST Cosine Sim'] = [s['ST_Cosine_Sim'] for s in q_scores]
    data_to_plot['SAS Score'] = [s['SAS_Score'] for s in q_scores]
    if grader_results:
        grader_correctness = [res['scores']['correctness'] for res in grader_results if 'scores' in res and 'correctness' in res['scores']]
        if grader_correctness:
            data_to_plot['LLM Grader: Correctness'] = grader_correctness

    if not data_to_plot:
        print("No data available for plotting.")
        return
        
    labels, values = list(data_to_plot.keys()), list(data_to_plot.values())
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(values); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(f'Distribution of Evaluation Scores ({config.EXPERIMENT_NAME})', fontsize=16)
    ax.set_ylabel('Score'); ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, f'score_distribution_plot_{config.EXPERIMENT_NAME}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Score distribution plot saved to '{plot_path}'"); plt.close()

# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    class Config:
        EVALUATION_DATA_PATH = '../question_set/model-comp-adv.json'
        OUTPUT_DIR = 'evaluation_report'
        EXPERIMENT_NAME = "baseline_rag_performance"

    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    try:
        with open(config.EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f: eval_data = json.load(f)
        print(f"Successfully loaded {len(eval_data)} items from '{config.EVALUATION_DATA_PATH}'")
    except Exception as e: print(f"Failed to load or parse data file: {e}"); exit()
        
    quantitative_results = run_quantitative_evaluation(config, eval_data)
    json_path = os.path.join(config.OUTPUT_DIR, f'quantitative_report_{config.EXPERIMENT_NAME}.json')
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(quantitative_results, f, indent=2)
    print(f"\nDetailed quantitative results saved to '{json_path}'")

    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not set in .env file.")
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using the model name you specified
        grader_llm = genai.GenerativeModel('gemini-2.5-pro')
        
        grader_results = run_llm_grader_evaluation(config, eval_data, grader_llm)
        if grader_results:
            df_grader = pd.DataFrame(grader_results)
            csv_path = os.path.join(config.OUTPUT_DIR, f'llm_grader_report_{config.EXPERIMENT_NAME}.csv')
            df_grader.to_csv(csv_path, index=False)
            print(f"LLM Grader results saved to '{csv_path}'")
        else:
            print("LLM Grader did not return any valid results.")
        
    except Exception as e:
        print(f"\nCould not run LLM-as-a-Grader. Error: {e}")
        grader_results = []

    if quantitative_results:
        generate_visualizations(quantitative_results, grader_results, config)

    # The final print summary block
    if not quantitative_results: print("No quantitative results were generated."); exit()
    q_scores = [res['scores'] for res in quantitative_results]
    print("\n" + "="*80)
    print("--- AGGREGATED EVALUATION SCORES (AVERAGE ACROSS ALL QUESTIONS) ---")
    print(f"Experiment: '{config.EXPERIMENT_NAME}' | Items Evaluated: {len(quantitative_results)}")
    print("="*80)
    print("\n--- Lexical Similarity (Keyword-Based) ---")
    print(f"  BLEU Score                      : {np.mean([s['BLEU'] for s in q_scores]):.4f} (Higher is better)")
    print(f"  ROUGE-1 (Unigram Recall)        : {np.mean([s['ROUGE']['rouge-1'] for s in q_scores]):.4f} (Higher is better)")
    print(f"  ROUGE-2 (Bigram Recall)         : {np.mean([s['ROUGE']['rouge-2'] for s in q_scores]):.4f} (Higher is better)")
    print(f"  ROUGE-L (Longest Subsequence)   : {np.mean([s['ROUGE']['rouge-L'] for s in q_scores]):.4f} (Higher is better)")
    print(f"  TF-IDF Cosine Similarity        : {np.mean([s['TF-IDF_Cosine_Sim'] for s in q_scores]):.4f} (Higher is better)")
    print("\n--- Semantic Similarity (Meaning-Based vs. Reference Answer) ---")
    print(f"  BERTScore Precision             : {np.mean([s['BERTScore']['precision'] for s in q_scores]):.4f} (Higher is better, less hallucination)")
    print(f"  BERTScore Recall                : {np.mean([s['BERTScore']['recall'] for s in q_scores]):.4f} (Higher is better, more complete)")
    print(f"  BERTScore F1                    : {np.mean([s['BERTScore']['f1'] for s in q_scores]):.4f} (Higher is better, balanced P and R)")
    print(f"  Sentence-T Cosine Similarity    : {np.mean([s['ST_Cosine_Sim'] for s in q_scores]):.4f} (Higher is better)")
    print(f"  Sentence-T Dot Product          : {np.mean([s['ST_Dot_Product'] for s in q_scores]):.4f} (Higher is better, unnormalized)")
    print("\n--- Relevance & Correctness (Meaning-Based vs. Original Question) ---")
    print(f"  Semantic Answer Similarity (SAS): {np.mean([s['SAS_Score'] for s in q_scores]):.4f} (Higher is better)")
    if grader_results:
        print("\n--- LLM-as-a-Grader Scores (1-5 Scale) ---")
        avg_correct = np.mean([res['scores']['correctness'] for res in grader_results if 'scores' in res and 'correctness' in res.get('scores', {})])
        avg_complete = np.mean([res['scores']['completeness'] for res in grader_results if 'scores' in res and 'completeness' in res.get('scores', {})])
        avg_concise = np.mean([res['scores']['conciseness'] for res in grader_results if 'scores' in res and 'conciseness' in res.get('scores', {})])
        print(f"  Avg. Correctness              : {avg_correct:.4f} (Higher is better)")
        print(f"  Avg. Completeness             : {avg_complete:.4f} (Higher is better)")
        print(f"  Avg. Conciseness              : {avg_concise:.4f} (Higher is better)")
    print("="*80)