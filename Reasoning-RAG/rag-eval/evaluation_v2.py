import os
import json
import pandas as pd
import numpy as np
import re
import glob
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load environment variables from a .env file in the parent directory
load_dotenv('../../.env')

# ==============================================================================
# --- 1. SETUP & ONE-TIME LOADING ---
# ==============================================================================
print("Downloading NLTK 'punkt' for tokenization...")
nltk.download('punkt', quiet=True)
print("Loading evaluation models (this may take a moment)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BI_ENCODER = SentenceTransformer('all-MiniLM-L6-v2', device=device)
CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
print(f"Evaluation models loaded successfully to '{device}'.")

# ==============================================================================
# --- 2. METRIC COMPUTATION & HELPER FUNCTIONS (Final, Robust Versions) ---
# ==============================================================================
def clean_llm_answer(text: any) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text); text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\n+', ' ', text).strip(); return re.sub(r'\s{2,}', ' ', text)

def sanitize_for_json_prompt(text: any) -> str:
    if not isinstance(text, str): return ""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '').replace('\t', ' ')

def compute_bleu(reference: str, candidate: str) -> float:
    return sentence_bleu([nltk.word_tokenize(reference.lower())], nltk.word_tokenize(candidate.lower()), smoothing_function=SmoothingFunction().method1)

def compute_rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True); scores = scorer.score(reference, candidate)
    return {"rouge-1": scores['rouge1'].fmeasure, "rouge-2": scores['rouge2'].fmeasure, "rouge-L": scores['rougeL'].fmeasure}

def compute_tfidf_cosine_similarity(reference: str, candidate: str) -> float:
    try:
        if not reference.strip() or not candidate.strip(): return 0.0
        vectorizer = TfidfVectorizer().fit_transform([reference, candidate]); return float(sklearn_cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0])
    except ValueError: return 0.0

def compute_bert_score(reference: str, candidate: str) -> dict:
    if not reference.strip() or not candidate.strip(): return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        P, R, F1 = bert_score_calc([candidate], [reference], lang="en", verbose=False, device=device)
        return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}
    except Exception as e: print(f"  [BERTScore Error]: {e}"); return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def compute_st_scores(reference: str, candidate: str) -> dict:
    if not reference.strip() or not candidate.strip(): return {"cosine_similarity": 0.0, "dot_product": 0.0}
    try:
        embeddings = BI_ENCODER.encode([reference, candidate], convert_to_tensor=True)
        return {"cosine_similarity": util.cos_sim(embeddings[0], embeddings[1]).item(), "dot_product": util.dot_score(embeddings[0], embeddings[1]).item()}
    except Exception as e: print(f"  [SentenceTransformer Error]: {e}"); return {"cosine_similarity": 0.0, "dot_product": 0.0}

def compute_sas(question: str, answer: str) -> float:
    MAX_ANSWER_LENGTH_CHARS = 2048
    if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip(): return 0.0
    truncated_answer = answer[:MAX_ANSWER_LENGTH_CHARS]
    try:
        score = CROSS_ENCODER.predict([(question, truncated_answer)], show_progress_bar=False, convert_to_tensor=True)
        return float(score[0])
    except Exception as e: print(f"  [SAS Score CRITICAL Error]: Could not compute for Q: '{question[:30]}...'. Error: {e}"); return 0.0

# ==============================================================================
# --- 3. MAIN EVALUATION WORKFLOWS (with Final Grader Prompt) ---
# ==============================================================================
def run_quantitative_evaluation(eval_data: list) -> list:
    all_results = []; print("\n--- Running Quantitative Evaluation ---")
    for i, item in enumerate(eval_data):
        print(f"  Processing quantitative metrics for item {i+1}/{len(eval_data)}...")
        question, gt_clean, gen_clean = clean_llm_answer(item.get('question')), clean_llm_answer(item.get('answer')), clean_llm_answer(item.get('RAG'))
        st_scores = compute_st_scores(gt_clean, gen_clean)
        scores = { 'BLEU': compute_bleu(gt_clean, gen_clean), 'ROUGE': compute_rouge(gt_clean, gen_clean), 'TF-IDF_Cosine_Sim': compute_tfidf_cosine_similarity(gt_clean, gen_clean), 'BERTScore': compute_bert_score(gt_clean, gen_clean), 'ST_Cosine_Sim': st_scores['cosine_similarity'], 'ST_Dot_Product': st_scores['dot_product'], 'SAS_Score': compute_sas(question, gen_clean) }
        all_results.append({"question": question, "ground_truth": gt_clean, "generated_answer": gen_clean, "scores": scores})
    return all_results

def run_llm_grader_evaluation(eval_data: list, grader_model) -> list:
    print("\n--- Running LLM-as-a-Grader Evaluation ---"); grader_results = []
    for i, item in enumerate(eval_data):
        print(f"  Grader is evaluating item {i+1}/{len(eval_data)}...")
        question, ground_truth, generated_answer = sanitize_for_json_prompt(item.get('question')), sanitize_for_json_prompt(item.get('answer')), sanitize_for_json_prompt(item.get('RAG'))
        
        if not all([question, ground_truth, generated_answer]):
            print(f"    Skipping grading for Q{i+1} due to missing content.")
            continue

        # --- THE DEFINITIVE, RE-ENGINEERED PROMPT ---
        prompt = f"""Your task is to act as an impartial judge and evaluate a generated answer against a ground truth answer.

CRITICAL INSTRUCTION: Your entire response MUST be a single, valid JSON object with no other text before or after it.

--- EVALUATION CRITERIA ---
Score the "Generated Answer" from 1 to 5 on the following criteria:
1.  **Correctness:** How factually accurate is the generated answer compared to the ground truth? (1=Incorrect, 5=Correct)
2.  **Completeness:** Does the generated answer cover all key points from the ground truth? (1=Misses key points, 5=Covers all key points)
3.  **Conciseness:** Is the answer free of irrelevant information? (1=Verbose, 5=Concise)

--- DATA TO EVALUATE ---
Question: "{question}"
Ground Truth Answer: "{ground_truth}"
Generated Answer: "{generated_answer}"

--- REQUIRED JSON OUTPUT ---
{{
  "scores": {{
    "correctness": <score_1_to_5>,
    "completeness": <score_1_to_5>,
    "conciseness": <score_1_to_5>
  }},
  "justification": "<A brief, one-sentence justification for your scores>"
}}"""
        try:
            response = grader_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0),
                safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
            )
            
            if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                block_reason = "Unknown";
                if response.prompt_feedback: block_reason = response.prompt_feedback.block_reason
                raise ValueError(f"Response was blocked by API. Reason: {block_reason}")
            
            raw_grade = response.text
            json_match = re.search(r'\{.*\}', raw_grade, re.DOTALL)
            if json_match:
                grade_json = json.loads(json_match.group(0))
                if 'scores' in grade_json and all(k in grade_json['scores'] for k in ['correctness', 'completeness', 'conciseness']):
                    grader_results.append({"question": item['question'], **grade_json})
                else: raise ValueError("Parsed JSON is missing required score keys.")
            else: raise ValueError("No JSON object found in grader response.")
        except Exception as e:
            raw_grade_for_print = "N/A"
            if 'response' in locals() and hasattr(response, 'text'): raw_grade_for_print = response.text
            print(f"    An error occurred while grading Q{i+1}: {e}\n    LLM Raw Response: '{raw_grade_for_print}'")
    return grader_results

# ==============================================================================
# --- 4. VISUALIZATION & SUMMARY FUNCTIONS ---
# ==============================================================================
def generate_comparison_heatmap(all_configs_summary: list, output_dir: str):
    """Generates a heatmap comparing all metrics across all configurations."""
    if not all_configs_summary:
        print("No summary data to generate a comparison heatmap.")
        return

    df = pd.DataFrame(all_configs_summary).set_index('Configuration')
    
    # Ensure a logical order for the columns in the heatmap
    column_order = [
        'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'TF-IDF Cosine Sim',
        'BERTScore P', 'BERTScore R', 'BERTScore F1', 'ST Cosine Sim', 'ST Dot Product',
        'SAS Score', 'LLM Correctness', 'LLM Completeness', 'LLM Conciseness'
    ]
    
    # Filter out any columns that might not exist (e.g., if LLM grader failed for all)
    # and reorder the existing ones
    df_reordered = df[[col for col in column_order if col in df.columns]]

    plt.figure(figsize=(22, 12))
    heatmap = sns.heatmap(
        df_reordered,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Average Score'}
    )
    plt.title('Comprehensive Comparison of RAG Ablation Study Configurations', fontsize=20, pad=20)
    plt.xlabel('Evaluation Metric', fontsize=14)
    plt.ylabel('RAG Configuration', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    comparison_heatmap_path = os.path.join(output_dir, 'plot_final_comparison_heatmap.png')
    plt.savefig(comparison_heatmap_path, dpi=150)
    print(f"\nFinal comparison heatmap saved to '{comparison_heatmap_path}'")
    plt.close()

def generate_summary_report(quantitative_results: list, grader_results: list, experiment_name: str, output_dir: str):
    """Generates and saves a detailed, human-readable summary report for a single configuration."""
    if not quantitative_results:
        print(f"  Skipping summary report for {experiment_name} due to no quantitative results.")
        return

    q_scores = [res['scores'] for res in quantitative_results]
    num_items = len(q_scores)

    report_lines = [
        "#"*80, f"# --- AGGREGATED EVALUATION SCORES: {experiment_name} ---", f"# Items Evaluated: {num_items}", "#"*80,
        "\n--- Lexical Similarity (Keyword-Based) ---",
        f"  BLEU Score                      : {np.mean([s['BLEU'] for s in q_scores]):.4f} (Higher is better)",
        f"  ROUGE-1 (Unigram Recall)        : {np.mean([s['ROUGE']['rouge-1'] for s in q_scores]):.4f} (Higher is better)",
        f"  ROUGE-2 (Bigram Recall)         : {np.mean([s['ROUGE']['rouge-2'] for s in q_scores]):.4f} (Higher is better)",
        f"  ROUGE-L (Longest Subsequence)   : {np.mean([s['ROUGE']['rouge-L'] for s in q_scores]):.4f} (Higher is better)",
        f"  TF-IDF Cosine Similarity        : {np.mean([s['TF-IDF_Cosine_Sim'] for s in q_scores]):.4f} (Higher is better)",
        "\n--- Semantic Similarity (Meaning-Based vs. Reference Answer) ---",
        f"  BERTScore Precision             : {np.mean([s['BERTScore']['precision'] for s in q_scores]):.4f} (Higher is better, less hallucination)",
        f"  BERTScore Recall                : {np.mean([s['BERTScore']['recall'] for s in q_scores]):.4f} (Higher is better, more complete)",
        f"  BERTScore F1                    : {np.mean([s['BERTScore']['f1'] for s in q_scores]):.4f} (Higher is better, balanced P and R)",
        f"  Sentence-T Cosine Similarity    : {np.mean([s['ST_Cosine_Sim'] for s in q_scores]):.4f} (Higher is better)",
        f"  Sentence-T Dot Product          : {np.mean([s['ST_Dot_Product'] for s in q_scores]):.4f} (Higher is better, unnormalized)",
        "\n--- Relevance & Correctness (Meaning-Based vs. Original Question) ---",
        f"  Semantic Answer Similarity (SAS): {np.mean([s['SAS_Score'] for s in q_scores]):.4f} (Higher is better)",
        "\n--- LLM-as-a-Grader Scores (1-5 Scale) ---"
    ]
    if grader_results:
        report_lines.append(f"  Avg. Correctness              : {np.mean([res['scores']['correctness'] for res in grader_results if 'scores' in res]):.4f} (Higher is better)")
        report_lines.append(f"  Avg. Completeness             : {np.mean([res['scores']['completeness'] for res in grader_results if 'scores' in res]):.4f} (Higher is better)")
        report_lines.append(f"  Avg. Conciseness              : {np.mean([res['scores']['conciseness'] for res in grader_results if 'scores' in res]):.4f} (Higher is better)")
    else:
        report_lines.append("  (No valid LLM Grader results were generated for this run)")
    report_lines.append("\n" + "#"*80)
    
    report_path = os.path.join(output_dir, f'summary_{experiment_name}.txt')
    with open(report_path, 'w', encoding='utf-8') as f: f.write("\n".join(report_lines))
    print(f"Detailed summary report saved to '{report_path}'")

# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    EVAL_DATA_DIR = '../ablation_qa_outputs'
    REPORTS_DIR = 'ablation_evaluation_reports'
    RAW_DATA_SUBDIR = 'raw_data'
    os.makedirs(REPORTS_DIR, exist_ok=True)
    raw_data_dir = os.path.join(REPORTS_DIR, RAW_DATA_SUBDIR); os.makedirs(raw_data_dir, exist_ok=True)
    
    grader_llm = None
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not set in .env file.")
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using the standard, reliable model name
        grader_llm = genai.GenerativeModel('gemini-2.5-pro')
        print("LLM Grader initialized with 'gemini-2.5-pro'.")
    except Exception as e: print(f"Could not initialize LLM Grader. It will be skipped. Error: {e}"); grader_llm = None

    json_files = sorted(glob.glob(os.path.join(EVAL_DATA_DIR, '*.json')))
    if not json_files: print(f"Error: No JSON files found in '{EVAL_DATA_DIR}'."); exit()

    all_configs_summary = []
    for filepath in json_files:
        experiment_name = os.path.splitext(os.path.basename(filepath))[0]
        print("\n" + "#"*80 + f"\n# EVALUATING CONFIGURATION: {experiment_name}\n" + "#"*80)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: eval_data = json.load(f)
            print(f"Successfully loaded {len(eval_data)} items from '{filepath}'")
        except Exception as e: print(f"Failed to load or parse '{filepath}': {e}"); continue
        
        quantitative_results = run_quantitative_evaluation(eval_data)
        grader_results = run_llm_grader_evaluation(eval_data, grader_llm) if grader_llm else []
            
        quant_path = os.path.join(raw_data_dir, f'quantitative_report_{experiment_name}.json')
        with open(quant_path, 'w', encoding='utf-8') as f: json.dump(quantitative_results, f, indent=2)
        print(f"\nDetailed quantitative results saved to '{quant_path}'")
        if grader_results:
            grader_df = pd.DataFrame(grader_results); grader_csv_path = os.path.join(raw_data_dir, f'llm_grader_report_{experiment_name}.csv')
            grader_df.to_csv(grader_csv_path, index=False); print(f"LLM Grader results saved to '{grader_csv_path}'")

        if quantitative_results:
            generate_summary_report(quantitative_results, grader_results, experiment_name, REPORTS_DIR)
            q_scores = [res['scores'] for res in quantitative_results]
            
            summary = {
                'Configuration': experiment_name,
                'BLEU': np.mean([s['BLEU'] for s in q_scores]), 'ROUGE-1': np.mean([s['ROUGE']['rouge-1'] for s in q_scores]), 'ROUGE-2': np.mean([s['ROUGE']['rouge-2'] for s in q_scores]), 'ROUGE-L': np.mean([s['ROUGE']['rouge-L'] for s in q_scores]), 'TF-IDF Cosine Sim': np.mean([s['TF-IDF_Cosine_Sim'] for s in q_scores]),
                'BERTScore P': np.mean([s['BERTScore']['precision'] for s in q_scores]), 'BERTScore R': np.mean([s['BERTScore']['recall'] for s in q_scores]), 'BERTScore F1': np.mean([s['BERTScore']['f1'] for s in q_scores]),
                'ST Cosine Sim': np.mean([s['ST_Cosine_Sim'] for s in q_scores]), 'ST Dot Product': np.mean([s['ST_Dot_Product'] for s in q_scores]), 'SAS Score': np.mean([s['SAS_Score'] for s in q_scores]),
                'LLM Correctness': np.nan, 'LLM Completeness': np.nan, 'LLM Conciseness': np.nan
            }
            if grader_results:
                summary['LLM Correctness'] = np.mean([res['scores']['correctness'] for res in grader_results if 'scores' in res])
                summary['LLM Completeness'] = np.mean([res['scores']['completeness'] for res in grader_results if 'scores' in res])
                summary['LLM Conciseness'] = np.mean([res['scores']['conciseness'] for res in grader_results if 'scores' in res])
            all_configs_summary.append(summary)

    if all_configs_summary:
        print("\n\n" + "#"*80 + "\n# FINAL ABLATION STUDY COMPARISON\n" + "#"*80)
        summary_df = pd.DataFrame(all_configs_summary).set_index('Configuration')
        column_order = [ 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'TF-IDF Cosine Sim', 'BERTScore P', 'BERTScore R', 'BERTScore F1', 'ST Cosine Sim', 'ST Dot Product', 'SAS Score', 'LLM Correctness', 'LLM Completeness', 'LLM Conciseness' ]
        summary_df = summary_df[[col for col in column_order if col in summary_df.columns]]
        print(summary_df.round(4).to_markdown())
        generate_comparison_heatmap(all_configs_summary, REPORTS_DIR)