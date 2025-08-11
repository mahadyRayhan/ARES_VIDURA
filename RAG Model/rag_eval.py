import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json
import numpy as np # For calculating averages

# --- NLTK Resource Checks ---
nltk.download('punkt_tab')

# --- Evaluation Functions ---
def compute_bleu(reference, candidate):
    """Computes BLEU score between a reference and a candidate string."""
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

def compute_rouge(reference, candidate):
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge-1": scores['rouge1'].fmeasure,
        "rouge-2": scores['rouge2'].fmeasure,
        "rouge-L": scores['rougeL'].fmeasure
    }

def compute_cosine_similarity(reference, candidate):
    """Computes Cosine Similarity between a reference and a candidate string."""
    vectorizer = TfidfVectorizer()
    try:
        # Ensure inputs are not empty strings, which can cause TfidfVectorizer to raise an error
        if not reference.strip() or not candidate.strip():
            print(f"Warning: Cosine similarity called with empty string. Reference: '{reference[:50]}...', Candidate: '{candidate[:50]}...'. Returning 0.0")
            return 0.0
        vectors = vectorizer.fit_transform([reference, candidate])
        cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])
        return cos_sim[0][0]
    except ValueError as e:
        # This can happen if strings have no common terms after vectorization or are empty
        print(f"Warning: Cosine similarity ValueError. Reference: '{reference[:50]}...', Candidate: '{candidate[:50]}...'. Error: {e}. Returning 0.0")
        return 0.0

def compute_f1(reference, candidate):
    """Computes token-level F1 score between reference and candidate."""
    ref_tokens = set(nltk.word_tokenize(reference.lower()))
    cand_tokens = set(nltk.word_tokenize(candidate.lower()))
    common = ref_tokens & cand_tokens

    if not common:
        return 0.0

    precision = len(common) / len(cand_tokens) if cand_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_answers(reference_answer_text, generated_answer_text, original_question_text=""):
    """
    Evaluates a reference answer against a generated answer.
    original_question_text is used for context in warnings.
    """
    # Ensure answer inputs are strings
    if not isinstance(reference_answer_text, str) or not isinstance(generated_answer_text, str):
        print(f"Warning: Non-string answer provided for question '{original_question_text}'. Skipping evaluation for this item.")
        return {
            "bleu_answer": 0.0,
            "rouge_answer": {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-L": 0.0},
            "cosine_similarity_answer": 0.0,
            "f1_answer": 0.0
        }

    bleu_score = compute_bleu(reference_answer_text, generated_answer_text)
    rouge_scores = compute_rouge(reference_answer_text, generated_answer_text)
    cosine_sim_score = compute_cosine_similarity(reference_answer_text, generated_answer_text)
    f1_score = compute_f1(reference_answer_text, generated_answer_text)

    return {
        "bleu_answer": bleu_score,
        "rouge_answer": rouge_scores, # This is a dict: {"rouge-1": ..., "rouge-2": ..., "rouge-L": ...}
        "cosine_similarity_answer": cosine_sim_score,
        "f1_answer": f1_score
    }

# --- Main Script ---
if __name__ == "__main__":
    file_path = 'question_set/novice.json' # Or your qa.json
    data_to_evaluate = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_to_evaluate = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")

    all_evaluations_data = [] 
    # Lists to store scores for aggregation
    bleu_scores_list = []
    rouge1_scores_list = []
    rouge2_scores_list = []
    rougeL_scores_list = []
    cosine_scores_list = []
    f1_scores_list = []

    if data_to_evaluate:
        for item_idx, item in enumerate(data_to_evaluate):
            if not all(k in item for k in ["question", "answer", "LLM"]):
                print(f"Skipping item {item_idx+1} due to missing keys: {item.get('question', 'Unknown Question')}")
                continue

            reference_answer = item["answer"]
            llm_generated_answer = item["LLM"]
            original_question = item["question"] # Kept for context

            try:
                # Call the new evaluation function that only processes answers
                scores = evaluate_answers(reference_answer, llm_generated_answer, original_question)
                
                all_evaluations_data.append({
                    "item_index": item_idx + 1,
                    "original_question": original_question,
                    "reference_answer": reference_answer,
                    "llm_answer": llm_generated_answer,
                    "scores": scores 
                })

                # Store scores for aggregation
                bleu_scores_list.append(scores["bleu_answer"])
                current_rouge_scores = scores["rouge_answer"]
                if isinstance(current_rouge_scores, dict):
                    rouge1_scores_list.append(current_rouge_scores.get("rouge-1", 0.0))
                    rouge2_scores_list.append(current_rouge_scores.get("rouge-2", 0.0))
                    rougeL_scores_list.append(current_rouge_scores.get("rouge-L", 0.0))
                else:
                    rouge1_scores_list.append(0.0)
                    rouge2_scores_list.append(0.0)
                    rougeL_scores_list.append(0.0)
                cosine_scores_list.append(scores["cosine_similarity_answer"])
                f1_scores_list.append(scores.get("f1_answer", 0.0))


            except Exception as e:
                print(f"Error evaluating item {item_idx+1} ('{original_question}'): {e}")
                # Append placeholder scores to maintain list lengths for averaging if an error occurs
                bleu_scores_list.append(0.0)
                rouge1_scores_list.append(0.0)
                rouge2_scores_list.append(0.0)
                rougeL_scores_list.append(0.0)
                cosine_scores_list.append(0.0)
                f1_scores_list.append(0.0)

    # --- Print Individual Results (Optional) ---
    if not all_evaluations_data:
        print("No data was successfully evaluated.")
    else:
        # # To print individual scores, uncomment and adapt this section:
        # for eval_data in all_evaluations_data:
        #     print(f"--- Evaluation for Item {eval_data['item_index']} (Question: {eval_data['original_question']}) ---")
        #     print("Scores (Answer vs LLM):")
        #     print(f"  BLEU: {eval_data['scores']['bleu_answer']:.4f}")
        #     rouge_details = eval_data['scores']['rouge_answer']
        #     print(f"  ROUGE-1: {rouge_details.get('rouge-1', 0.0):.4f}")
        #     print(f"  ROUGE-2: {rouge_details.get('rouge-2', 0.0):.4f}")
        #     print(f"  ROUGE-L: {rouge_details.get('rouge-L', 0.0):.4f}")
        #     print(f"  Cosine Similarity: {eval_data['scores']['cosine_similarity_answer']:.4f}")
        #     print("\n")
        pass

    # --- Calculate and Print Aggregated Scores for Answers ---
    if bleu_scores_list: # Check if there are any scores to aggregate
        avg_bleu = np.mean(bleu_scores_list)
        avg_rouge1 = np.mean(rouge1_scores_list)
        avg_rouge2 = np.mean(rouge2_scores_list)
        avg_rougeL = np.mean(rougeL_scores_list)
        avg_cosine = np.mean(cosine_scores_list)
        avg_f1 = np.mean(f1_scores_list)


        # Helper function to format the list of scores
        def format_score_list(score_list):
            return "[" + ", ".join([f"{score:.4f}" for score in score_list]) + "]"

        print("\n--- Aggregated Evaluation Scores (Reference Answer vs LLM Answer) ---")
        print(f"Average BLEU Score:           {avg_bleu:.4f} {format_score_list(bleu_scores_list)}")
        print(f"Average ROUGE-1 F1 Score:   {avg_rouge1:.4f} {format_score_list(rouge1_scores_list)}")
        print(f"Average ROUGE-2 F1 Score:   {avg_rouge2:.4f} {format_score_list(rouge2_scores_list)}")
        print(f"Average ROUGE-L F1 Score:   {avg_rougeL:.4f} {format_score_list(rougeL_scores_list)}")
        print(f"Average Cosine Similarity:  {avg_cosine:.4f} {format_score_list(cosine_scores_list)}")
        print(f"Average F1 Score:             {avg_f1:.4f} {format_score_list(f1_scores_list)}")

    else:
        print("\nNo scores were available to calculate aggregated results.")

