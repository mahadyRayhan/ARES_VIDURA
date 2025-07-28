import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np # For calculating averages
import re # For cleaning text
# Ashish test

# --- NLTK Resource Checks ---
nltk.download('punkt')
# The original 'punkt_tab' is not a standard NLTK resource.
# 'punkt' is the correct one for sentence tokenization and word tokenization.

# --- Text Cleaning Function ---
def clean_llm_answer(text):
    """Cleans common LLM formatting artifacts from text."""
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    # Remove markdown-like formatting (bold, italics, etc.)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*(.*?)\*', r'\1', text)    # Italics *text*
    text = re.sub(r'__(.*?)__', r'\1', text)  # Underline __text__ (less common for LLMs)
    text = re.sub(r'_(.*?)_', r'\1', text)    # Italics _text_

    # Remove common conversational or instructional prefixes/suffixes if observed
    # Example: "Here's the answer:", "Based on the context:", "Final Answer:"
    # Be cautious with these, as they might remove legitimate parts of an answer.
    # text = re.sub(r"^\s*(Here's the answer:|Final Answer:|Answer:|Okay, here you go:)\s*", "", text, flags=re.IGNORECASE)

    # Replace multiple newlines with a single space or newline
    text = re.sub(r'\n\s*\n', '\n', text) # Collapses multiple newlines with spaces in between to one newline
    text = re.sub(r'\n+', ' ', text)       # Replace newlines with spaces (or use '\n' to keep single newlines)

    # Remove leading/trailing whitespace and excessive internal spaces
    text = text.strip()
    text = re.sub(r'\s{2,}', ' ', text) # Replace 2 or more spaces with a single space

    return text

# --- Evaluation Functions (compute_bleu, compute_rouge, compute_cosine_similarity, evaluate_answers remain the same) ---
def compute_bleu(reference, candidate):
    """Computes BLEU score between a reference and a candidate string."""
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothing = SmoothingFunction().method1 # Use a standard smoothing method
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
        if not reference.strip() or not candidate.strip():
            # print(f"Warning: Cosine similarity called with empty string. Ref: '{reference[:50]}...', Cand: '{candidate[:50]}...'. Ret 0.0")
            return 0.0
        vectors = vectorizer.fit_transform([reference, candidate])
        cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])
        return cos_sim[0][0]
    except ValueError as e:
        # print(f"Warning: Cosine similarity ValueError. Ref: '{reference[:50]}...', Cand: '{candidate[:50]}...'. Err: {e}. Ret 0.0")
        return 0.0

def evaluate_answers(reference_answer_text, generated_answer_text, original_question_text=""):
    """
    Evaluates a reference answer against a generated answer.
    original_question_text is used for context in warnings.
    """
    if not isinstance(reference_answer_text, str) or not isinstance(generated_answer_text, str):
        print(f"Warning: Non-string answer provided for question '{original_question_text}'. Skipping evaluation.")
        return {
            "bleu_answer": 0.0,
            "rouge_answer": {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-L": 0.0},
            "cosine_similarity_answer": 0.0
        }

    bleu_score = compute_bleu(reference_answer_text, generated_answer_text)
    rouge_scores = compute_rouge(reference_answer_text, generated_answer_text)
    cosine_sim_score = compute_cosine_similarity(reference_answer_text, generated_answer_text)

    return {
        "bleu_answer": bleu_score,
        "rouge_answer": rouge_scores,
        "cosine_similarity_answer": cosine_sim_score
    }

# --- Main Script ---
if __name__ == "__main__":
    file_path = 'question_set/advanced.json' # Or your qa.json
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
    bleu_scores_list = []
    rouge1_scores_list = []
    rouge2_scores_list = []
    rougeL_scores_list = []
    cosine_scores_list = []

    if data_to_evaluate:
        for item_idx, item in enumerate(data_to_evaluate):
            if not all(k in item for k in ["question", "answer", "LLM"]):
                print(f"Skipping item {item_idx+1} due to missing keys: {item.get('question', 'Unknown Question')}")
                continue

            reference_answer = item.get("answer", "") # Use .get for safety
            llm_raw_answer = item.get("LLM", "")      # Use .get for safety
            original_question = item.get("question", "Unknown Question")

            # --- Apply Cleaning ---
            llm_cleaned_answer = clean_llm_answer(llm_raw_answer)
            # You might also want to clean the reference_answer if it could have similar artifacts
            # reference_cleaned_answer = clean_llm_answer(reference_answer)
            # For now, we'll assume reference_answer is already clean or should be treated as is.

            # Optional: Print to see the effect of cleaning
            if llm_raw_answer != llm_cleaned_answer and item_idx < 5: # Print for first 5 items if changed
                print(f"\n--- Cleaning Applied to LLM Answer for Item {item_idx+1} ---")
                print(f"Original LLM: '{llm_raw_answer[:100]}...'")
                print(f"Cleaned LLM : '{llm_cleaned_answer[:100]}...'")


            try:
                # Use cleaned answer for evaluation
                scores = evaluate_answers(reference_answer, llm_cleaned_answer, original_question)

                all_evaluations_data.append({
                    "item_index": item_idx + 1,
                    "original_question": original_question,
                    "reference_answer": reference_answer,
                    "llm_raw_answer": llm_raw_answer, # Store raw for inspection
                    "llm_cleaned_answer": llm_cleaned_answer, # Store cleaned
                    "scores": scores
                })

                bleu_scores_list.append(scores["bleu_answer"])
                current_rouge_scores = scores["rouge_answer"]
                if isinstance(current_rouge_scores, dict):
                    rouge1_scores_list.append(current_rouge_scores.get("rouge-1", 0.0))
                    rouge2_scores_list.append(current_rouge_scores.get("rouge-2", 0.0))
                    rougeL_scores_list.append(current_rouge_scores.get("rouge-L", 0.0))
                else: # Should not happen if evaluate_answers returns consistent dict
                    rouge1_scores_list.append(0.0)
                    rouge2_scores_list.append(0.0)
                    rougeL_scores_list.append(0.0)
                cosine_scores_list.append(scores["cosine_similarity_answer"])

            except Exception as e:
                print(f"Error evaluating item {item_idx+1} ('{original_question}'): {e}")
                bleu_scores_list.append(0.0)
                rouge1_scores_list.append(0.0)
                rouge2_scores_list.append(0.0)
                rougeL_scores_list.append(0.0)
                cosine_scores_list.append(0.0)

    # --- Print Individual Results (Optional, modified to show cleaned answer) ---
    if not all_evaluations_data:
        print("No data was successfully evaluated.")
    else:
        # To print individual scores, uncomment and adapt this section:
        # for eval_data in all_evaluations_data[:5]: # Print first 5 for example
        #     print(f"--- Evaluation for Item {eval_data['item_index']} (Question: {eval_data['original_question']}) ---")
        #     # print(f"Reference Answer: {eval_data['reference_answer']}")
        #     # print(f"LLM Raw Answer: {eval_data['llm_raw_answer']}")
        #     # print(f"LLM Cleaned Answer: {eval_data['llm_cleaned_answer']}")
        #     print("Scores (Answer vs Cleaned LLM):")
        #     print(f"  BLEU: {eval_data['scores']['bleu_answer']:.4f}")
        #     rouge_details = eval_data['scores']['rouge_answer']
        #     print(f"  ROUGE-1: {rouge_details.get('rouge-1', 0.0):.4f}")
        #     print(f"  ROUGE-2: {rouge_details.get('rouge-2', 0.0):.4f}")
        #     print(f"  ROUGE-L: {rouge_details.get('rouge-L', 0.0):.4f}")
        #     print(f"  Cosine Similarity: {eval_data['scores']['cosine_similarity_answer']:.4f}")
        #     print("\n")
        pass

    # --- Calculate and Print Aggregated Scores for Answers ---
    if bleu_scores_list:
        avg_bleu = np.mean(bleu_scores_list)
        avg_rouge1 = np.mean(rouge1_scores_list)
        avg_rouge2 = np.mean(rouge2_scores_list)
        avg_rougeL = np.mean(rougeL_scores_list)
        avg_cosine = np.mean(cosine_scores_list)

        def format_score_list(score_list):
            return "[" + ", ".join([f"{score:.4f}" for score in score_list]) + "]"

        print("\n--- Aggregated Evaluation Scores (Reference Answer vs CLEANED LLM Answer) ---")
        print(f"Average BLEU Score:           {avg_bleu:.4f} {format_score_list(bleu_scores_list)}")
        print(f"Average ROUGE-1 F1 Score:   {avg_rouge1:.4f} {format_score_list(rouge1_scores_list)}")
        print(f"Average ROUGE-2 F1 Score:   {avg_rouge2:.4f} {format_score_list(rouge2_scores_list)}")
        print(f"Average ROUGE-L F1 Score:   {avg_rougeL:.4f} {format_score_list(rougeL_scores_list)}")
        print(f"Average Cosine Similarity:  {avg_cosine:.4f} {format_score_list(cosine_scores_list)}")
    else:
        print("\nNo scores were available to calculate aggregated results.")