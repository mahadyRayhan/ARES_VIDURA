import logging
import pandas as pd
import os
import re
import json
from typing import Dict, Any, Optional, List


# Assuming llm_interface.py contains LLMInterface class
from llm_interface import LLMInterface
import config # For defaults

# --- Feedback Management ---

def load_feedback_history(feedback_db_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load feedback history from CSV if it exists."""
    if os.path.exists(feedback_db_path):
         try:
             feedback_df = pd.read_csv(feedback_db_path)
             # Handle potential NaN values that can cause issues with to_dict
             feedback_df = feedback_df.fillna('')
             # Basic type conversion attempt (add more complex parsing if needed)
             # Example: Convert ratings back to numeric if possible
             if 'user_rating' in feedback_df.columns:
                 feedback_df['user_rating'] = pd.to_numeric(feedback_df['user_rating'], errors='coerce').fillna(-1).astype(int) # Use -1 for parse failure

             history = feedback_df.to_dict('records')
             logger.info(f"Loaded {len(history)} feedback records from {feedback_db_path}.")
             return history
         except Exception as e:
             logger.warning(f"Could not load feedback history from {feedback_db_path}: {e}")
             return []
    else:
         logger.info(f"Feedback history file not found at {feedback_db_path}. Starting fresh.")
         return []

def save_feedback_history(feedback_history: List[Dict[str, Any]], feedback_db_path: str, logger: logging.Logger):
    """Save the current feedback history to CSV."""
    if not feedback_history:
        logger.debug("No feedback history to save.")
        return

    try:
         dir_name = os.path.dirname(feedback_db_path)
         if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

         feedback_df = pd.DataFrame(feedback_history)
         # Ensure complex types are strings for CSV
         for col in feedback_df.columns:
              if feedback_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                   feedback_df[col] = feedback_df[col].astype(str)

         feedback_df.to_csv(feedback_db_path, index=False)
         logger.info(f"Saved {len(feedback_history)} feedback records to {feedback_db_path}.")
    except Exception as e:
        logger.error(f"Failed to save feedback history to {feedback_db_path}: {e}")


def record_feedback(feedback_history: List[Dict[str, Any]], # Pass history list
                    feedback_db_path: str, # Pass path for saving
                    logger: logging.Logger, # Pass logger
                    query: str,
                    answer: str,
                    hop_count: int,
                    final_context: str, # Consider truncating if too long
                    reasoning_trace: List[str],
                    search_query_history: List[str],
                    user_rating: Optional[int] = None,
                    user_comment: Optional[str] = None,
                    evaluation_metrics: Optional[Dict[str, Any]] = None,
                    confidence_score: Optional[float] = None, # Add confidence
                    debug_info: Optional[Dict[str, Any]] = None): # Add debug info
    """Record user feedback and internal metrics, append to history, and save."""

    feedback_record = {
         "timestamp": pd.Timestamp.now().isoformat(),
         "query": query,
         "answer": answer,
         "hop_count": hop_count,
         "final_context_length": len(final_context),
         "confidence_score": confidence_score,
         "reasoning_trace": " -> ".join(reasoning_trace),
         "search_queries": " | ".join(search_query_history),
         "user_rating": user_rating,
         "user_comment": user_comment,
         # Flatten evaluation metrics and debug info into the main record
         **(evaluation_metrics or {}),
         **(debug_info or {})
     }
    # Ensure all values are suitable for CSV
    for key, value in feedback_record.items():
        if isinstance(value, (dict, list)):
             feedback_record[key] = str(value)
        elif value is None:
             feedback_record[key] = '' # Use empty string for None

    feedback_history.append(feedback_record)
    logger.info(f"Recorded feedback for query: '{query[:50]}...' (User Rating: {user_rating})")
    save_feedback_history(feedback_history, feedback_db_path, logger) # Save updated history

# --- LLM-Based Evaluation ---

def _parse_json_response(response: str, keys: List[str], logger: logging.Logger) -> Optional[Dict]:
    """Tries to parse JSON from LLM response, falling back to regex."""
    try:
        # Handle potential markdown code blocks ```json ... ```
        response_clean = re.sub(r"```json\s*(.*?)\s*```", r"\1", response, flags=re.DOTALL | re.IGNORECASE)
        response_clean = response_clean.strip()
        parsed_json = json.loads(response_clean)
        # Validate required keys exist
        if all(k in parsed_json for k in keys):
             # Attempt basic type conversion for ratings
             for k in keys:
                 if 'rating' in k:
                     try:
                         parsed_json[k] = int(parsed_json[k])
                     except (ValueError, TypeError):
                          logger.warning(f"Could not convert rating '{k}' value '{parsed_json[k]}' to int.")
                          parsed_json[k] = None # Mark as None if conversion fails
                 else:
                     parsed_json[k] = str(parsed_json[k]) # Ensure explanations are strings
             return parsed_json
        else:
            logger.warning(f"Parsed JSON missing required keys ({keys}). Response: {response_clean}")
            return None
    except (json.JSONDecodeError, TypeError):
        logger.debug(f"Could not parse JSON, trying regex. Response: {response}")
        # Build a dynamic regex based on expected keys
        patterns = []
        for key in keys:
            # Handle rating (numeric) vs explanation (string) differently
            if 'rating' in key:
                 patterns.append(rf'"{key}"\s*:\s*(\d+)')
            else:
                 patterns.append(rf'"{key}"\s*:\s*"([^"]*)"')
        regex_pattern = r"{\s*" + r"\s*,\s*".join(patterns) + r"\s*}"
        match = re.search(regex_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            parsed_data = {}
            for i, key in enumerate(keys):
                value = match.group(i + 1).strip()
                if 'rating' in key:
                    try:
                         parsed_data[key] = int(value)
                    except ValueError:
                         logger.warning(f"Regex extracted non-integer rating for {key}: {value}")
                         parsed_data[key] = None
                else:
                    parsed_data[key] = value
            return parsed_data
        else:
            logger.warning(f"Could not parse rating using JSON or regex. Response: {response}")
            return None


def evaluate_response(question: str, answer: str, retrieved_context_str: str,
                        llm_interface: LLMInterface, logger: logging.Logger,
                        ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the quality of the generated answer using LLM."""
    logger.info("Evaluating generated scientific response...")
    metrics = {}
    context_limit = 6000 # Limit context sent for evaluation

    # --- Relevance ---
    try:
        relevance_keys = ["relevance_rating", "relevance_explanation"]
        relevance_prompt = f"""Rate the relevance of the Answer to the scientific Question on a scale of 1 (Not relevant) to 5 (Highly relevant and directly addresses the scientific query).

        Question: "{question}"
        Answer: "{answer}"

        Output format (JSON): {{"{relevance_keys[0]}": [1-5], "{relevance_keys[1]}": "[Brief explanation]"}}
        """
        relevance_response = llm_interface.generate_response(relevance_prompt)
        parsed = _parse_json_response(relevance_response, relevance_keys, logger)
        if parsed:
            metrics.update(parsed)
        else:
            metrics[relevance_keys[0]] = None
            metrics[relevance_keys[1]] = f"Parsing Failed: {relevance_response}"
    except Exception as e:
        logger.error(f"Error during relevance evaluation: {e}")
        metrics['relevance_rating'] = None; metrics['relevance_explanation'] = f"Error: {e}"

    # --- Faithfulness ---
    try:
        faithfulness_keys = ["faithfulness_rating", "faithfulness_explanation"]
        faithfulness_prompt = f"""Rate how faithful the scientific Answer is to the provided Context on a scale of 1 (Contradicts/external info) to 5 (Fully supported).

        Context:
        ---
        {retrieved_context_str[:context_limit]}...
        ---
        Answer: "{answer}"

        Output format (JSON): {{"{faithfulness_keys[0]}": [1-5], "{faithfulness_keys[1]}": "[Brief explanation]"}}
        """
        faithfulness_response = llm_interface.generate_response(faithfulness_prompt)
        parsed = _parse_json_response(faithfulness_response, faithfulness_keys, logger)
        if parsed:
            metrics.update(parsed)
        else:
            metrics[faithfulness_keys[0]] = None
            metrics[faithfulness_keys[1]] = f"Parsing Failed: {faithfulness_response}"
    except Exception as e:
        logger.error(f"Error during faithfulness evaluation: {e}")
        metrics['faithfulness_rating'] = None; metrics['faithfulness_explanation'] = f"Error: {e}"

    # --- Ground Truth Comparison ---
    if ground_truth:
        try:
            gt_keys = ["ground_truth_similarity_rating", "ground_truth_similarity_explanation"]
            gt_prompt = f"""Compare the Generated Answer to the Ground Truth Answer for the scientific Question. Rate similarity/correctness 1 (Very different/Incorrect) to 5 (Highly similar/Correct).

            Question: "{question}"
            Ground Truth Answer: "{ground_truth}"
            Generated Answer: "{answer}"

            Output format (JSON): {{"{gt_keys[0]}": [1-5], "{gt_keys[1]}": "[Brief explanation]"}}
            """
            gt_response = llm_interface.generate_response(gt_prompt)
            parsed = _parse_json_response(gt_response, gt_keys, logger)
            if parsed:
                metrics.update(parsed)
            else:
                metrics[gt_keys[0]] = None
                metrics[gt_keys[1]] = f"Parsing Failed: {gt_response}"
        except Exception as e:
            logger.error(f"Error during ground truth comparison: {e}")
            metrics['ground_truth_similarity_rating'] = None; metrics['ground_truth_similarity_explanation'] = f"Error: {e}"

    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

def assess_answer_confidence(answer: str, context: str, query: str,
                             llm_interface: LLMInterface, logger: logging.Logger) -> Optional[float]:
    """Estimate confidence (0.0 to 1.0) in the answer based on context support."""
    logger.debug("Assessing answer confidence...")
    context_limit = 6000 # Limit context sent
    prompt = f"""Assess how well the scientific Answer is supported *solely* by the provided Context in relation to the Query.
    Provide a confidence score between 0.0 (no support/contradicted) and 1.0 (fully and explicitly supported).

    Query: {query}
    Context:
    ---
    {context[:context_limit]}...
    ---
    Answer: {answer}

    Confidence Score (Return ONLY the numerical score between 0.0 and 1.0, e.g., 0.85):"""

    response = llm_interface.generate_response(prompt)

    if response.startswith("LLM_ERROR"):
        logger.warning(f"Confidence assessment failed due to LLM error: {response}")
        return None

    try:
        match = re.search(r"(\d(?:[.,]\d+)?)", response)
        if match:
            confidence_str = match.group(1).replace(',', '.')
            confidence = float(confidence_str)
            confidence = max(0.0, min(1.0, confidence))
            logger.debug(f"Assessed Confidence Score: {confidence:.2f}")
            return confidence
        else:
            logger.warning(f"Could not parse confidence score from response: '{response}'. Returning moderate confidence.")
            return 0.5
    except ValueError:
        logger.warning(f"Could not convert confidence score response to float: '{response}'. Returning moderate confidence.")
        return 0.5
    except Exception as e:
        logger.error(f"Error parsing confidence score: {e}")
        return None
