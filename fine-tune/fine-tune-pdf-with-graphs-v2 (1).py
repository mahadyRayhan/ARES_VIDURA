#!/usr/bin/env python
# coding: utf-8

# --- STEP 0: IMPORTS ---
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import transformers
# REMOVED from transformers import EarlyStoppingCallback
import re, random
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import PyPDF2
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import evaluate
from sentence_transformers import SentenceTransformer, util

# --- STEP 1: CONFIGURATION AND INITIALIZATION ---
print("--- STEP 1: CONFIGURATION AND INITIALIZATION ---")
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY_HOME")
google_api_key = os.getenv("GOOGLE_API_KEY")
# login(token=hf_api_key) # Uncomment if needed
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
pdf_folder_path = "DataSets/"
scratch_dir = f"/tmp/{os.getenv('USER', 'user')}/cnt_finetuning"
if not os.path.exists(scratch_dir): os.makedirs(scratch_dir)
print(f"Using scratch directory for large files: {scratch_dir}")
if not os.path.exists('visualizations'): os.makedirs('visualizations')
if not os.path.exists('evaluation_outputs'): os.makedirs('evaluation_outputs')
dataset_v1_path = "dataset_gemini_v1.json"
model_v1_output_dir = os.path.join(scratch_dir, "cnt-finetune-v1-on-gemini-data")
dataset_v2_path = "dataset_mistral_v2.json"
model_v2_output_dir = os.path.join(scratch_dir, "cnt-finetune-v2-on-mistral-data")
evaluation_data_path = "model-comp-adv.json"
NUM_SAMPLES_TO_GENERATE = 500
MAX_TRAINING_STEPS = 300

# --- STEP 2 & 3: PDF Processing and Helper Functions ---
# ... (pdf_to_text_from_folder, generate_response_gemini are unchanged) ...
def pdf_to_text_from_folder(folder_path, skip_start_pages=5, skip_last_pages=0, header_lines=2, footer_lines=1):
    combined_text = ""
    if not os.path.isdir(folder_path): return ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages[max(0, skip_start_pages -1):len(pdf_reader.pages)-skip_last_pages]:
                        page_text = page.extract_text()
                        if not page_text: continue
                        lines = page_text.splitlines(True)[header_lines:-footer_lines if footer_lines > 0 else None]
                        combined_text += "".join(lines)
            except Exception as e: print(f"Could not process {filename}. Error: {e}")
    return combined_text
def generate_response_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response.parts else None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None
def generate_dataset(generation_function, num_samples, output_filename, **kwargs):
    if os.path.exists(output_filename):
        print(f"Dataset '{output_filename}' already exists. Skipping generation.")
        with open(output_filename, "r") as f: return json.load(f)
    data = []
    sample_char_lens = [1024, 768, 512]
    if len(raw_text) == 0:
        print("Error: Raw text is empty. Cannot generate dataset.")
        return []
    iterations_per_len = num_samples // len(sample_char_lens) if len(sample_char_lens) > 0 else 0
    for char_len in sample_char_lens:
        for i in range(iterations_per_len):
            print(f"Generating sample {len(data) + 1}/{num_samples} (length: {char_len})...")
            random_start = random.randint(0, len(raw_text) - char_len)
            doc_sample = " ".join(raw_text[random_start:random_start + char_len].split()).replace("\u2010", "-")
            qa_pair = generation_function(doc_sample, **kwargs)
            if qa_pair: data.append(qa_pair)
    with open(output_filename, "w") as f: json.dump(data, f, indent=4)
    print(f"Successfully generated {len(data)} samples and saved to '{output_filename}'.")
    return data

# ... (train_model and prepare_and_save_splits are unchanged from the simplified version) ...
def train_model(train_dataset, eval_dataset, base_model_id, output_dir, max_steps=100):
    print(f"\n--- Starting Training for model: {output_dir} ---")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(sample):
        user_content = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}"
        assistant_content = f"### Response:\n{sample['output']}"
        text = f"<s>[INST] {user_content.strip()} [/INST] {assistant_content.strip()} </s>"
        result = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result
    tokenized_train = train_dataset.map(tokenize_function, remove_columns=list(train_dataset.features))
    tokenized_eval = eval_dataset.map(tokenize_function, remove_columns=list(eval_dataset.features))
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)
    trainer = transformers.Trainer(
        model=model, train_dataset=tokenized_train, eval_dataset=tokenized_eval,
        args=transformers.TrainingArguments(
            output_dir=output_dir, 
            warmup_steps=5, 
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=max_steps, 
            learning_rate=2.5e-5, 
            logging_steps=10, 
            bf16=True, optim="paged_adamw_8bit",
            save_strategy="steps", 
            do_eval=True, 
            eval_steps=20,
            save_steps=20,
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()
    final_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_path)
    print(f"--- Finished Training. Final model saved to: {final_path} ---")
    return final_path, trainer.state.log_history

def prepare_and_save_splits(dataset_list, prefix):
    random.shuffle(dataset_list)
    train_size = int(0.9 * len(dataset_list))
    train_data = Dataset.from_list(dataset_list[:train_size])
    eval_data = Dataset.from_list(dataset_list[train_size:])
    print(f"Split for '{prefix}': {len(train_data)} training samples, {len(eval_data)} evaluation samples.")
    return train_data, eval_data

# --- STEP 4-5: STAGES 1 & 2 (Training Loops) ---
print("\n--- STEP 2: LOAD AND PROCESS ALL PDFS FROM A FOLDER ---")
raw_text = pdf_to_text_from_folder(pdf_folder_path)
print(f"\nTotal characters extracted from all PDFs: {len(raw_text)}")

# *** NEW: Persona-driven data generation function for Gemini ***
def generate_qa_with_expert_persona_gemini(doc_sample):
    """Generates a Q&A pair using Gemini with an expert persona."""
    q_prompt = f"""You are an expert CNT researcher reviewing a document. Based on the following text snippet, formulate one insightful and specific technical question that an advanced colleague might ask. The question should probe a key concept within the text.

**Text Snippet:**
"{doc_sample}"

**Generated Question:**"""
    question = generate_response_gemini(q_prompt)
    if not question: return None

    a_prompt = f"""You are an expert CNT researcher answering a colleague's question. Use ONLY the provided context to write a comprehensive, well-structured answer. Synthesize the information rather than just extracting it. Do not add any preamble like 'Based on the text...'.

**Colleague's Question:**
"{question}"

**Context to Use:**
"{doc_sample}"

**Synthesized Answer:**"""
    answer = generate_response_gemini(a_prompt)
    if not answer: return None
    
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

print("\n\n--- STAGE 1: GENERATE DATA WITH GEMINI AND TRAIN MODEL V1 ---")
# MODIFIED: Call the new persona-driven function
dataset_v1_list = generate_dataset(generate_qa_with_expert_persona_gemini, NUM_SAMPLES_TO_GENERATE, dataset_v1_path)
model_v1_final_path, history_v1 = (None, None)
if dataset_v1_list:
    train_ds_v1, eval_ds_v1 = prepare_and_save_splits(dataset_v1_list, "v1")
    model_v1_final_path, history_v1 = train_model(train_ds_v1, eval_ds_v1, base_model_id, model_v1_output_dir, max_steps=MAX_TRAINING_STEPS)
else:
    print("Could not proceed with Stage 1 training as dataset is empty.")

# ... (generate_response_finetuned is unchanged) ...
def generate_response_finetuned(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    encodeds = tokenizer(text, return_tensors="pt").to('cuda')
    generated_ids = model.generate(**encodeds, max_new_tokens=1024, do_sample=True, temperature=0.5, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    new_tokens = generated_ids[0, encodeds['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# *** NEW: Persona-driven data generation function for the fine-tuned model ***
def generate_qa_with_expert_persona_finetuned(doc_sample, model, tokenizer):
    """Generates a Q&A pair using a fine-tuned model with an expert persona."""
    q_prompt = f"You are an expert CNT researcher. From the text below, generate one specific, technical question.\n\nText: \"{doc_sample}\""
    question = generate_response_finetuned(model, tokenizer, q_prompt)
    if not question: return None

    a_prompt = f"As an expert CNT researcher, use ONLY the provided text to answer the following question in a synthesized manner.\n\nQuestion: \"{question}\"\n\nContext: \"{doc_sample}\""
    answer = generate_response_finetuned(model, tokenizer, a_prompt)
    if not answer: return None
    
    return {"instruction": question.strip(), "input": "", "output": answer.strip()}

print("\n\n--- STAGE 2: GENERATE DATA WITH MODEL V1 AND TRAIN MODEL V2 ---")
model_v2_final_path, history_v2 = (None, None)
if model_v1_final_path and os.path.exists(model_v1_final_path):
    print(f"Loading fine-tuned model v1 for data generation from: {model_v1_final_path}")
    base_model_for_v1 = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
    ft_model_v1 = PeftModel.from_pretrained(base_model_for_v1, model_v1_final_path)
    tokenizer_v1 = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer_v1.pad_token = tokenizer_v1.eos_token
    # MODIFIED: Call the new persona-driven function for stage 2
    dataset_v2_list = generate_dataset(
        generate_qa_with_expert_persona_finetuned, NUM_SAMPLES_TO_GENERATE, dataset_v2_path, 
        model=ft_model_v1, tokenizer=tokenizer_v1
    )
    del ft_model_v1, base_model_for_v1
    torch.cuda.empty_cache()
    if dataset_v2_list:
        train_ds_v2, eval_ds_v2 = prepare_and_save_splits(dataset_v2_list, "v2")
        model_v2_final_path, history_v2 = train_model(train_ds_v2, eval_ds_v2, base_model_id, model_v2_output_dir, max_steps=MAX_TRAINING_STEPS)
    else:
        print("Could not proceed with Stage 2 training as dataset is empty.")
else:
    print("Skipping Stage 2 because Stage 1 model was not trained or path not found.")

# --- STEP 6: FINAL EVALUATION AND VISUALIZATION ---
# ... (This entire section is unchanged as it already handles detailed metrics) ...
print("\n\n--- STEP 6: FINAL EVALUATION AND VISUALIZATION ---")
# ... (plot_loss_curves, plot_score_comparison, format_rag_prompt, run_final_evaluation functions are unchanged) ...
def plot_loss_curves(log_history, model_version_str):
    if not log_history:
        print(f"No log history for model {model_version_str}, skipping loss plot.")
        return
    df = pd.DataFrame(log_history)
    if 'loss' not in df.columns:
        print(f"No training loss ('loss' key) found in log history for {model_version_str}. Skipping plot.")
        return
    train_df = df[df['loss'].notna()]
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['step'], train_df['loss'], label='Training Loss', marker='.')
    if 'eval_loss' in df.columns:
        eval_df = df[df['eval_loss'].notna()]
        if not eval_df.empty:
            plt.plot(eval_df['step'], eval_df['eval_loss'], label='Validation Loss', marker='o')
        else:
            print(f"Warning: 'eval_loss' column exists but contains no data for model {model_version_str}.")
    else:
        print(f"Warning: No 'eval_loss' found for model {model_version_str}. Plotting training loss only.")
    plt.title(f'Training and Validation Loss for Model {model_version_str.upper()}')
    plt.xlabel('Training Steps'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    filename = f'visualizations/loss_curve_{model_version_str}.png'
    plt.savefig(filename); print(f"Loss curve plot saved to {filename}"); plt.close()
def plot_score_comparison(results_df):
    questions = [f"Q{i+1}" for i in range(len(results_df))]
    metrics = ['Cosine Similarity', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    models = ['RAG', 'v1', 'v2']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5 * len(metrics)))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_keys = [f'{metric} ({model})' for model in models]
        scores = results_df[metric_keys].astype(float).values
        x = np.arange(len(questions))
        width = 0.25
        rects1 = ax.bar(x - width, scores[:, 0], width, label='Baseline RAG')
        rects2 = ax.bar(x, scores[:, 1], width, label='Fine-Tuned v1')
        rects3 = ax.bar(x + width, scores[:, 2], width, label='Fine-Tuned v2')
        ax.set_ylabel(metric)
        ax.set_title(f'Comparison of {metric} Scores')
        ax.set_xticks(x, questions)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')
        ax.bar_label(rects3, padding=3, fmt='%.2f')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = 'visualizations/all_metrics_comparison.png'
    plt.savefig(filename)
    print(f"All metrics comparison graph saved to {filename}")
    plt.close()
def format_rag_prompt(original_query, context_str):
    return f"""You are an expert researcher synthesizing a detailed technical analysis for an advanced colleague on Carbon Nanotubes (CNTs).
**Original Question:**\n"{original_query}"
**Accumulated Context:**\nUse ONLY the following context to answer the question. Each piece of context is preceded by its source information in the format [Source: Doc: '...', Page: ..., ChunkID: '...'].
--- START CONTEXT ---\n{context_str}\n--- END CONTEXT ---
**Your Task:**
1.  **Synthesize and Structure:** Write a comprehensive, well-structured answer to the "Original Question". Organize your answer into logical sections with clear, bolded headings (e.g., using Markdown like **Heading Title**). Explain concepts and their relationships.
2.  **Cite Everything:** You MUST cite every piece of information you use from the context. Place the citation at the end of the sentence or clause that uses the information.
3.  **Citation Format:** The citation format MUST BE: `(Source: Doc: '[Doc]', Page: [Page], ChunkID: '[ChunkID]')`.
4.  **Strictly Adhere to Context:** Base your entire answer ONLY on the "Accumulated Context".
5.  **Identify Gaps:** After providing the main answer, add a final section titled "**Missing Information**". In this section, describe what specific technical details are needed but are not present in the provided context.
---
**Final Synthesized Answer (Detailed, Structured, and with In-Text Citations):**"""
def run_final_evaluation(model_v1, model_v2, tokenizer, eval_data_path):
    if not os.path.exists(eval_data_path):
        print(f"Evaluation data file not found at {eval_data_path}. Skipping final evaluation.")
        return
    with open(eval_data_path, 'r', encoding='utf-8') as f: eval_data = json.load(f)
    rouge = evaluate.load('rouge')
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    print("Running final comparison with detailed metrics...")
    results = []
    for i, item in enumerate(eval_data):
        q, ground_truth, rag_response = item['question'], item['answer'], item['RAG']
        prompt_for_ft_models = format_rag_prompt(q, "No external context provided. Rely on your fine-tuned knowledge.")
        print(f"  Evaluating Question {i+1}...")
        v1_response = generate_response_finetuned(model_v1, tokenizer, prompt_for_ft_models)
        v2_response = generate_response_finetuned(model_v2, tokenizer, prompt_for_ft_models)
        with open(f"evaluation_outputs/q{i+1}_model_v1_response.txt", "w", encoding='utf-8') as f: f.write(v1_response)
        with open(f"evaluation_outputs/q{i+1}_model_v2_response.txt", "w", encoding='utf-8') as f: f.write(v2_response)
        score_rag = rouge.compute(predictions=[rag_response], references=[ground_truth])
        score_v1 = rouge.compute(predictions=[v1_response], references=[ground_truth])
        score_v2 = rouge.compute(predictions=[v2_response], references=[ground_truth])
        embeddings = similarity_model.encode([ground_truth, rag_response, v1_response, v2_response])
        cos_sim_rag = util.cos_sim(embeddings[0], embeddings[1]).item()
        cos_sim_v1 = util.cos_sim(embeddings[0], embeddings[2]).item()
        cos_sim_v2 = util.cos_sim(embeddings[0], embeddings[3]).item()
        results.append({
            "Question": q, "Ground Truth": ground_truth,
            "RAG Response": rag_response, "Cosine Similarity (RAG)": f"{cos_sim_rag:.4f}",
            "ROUGE-1 (RAG)": f"{score_rag['rouge1']:.4f}", "ROUGE-2 (RAG)": f"{score_rag['rouge2']:.4f}",
            "ROUGE-L (RAG)": f"{score_rag['rougeL']:.4f}", "FT Model v1 Response": v1_response,
            "Cosine Similarity (v1)": f"{cos_sim_v1:.4f}", "ROUGE-1 (v1)": f"{score_v1['rouge1']:.4f}",
            "ROUGE-2 (v1)": f"{score_v1['rouge2']:.4f}", "ROUGE-L (v1)": f"{score_v1['rougeL']:.4f}",
            "FT Model v2 Response": v2_response, "Cosine Similarity (v2)": f"{cos_sim_v2:.4f}",
            "ROUGE-1 (v2)": f"{score_v2['rouge1']:.4f}", "ROUGE-2 (v2)": f"{score_v2['rouge2']:.4f}",
            "ROUGE-L (v2)": f"{score_v2['rougeL']:.4f}"
        })
    df = pd.DataFrame(results)
    plot_score_comparison(df)
    styled_df = df.style.set_properties(**{'text-align': 'left', 'vertical-align': 'top', 'border': '1px solid black', 'padding': '8px'})\
                       .set_table_styles([{'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'left')]}])
    filename = 'visualizations/final_comparison_report.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(styled_df.to_html(escape=False))
    print(f"Final evaluation complete. Report and graph saved in 'visualizations/'.")

# --- Run Visualizations for Training ---
if history_v1: plot_loss_curves(history_v1, "v1")
if history_v2: plot_loss_curves(history_v2, "v2")

# --- STEP 7: RUN FINAL COMPARATIVE EVALUATION ---
if model_v1_final_path and model_v2_final_path:
    print("\n\n--- STEP 7: FINAL COMPARATIVE EVALUATION ---")
    base_model_for_inference = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading Model v1...")
    ft_model_v1 = PeftModel.from_pretrained(base_model_for_inference, model_v1_final_path)
    print("Loading Model v2...")
    ft_model_v2 = PeftModel.from_pretrained(base_model_for_inference, model_v2_final_path)
    run_final_evaluation(ft_model_v1, ft_model_v2, tokenizer, evaluation_data_path)
else:
    print("\nSkipping final evaluation as one or both models were not trained.")