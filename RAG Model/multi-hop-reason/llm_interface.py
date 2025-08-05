# llm_interface.py
import logging
import time
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Dict, List, Optional, Any

import google.generativeai as genai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Using Langchain for OpenAI
# Alternatively, for direct OpenAI API usage for embeddings:
# import openai
from PIL import Image
import io

import config # Import config for defaults

class LLMInterface:
    """Handles interactions with LLM providers (Google Gemini or OpenAI)."""

    def __init__(self,
                 llm_provider: str = config.DEFAULT_GENERATIVE_LLM_PROVIDER,
                 google_api_key: Optional[str] = config.GOOGLE_API_KEY,
                 openai_api_key: Optional[str] = config.OPENAI_API_KEY,
                 # Google specific
                 google_model_id: str = config.DEFAULT_GOOGLE_MODEL_ID,
                 google_embedding_model_id: str = config.DEFAULT_GOOGLE_EMBEDDING_MODEL,
                 google_generation_config: Optional[Dict] = None,
                 # OpenAI specific
                 openai_chat_model: str = config.DEFAULT_OPENAI_CHAT_MODEL,
                 openai_embedding_model: str = config.DEFAULT_OPENAI_EMBEDDING_MODEL,
                 openai_generation_config: Optional[Dict] = None,
                 # Common
                 use_embedding_cache: bool = config.DEFAULT_USE_EMBEDDING_CACHE,
                 use_llm_cache: bool = config.DEFAULT_USE_LLM_CACHE,
                 logger: logging.Logger = logging.getLogger("CNTRAG")):

        self.logger = logger
        self.llm_provider = llm_provider.lower()

        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self.use_llm_cache = use_llm_cache
        self.llm_cache: Dict[str, str] = {}
        self.embedding_cache_hits = 0
        self.llm_cache_hits = 0

        if self.llm_provider == "google":
            print(f"\n\nInitializing Google LLM Interface with model: {google_model_id}\n\n")
            if not google_api_key:
                raise ValueError("Google API Key is required for Google provider.")
            self.google_api_key = google_api_key
            self.google_model_id = google_model_id
            self.google_embedding_model_name = f"models/{google_embedding_model_id}"
            self.google_generation_config = google_generation_config or config.DEFAULT_GOOGLE_GENERATION_CONFIG
            try:
                genai.configure(api_key=self.google_api_key)
                self.google_generative_model = genai.GenerativeModel(
                    model_name=self.google_model_id,
                    generation_config=self.google_generation_config,
                )
                self.logger.info(f"Initialized Google Generative Model: {self.google_model_id}")
                self.logger.info(f"Initialized Google Embedding Model: {google_embedding_model_id}")
            except Exception as e:
                self.logger.error(f"Error initializing Google AI clients: {e}")
                raise
        elif self.llm_provider == "openai":
            print(f"\n\nInitializing OpenAI LLM Interface with chat model: {openai_chat_model}\n\n")
            if not openai_api_key:
                raise ValueError("OpenAI API Key is required for OpenAI provider.")
            self.openai_api_key = openai_api_key
            self.openai_chat_model_name = openai_chat_model
            self.openai_embedding_model_name = openai_embedding_model
            self.openai_generation_config = openai_generation_config or config.DEFAULT_OPENAI_GENERATION_CONFIG
            try:
                self.openai_chat_llm = ChatOpenAI(
                    model=self.openai_chat_model_name,
                    openai_api_key=self.openai_api_key,
                    temperature=self.openai_generation_config.get("temperature", 0.1),
                    max_tokens=self.openai_generation_config.get("max_tokens", 2048)
                )
                self.openai_embedding_client = OpenAIEmbeddings(
                    model=self.openai_embedding_model_name,
                    openai_api_key=self.openai_api_key
                )
                self.logger.info(f"Initialized OpenAI Chat Model: {self.openai_chat_model_name}")
                self.logger.info(f"Initialized OpenAI Embedding Model: {self.openai_embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI clients: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'google' or 'openai'.")

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def get_embedding(self, text: str, task_type="RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """Generates embeddings with retry logic and caching, supporting multiple providers."""
        # For OpenAI, task_type is not directly used in the same way as Google's embed_content API
        # We'll use the text directly.
        cache_key = f"{self.llm_provider}::{task_type}::{text}" # Add provider to cache key
        if self.use_embedding_cache and cache_key in self.embedding_cache:
             self.logger.debug(f"Embedding cache HIT for provider '{self.llm_provider}', task '{task_type}'")
             self.embedding_cache_hits += 1
             return self.embedding_cache[cache_key]

        if not text or not text.strip():
            self.logger.warning(f"Attempted to embed empty text for task '{task_type}'. Returning None.")
            return None

        self.logger.debug(f"Getting embedding via '{self.llm_provider}' for task '{task_type}' (text length: {len(text)})...")
        embedding = None
        try:
            if self.llm_provider == "google":
                response = genai.embed_content(
                    model=self.google_embedding_model_name,
                    content=text,
                    task_type=task_type,
                )
                embedding = response.get('embedding')
            elif self.llm_provider == "openai":
                embedding = self.openai_embedding_client.embed_query(text) # Langchain method

            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                if self.use_embedding_cache:
                    self.embedding_cache[cache_key] = embedding
                return embedding
            else:
                self.logger.error(f"Embedding API ({self.llm_provider}) returned invalid/empty embedding. Response: {response if self.llm_provider == 'google' else 'N/A for OpenAI direct response'}")
                return None
        except Exception as e:
            error_str = str(e).lower()
            # Generic error handling, specific API error messages might differ
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                self.logger.warning(f"Rate limit or quota error during embedding ({self.llm_provider}): {e}. Retrying...")
                raise
            elif "api key" in error_str or "authentication" in error_str or "permission" in error_str or "401" in error_str or "403" in error_str:
                self.logger.error(f"Authentication/Permission Error during embedding ({self.llm_provider}): {e}. NOT RETRYING.")
                return None
            elif "server error" in error_str or "500" in error_str or "503" in error_str:
                self.logger.warning(f"Server error during embedding ({self.llm_provider}): {e}. Retrying...")
                raise
            elif "invalid" in error_str or "bad request" in error_str or "400" in error_str:
                 log_snippet = text[:100].encode('unicode_escape').decode('utf-8')
                 self.logger.error(f"Invalid request/content error during embedding ({self.llm_provider}): {e}. Text snippet (escaped): '{log_snippet}...'. NOT RETRYING.")
                 return None
            else:
                self.logger.error(f"Non-retryable or unexpected error ({self.llm_provider}) generating embeddings: {e}")
                return None

    def batch_get_embeddings(self, text_list: List[str], batch_size=10, task_type="RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """Process embeddings for a list of texts sequentially with retries."""
        # Note: OpenAI's Langchain client `embed_documents` can handle batching internally.
        # For simplicity and consistency with the Google path, we'll keep the loop.
        # If performance with OpenAI embeddings becomes an issue, consider using `self.openai_embedding_client.embed_documents(text_list)` directly.

        self.logger.info(f"Getting embeddings for {len(text_list)} texts (sequential calls) via '{self.llm_provider}'")
        all_embeddings: List[Optional[List[float]]] = []
        rate_limit_delay = 0.5 # May need adjustment based on provider

        for i, text in enumerate(text_list):
            self.logger.debug(f"Processing embedding for item {i+1}/{len(text_list)}")
            emb = None
            try:
                emb = self.get_embedding(text, task_type=task_type)
                if emb is None:
                    self.logger.warning(f"Embedding failed for item {i+1} after retries.")
            except Exception as e:
                self.logger.error(f"Embedding failed permanently for item {i+1} after retries: {e}")

            all_embeddings.append(emb)

            if (i + 1) % batch_size == 0: # Sleep every `batch_size` calls
                try:
                    time.sleep(rate_limit_delay)
                except KeyboardInterrupt:
                    self.logger.warning("Embedding process interrupted.")
                    break

            if (i + 1) % 100 == 0 or (i + 1) == len(text_list):
                successful_count = sum(1 for e in all_embeddings if e is not None)
                self.logger.info(f"Processed {i + 1}/{len(text_list)} embeddings ({successful_count} successful).")

        return all_embeddings

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM with caching and error handling."""
        cache_key = f"{self.llm_provider}::{prompt}" # Add provider to cache key
        if self.use_llm_cache and cache_key in self.llm_cache:
             self.logger.debug(f"LLM cache HIT for provider '{self.llm_provider}'")
             self.llm_cache_hits += 1
             return self.llm_cache[cache_key]

        self.logger.debug(f"Generating LLM response via '{self.llm_provider}' for prompt (length: {len(prompt)}): '{prompt[:200]}...'")
        start_time = time.time()
        generated_text = f"LLM_ERROR: Provider '{self.llm_provider}' not properly handled in generate_response." # Default error

        try:
            if self.llm_provider == "google":
                response = self.google_generative_model.generate_content(prompt)
                if not response.candidates:
                    feedback = getattr(response, 'prompt_feedback', None)
                    block_reason = "Unknown"
                    if feedback: block_reason = getattr(feedback, 'block_reason', 'Unknown')
                    self.logger.warning(f"Google LLM Response blocked. Reason: {block_reason}.")
                    generated_text = f"LLM_ERROR: Response blocked (Reason: {block_reason})."
                elif hasattr(response, 'text') and response.text:
                    generated_text = response.text
                else: # Fallback for Google response structure
                    try:
                        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                            if not generated_text: generated_text = "LLM_ERROR: Response content was empty."
                        else: generated_text = "LLM_ERROR: Response structure unexpected or empty."
                    except Exception as e_parse:
                        self.logger.error(f"Error accessing Google response parts: {e_parse}")
                        generated_text = "LLM_ERROR: Failed to access response content."

            elif self.llm_provider == "openai":
                # Langchain ChatOpenAI expects a list of messages or a string.
                # For simplicity here, we'll wrap the prompt if needed, or assume it's a direct string.
                # from langchain_core.messages import HumanMessage
                # response = self.openai_chat_llm.invoke([HumanMessage(content=prompt)])
                response = self.openai_chat_llm.invoke(prompt) # invoke directly with string prompt
                if hasattr(response, 'content') and isinstance(response.content, str):
                    generated_text = response.content
                else:
                    self.logger.warning(f"OpenAI LLM Response structure unexpected: {response}")
                    generated_text = "LLM_ERROR: OpenAI response content not found or not a string."

            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM generation ({self.llm_provider}) took {elapsed_time:.2f}s. Response length: {len(generated_text)}")

            if self.use_llm_cache:
                 self.llm_cache[cache_key] = generated_text # Use 'cache_key' to save
            return generated_text

        except Exception as e:
            self.logger.exception(f"Exception during LLM generation call ({self.llm_provider}): {e}")
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                return f"LLM_ERROR: API rate limit or quota exceeded ({self.llm_provider})."
            elif "api key" in error_str or "authentication" in error_str or "permission" in error_str or "401" in error_str or "403" in error_str:
                return f"LLM_ERROR: Invalid API Key or insufficient permissions ({self.llm_provider})."
            return f"LLM_ERROR: An unexpected error occurred ({self.llm_provider}): {str(e)}"

    def get_cache_stats(self) -> Dict[str, int]:
        """Returns the current cache hit counts."""
        return {
            "embedding_cache_hits": self.embedding_cache_hits,
            "llm_cache_hits": self.llm_cache_hits
        }
    
    def get_image_summary(self, image_bytes: bytes, prompt: str) -> str:
        """
        Uses a multi-modal LLM to generate a description of an image.
        Currently implemented for Google Gemini.
        """
        if self.llm_provider != "google":
            self.logger.warning("Image summarization is currently only implemented for the 'google' provider.")
            return "Unsupported for this provider"

        self.logger.debug("Generating summary for image...")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Use the same generative model, it's multi-modal
            response = self.google_generative_model.generate_content([prompt, image])

            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                # Handle potential blocks or empty responses
                self.logger.warning(f"Vision model did not return text. Response parts: {response.parts}")
                return "Could not generate a summary for the image."

        except Exception as e:
            self.logger.exception(f"Exception during image summarization: {e}")
            return f"LLM_ERROR: Failed to process image ({e})"