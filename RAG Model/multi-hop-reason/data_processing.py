# data_processing.py
import os
import glob
import json
import PyPDF2
import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import config
# Import the new graph DB class
from graph_db import Neo4jGraphDB
from llm_interface import LLMInterface
from unstructured.partition.auto import partition

logger = logging.getLogger("CNTRAG_DataProcessing")

def _create_chunk_id(doc_name: str, element_idx: int, sub_chunk_idx: int, text_content: str) -> str:
    """Creates a unique and deterministic chunk ID."""
    # Using a hash of the content makes the ID stable even if parsing order changes slightly
    hasher = hashlib.sha1()
    # Use a prefix of the content to keep it fast
    hasher.update(text_content[:256].encode('utf-8'))
    content_hash = hasher.hexdigest()[:8]
    return f"{os.path.splitext(doc_name)[0]}_elem{element_idx}_sub{sub_chunk_idx}_{content_hash}"


def _elements_to_dicts(elements: List[Any], doc_name: str, doc_path: str) -> List[Dict[str, Any]]:
    """
    Converts unstructured elements to a list of dicts with standardized keys.
    """
    extracted_data = []
    for i, element in enumerate(elements):
        element_type = type(element).__name__
        text_content = str(element)

        base_metadata = {
            "document_name": doc_name,
            "document_path": doc_path,
            "element_index_in_doc": i,
            "element_type": element_type,
        }

        # Add page number if available
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
            base_metadata['page_number'] = element.metadata.page_number

        content_type = "text"
        if isinstance(element, Table):
            content_type = "table"
            try:
                from unstructured.cleaners.translate import translate_html_to_md
                text_content = translate_html_to_md(str(element))
                base_metadata["table_as_markdown"] = True
            except Exception:
                base_metadata["table_as_markdown"] = False

        base_metadata["content_type"] = content_type

        if text_content.strip():
            extracted_data.append({
                "raw_text": text_content,
                "metadata": base_metadata,
            })
    return extracted_data

# --- NEW FUNCTION: To extract entities using an LLM ---
def _extract_entities_from_chunk(chunk_text: str, llm_interface: LLMInterface) -> List[Dict[str, str]]:
    """
    Uses an LLM to extract key scientific entities from a text chunk with normalization.
    """
    prompt = f"""
    From the scientific text on Carbon Nanotubes below, extract key entities of types: "Method", "Catalyst", "Substrate", "CNT_Type", "Enhancer".
    - "Method": e.g., 'CVD', 'Laser Ablation', 'Vertically Aligned', 'Floating Catalyst'.
    - "Catalyst": e.g., 'Iron', 'Nickel', 'Fe', 'Ni', 'Cobalt'.
    - "Substrate": e.g., 'Silicon', 'Graphite', 'Alumina'.
    - "CNT_Type": e.g., 'SWCNT', 'MWCNT', 'Single-walled'.
    - "Enhancer": e.g., 'Water', 'H2O', 'CO2', 'Oxygen', 'Ammonia'.

    IMPORTANT INSTRUCTIONS:
    1. Normalize common names: 'Fe' -> 'Iron', 'Ni' -> 'Nickel', 'H2O' -> 'Water', 'single-walled' -> 'SWCNT', 'multi-walled' -> 'MWCNT'.
    2. Return the results as a valid JSON list of objects, where each object has a "type" and "name".
    3. If no entities are found, you MUST return an empty list [].

    Text:
    ---
    {chunk_text}
    ---

    JSON Output:
    """
    response_str = llm_interface.generate_response(prompt)
    try:
        response_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        entities = json.loads(response_str)
        if isinstance(entities, list):
            validated_entities = [e for e in entities if isinstance(e, dict) and "type" in e and "name" in e]
            return validated_entities
        return []
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to decode LLM response for entity extraction. Raw: {response_str[:100]}...")
        return []

def _extract_authors_from_document(doc_text_snippet: str, llm_interface: LLMInterface) -> List[str]:
    """
    Uses an LLM to extract author names from the beginning of a document.
    """
    if not doc_text_snippet.strip():
        return []
        
    prompt = f"""
    Your task is to act as an expert librarian. From the following text, which represents the first page of a scientific paper, your sole job is to extract the author names.

    RULES:
    - Author names are typically located between the paper title and the abstract or introduction.
    - Ignore affiliations, journal names, keywords, and any other text.
    - Handle complex names and initials (e.g., "A. John Hart", "Yoeri van de Burgt").
    - Return the names as a clean JSON list of strings. Example: ["C. Ryan Oliver", "Erik S. Polsen", "A. John Hart"].
    - If you cannot find any author names, you MUST return an empty list: [].

    Analyze this text snippet:
    ---
    {doc_text_snippet}
    ---

    JSON list of author names:
    """
    response_str = llm_interface.generate_response(prompt)
    try:
        # Basic cleanup
        response_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        authors = json.loads(response_str)
        if isinstance(authors, list) and all(isinstance(a, str) for a in authors):
            return authors
        return []
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to decode LLM response into JSON for author extraction. Raw: {response_str[:100]}...")
        return []
    
# This is a new helper function for our smart chunking
def _create_chunk_id(doc_name: str, element_idx: int, chunk_idx: int) -> str:
    """Creates a unique ID for a chunk based on its document and position."""
    # We remove the content hash for simplicity as position is now more stable
    return f"{os.path.splitext(doc_name)[0]}_elem{element_idx}_chunk{chunk_idx}"

# --- THE NEW, FULLY REVISED parse_and_chunk_documents ---
def parse_and_chunk_documents(
    documents_path_pattern: str,
    graph_db: Neo4jGraphDB,
    embedding_interface: LLMInterface,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    logger_parent: logging.Logger = logging.getLogger("CNTRAG")
) -> List[Dict[str, Any]]:
    """
    Parses documents using `unstructured` for multi-modal RAG. It handles text,
    tables, and images, creating descriptive text chunks for each.
    """
    global logger
    logger = logger_parent

    logger.info("Starting MULTI-MODAL document parsing with `unstructured`...")
    document_files = glob.glob(documents_path_pattern)
    document_files = [f for f in document_files if os.path.isfile(f) and not os.path.basename(f).startswith('~')]

    if not document_files:
        logger.error(f"No document files found matching pattern: {documents_path_pattern}")
        return []

    all_final_chunks_for_vector_store: List[Dict[str, Any]] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for doc_path in document_files:
        doc_name = os.path.basename(doc_path)
        logger.info(f"Processing document: {doc_name}")

        try:
            # Use "hi_res" strategy to extract images and tables accurately
            # This may require setting UNSTRUCTURED_API_KEY for the best results
            # elements = partition(filename=doc_path, strategy="hi_res", infer_table_structure=True)
            # The new line using a more direct strategy
            elements = partition(filename=doc_path, strategy="fast")
            
            # --- NEW AUTHOR EXTRACTION LOGIC ---
            # Concatenate the text from the first few elements to find authors
            doc_beginning_text = "\n".join([str(el) for el in elements[:10]]) # Use first 5 elements as context
            
            # --- START OF DEBUG BLOCK ---
            print("="*50)
            print(f"DEBUG: Analyzing snippet for authors from {doc_name}:")
            print(doc_beginning_text[:500] + "...") # Print the first 500 chars of the snippet
            # --- END OF DEBUG BLOCK ---
        
            authors = _extract_authors_from_document(doc_beginning_text, embedding_interface)
            
            # --- START OF DEBUG BLOCK ---
            print(f"DEBUG: LLM returned authors: {authors}")
            # --- END OF DEBUG BLOCK ---
            
            if authors:
                logger.info(f"Extracted {len(authors)} authors from {doc_name}: {authors}")
                # We will link them to the document in the KG (see Step 2)
            # --- END OF NEW LOGIC ---
            
            doc_all_chunks_for_kg = []
            
            for i, element in enumerate(elements):
                element_type = type(element).__name__
                element_chunks = []

                if element_type == "Table":
                    table_html = getattr(element.metadata, 'text_as_html', str(element))
                    # Convert HTML table to Markdown for better LLM readability
                    try:
                        from unstructured.cleaners.translate import translate_html_to_md
                        table_md = translate_html_to_md(table_html)
                        chunk_text = f"The following is a data table:\n\n{table_md}"
                        element_chunks.append(chunk_text)
                        logger.debug(f"Successfully processed a table on page {element.metadata.page_number}.")
                    except ImportError:
                        logger.warning("Could not import `translate_html_to_md`. Using raw table text. `pip install unstructured[md]` may be needed.")
                        element_chunks.append(f"Data Table: {str(element)}")
                
                elif element_type == "Image":
                    image_bytes = getattr(element, 'content', None)
                    if image_bytes:
                        prompt = "Analyze this image from a scientific paper on Carbon Nanotubes. Describe it in detail. What is being plotted? What are the axes? What is the key trend or feature shown? If it's a micrograph, describe the morphology."
                        summary = embedding_interface.get_image_summary(image_bytes, prompt)
                        if "LLM_ERROR" not in summary and "Unsupported" not in summary:
                           chunk_text = f"The following is a description of a scientific image/plot:\n\n{summary}"
                           element_chunks.append(chunk_text)
                           logger.debug(f"Successfully summarized an image on page {element.metadata.page_number}.")
                
                else: # Default handling for text elements
                    # Smart merging of titles/headers with the following text
                    if element_type == "Title" or "Header" in element_type:
                        if i + 1 < len(elements) and type(elements[i+1]).__name__ not in ["Table", "Image"]:
                            elements[i+1].text = f"{str(element)}\n\n{str(elements[i+1])}"
                        continue

                    # Split longer text elements
                    if len(str(element).strip()) > chunk_overlap:
                        element_chunks.extend(text_splitter.split_text(str(element)))
                    elif len(str(element).strip()) > 0:
                        element_chunks.append(str(element))

                # --- Process the generated chunks for this element ---
                for j, chunk_text in enumerate(element_chunks):
                    cleaned_chunk = chunk_text.strip()
                    if len(cleaned_chunk) < config.MIN_CHUNK_LENGTH:
                        continue

                    chunk_id = _create_chunk_id(doc_name, i, j)
                    
                    chunk_metadata = {
                        "document_name": doc_name,
                        "chunk_id": chunk_id,
                        "element_type": element_type, # Now includes 'Table', 'Image'
                        "page_number": getattr(element.metadata, 'page_number', None)
                    }

                    chunk_info = {"chunk_text": cleaned_chunk, "metadata": chunk_metadata}
                    doc_all_chunks_for_kg.append(chunk_info)
                    all_final_chunks_for_vector_store.append(chunk_info)
            
            # --- KG Population Step (works seamlessly with new chunk types) ---
            if doc_all_chunks_for_kg:
                # Add a 'type' property to the chunk node in the graph for better modeling
                for chunk in doc_all_chunks_for_kg:
                    chunk['metadata']['node_type'] = chunk['metadata']['element_type']

                graph_db.add_document_and_chunks(doc_name, doc_all_chunks_for_kg)
                # --- NEW: Link document to extracted authors ---
                if authors:
                    graph_db.link_document_to_authors(doc_name, authors)
                
                logger.info(f"Extracting and linking entities for {len(doc_all_chunks_for_kg)} multi-modal chunks...")
                for chunk in doc_all_chunks_for_kg:
                    entities = _extract_entities_from_chunk(chunk['chunk_text'], embedding_interface)
                    if entities:
                        graph_db.link_chunk_to_entities(chunk['metadata']['chunk_id'], entities)

        except Exception as e:
            logger.exception(f"Failed to process file {doc_path}: {e}")
            
    logger.info(f"Finished multi-modal parsing. Total chunks for vector store: {len(all_final_chunks_for_vector_store)}")
    return all_final_chunks_for_vector_store


# Fallback basic parser (ensure it's defined or imported correctly)
def parse_document_text(doc_path: str, logger_param: logging.Logger) -> Optional[List[Tuple[int, str]]]:
    """Basic text extraction for PDF/DOCX (simplified fallback)."""
    logger_param.debug(f"Basic parse_document_text called for {doc_path}")
    content_list = []
    try:
        if doc_path.lower().endswith(".pdf"):
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    logger_param.warning(f"PDF {doc_path} is encrypted. Skipping basic parsing.")
                    return None
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content_list.append((page_num + 1, page_text))
        elif doc_path.lower().endswith((".docx")):
            import docx
            document = docx.Document(doc_path)
            full_text_parts = [p.text for p in document.paragraphs if p.text.strip()]
            if full_text_parts:
                content_list.append((1, "\n".join(full_text_parts))) # Page 1 for whole docx
        else:
            logger_param.warning(f"Unsupported file type for basic parsing: {doc_path}")
            return None
        return content_list
    except ImportError as ie:
        logger_param.error(f"Import error during basic parsing of {doc_path}: {ie}. Ensure PyPDF2 and python-docx are installed.")
        return None
    except Exception as e:
        logger_param.error(f"Error in basic parse_document_text for {doc_path}: {e}")
        return None