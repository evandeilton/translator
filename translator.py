#!/usr/bin/env python3
import os
import sys
import math
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pdfplumber
import PyPDF2
import anthropic  # Anthropic library
import openai     # OpenAI library
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional progress bar; install with: pip install tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ------------------------------------------------------------------------
# Additional Libraries for Advanced PDF Conversion
# ------------------------------------------------------------------------
# PyMuPDF for native PDF text extraction
try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install it with 'pip install pymupdf'.")

# pdf2image and pytesseract for OCR-based extraction
from pdf2image import convert_from_path
import pytesseract

# MarkItDown (optional) for multi-format conversion preserving LaTeX/math
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

# ------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not any([DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY]):
    raise ValueError("No API key configured. Please set one of: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")

BASE_URLS = {
    "deepseek": os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
    "anthropic": "https://api.anthropic.com/v1/messages",  # Anthropic endpoint
    "openai": "https://api.openai.com/v1/chat/completions"
}

SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.doc,.docx,.odt").split(",")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simplified pricing table for DeepSeek (per 1M tokens)
DEEPSEEK_PRICING = {
    "deepseek-chat": {
        "cache_hit":  {"input": 0.07,  "output": 1.10},
        "cache_miss": {"input": 0.14,  "output": 1.10},
    },
    "deepseek-reasoner": {
        "cache_hit":  {"input": 0.14,  "output": 2.19},
        "cache_miss": {"input": 0.55,  "output": 2.19},
    }
}

# ------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------
def setup_logger(trace: bool) -> logging.Logger:
    """
    Configures and returns a logger with console output.
    
    Args:
        trace (bool): If True, sets log level to DEBUG; otherwise INFO.
    
    Returns:
        logging.Logger: Configured logger.
    """
    level = logging.DEBUG if trace else logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# ------------------------------------------------------------------------
# File Validation & Conversion
# ------------------------------------------------------------------------
def validate_file(file_path: Path, logger: logging.Logger) -> None:
    """
    Validates that a file exists and its format is supported.

    Args:
        file_path (Path): Path to the file to validate.
        logger (logging.Logger): Logger for diagnostic messages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        allowed_formats = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Format '{file_path.suffix}' is not supported. Allowed formats: {allowed_formats}"
        )
    logger.debug(f"File '{file_path.name}' is valid with suffix '{file_path.suffix}'.")

def convert_to_pdf(file_path: Path, logger: logging.Logger) -> Path:
    """
    Converts a document to PDF if necessary using 'unoconv'.

    Args:
        file_path (Path): Path to the input file.
        logger (logging.Logger): Logger for diagnostic messages.

    Returns:
        Path: Path to the converted PDF file.

    Raises:
        RuntimeError: If the 'unoconv' command fails.
    """
    if file_path.suffix.lower() == ".pdf":
        logger.info(f"'{file_path.name}' is already a PDF. Conversion not needed.")
        return file_path

    output_path = TEMP_DIR / f"{file_path.stem}.pdf"
    logger.info(f"Converting '{file_path.name}' to PDF...")

    try:
        result = subprocess.run(
            ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
            check=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"unoconv failed (exit code {result.returncode}).")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode().strip() if e.stderr else "Unknown error."
        logger.error(f"Failed to convert '{file_path}'. Error:\n{err_msg}")
        raise RuntimeError(f"Conversion of '{file_path}' to PDF failed.") from e

    logger.info(f"Conversion complete. PDF saved to '{output_path}'.")
    return output_path

# ------------------------------------------------------------------------
# Advanced PDF -> Text Extraction using Multiple Methods
# ------------------------------------------------------------------------
def convert_pdf_to_text(pdf_path: Path, logger: logging.Logger) -> str:
    """
    Converts a PDF file to a text string using multiple methods to maximize quality.
    It attempts extraction via:
      1. PyMuPDF (fitz) – preserves layout and LaTeX if embedded as text.
      2. Tesseract OCR (with pdf2image) – for scanned/image-based PDFs.
      3. MarkItDown – for conversion preserving Markdown and LaTeX equations (if available).
    
    The function inserts a page-break marker (<<<PAGE_BREAK>>>) when possible (e.g., via PyMuPDF)
    so that the resulting text can later be split into pages.
    
    Args:
        pdf_path (Path): Path to the PDF file.
        logger (logging.Logger): Logger for diagnostic messages.
    
    Returns:
        str: A string containing the transcribed text from the PDF.
    
    Raises:
        Exception: If all extraction methods fail to extract any text.
    """
    logger.info(f"Starting advanced PDF-to-text conversion for '{pdf_path.name}'...")
    extraction_results: Dict[str, str] = {}

    # 1. Extraction using PyMuPDF
    def extract_with_pymupdf() -> str:
        text = ""
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Append a marker after each page to aid later splitting.
                page_text = page.get_text().strip()
                text += page_text + "\n<<<PAGE_BREAK>>>\n"
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")
        return text

    logger.info("Attempting extraction using PyMuPDF...")
    text_pymupdf = extract_with_pymupdf()
    extraction_results['pymupdf'] = text_pymupdf
    logger.debug(f"PyMuPDF extracted {len(text_pymupdf)} characters.")

    # 2. Extraction using Tesseract OCR (with pdf2image)
    def extract_with_ocr() -> str:
        text = ""
        try:
            # Use dpi=300 for better OCR quality.
            images = convert_from_path(str(pdf_path), dpi=300)
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='eng')
                # Add a page-break marker manually.
                text += page_text.strip() + "\n<<<PAGE_BREAK>>>\n"
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
        return text

    logger.info("Attempting extraction using Tesseract OCR...")
    text_ocr = extract_with_ocr()
    extraction_results['ocr'] = text_ocr
    logger.debug(f"OCR extraction yielded {len(text_ocr)} characters.")

    # 3. Extraction using MarkItDown (if available)
    def extract_with_markitdown() -> str:
        text = ""
        if not MARKITDOWN_AVAILABLE:
            logger.warning("MarkItDown not available. Skipping MarkItDown extraction.")
            return text
        try:
            markitdown = MarkItDown()
            result = markitdown.convert(str(pdf_path))
            text = result.text_content
        except Exception as e:
            logger.error(f"MarkItDown extraction error: {e}")
        return text

    if MARKITDOWN_AVAILABLE:
        logger.info("Attempting extraction using MarkItDown...")
        text_markitdown = extract_with_markitdown()
        extraction_results['markitdown'] = text_markitdown
        logger.debug(f"MarkItDown extraction yielded {len(text_markitdown)} characters.")
    else:
        extraction_results['markitdown'] = ""

    # Choose the extraction method that returned the most (non-empty) text.
    best_method = None
    best_text = ""
    for method, text in extraction_results.items():
        if len(text.strip()) > len(best_text.strip()):
            best_text = text
            best_method = method

    logger.info(f"Selected extraction method: {best_method} with {len(best_text)} characters.")

    if not best_text.strip():
        raise Exception("Failed to extract text from the PDF using all available methods.")

    return best_text

# ------------------------------------------------------------------------
# Approximate Token Count / Chunking Logic
# ------------------------------------------------------------------------
def approximate_token_count(text: str) -> int:
    """
    Estimates the number of tokens in a text.
    For production, consider using a dedicated tokenizer (e.g., tiktoken).

    Args:
        text (str): Text to analyze.

    Returns:
        int: Estimated token count.
    """
    return math.ceil(len(text) / 4)

def chunk_text(text: str, max_chunk_tokens: int) -> List[str]:
    """
    Splits the text into chunks that do not exceed the maximum token count.

    Args:
        text (str): Original text.
        max_chunk_tokens (int): Maximum allowed tokens per chunk.

    Returns:
        List[str]: List of text chunks within the limit.
    """
    lines = text.split("\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_count = 0

    for line in lines:
        line_tokens = approximate_token_count(line)

        # If a single line is too long, break it by words
        if line_tokens >= max_chunk_tokens:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_count = 0

            words = line.split()
            subchunk: List[str] = []
            subcount = 0
            for word in words:
                wcount = approximate_token_count(word + " ")
                if subcount + wcount <= max_chunk_tokens:
                    subchunk.append(word)
                    subcount += wcount
                else:
                    chunks.append(" ".join(subchunk))
                    subchunk = [word]
                    subcount = wcount

            if subchunk:
                chunks.append(" ".join(subchunk))
            continue

        if current_count + line_tokens > max_chunk_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_count = line_tokens
        else:
            current_chunk.append(line)
            current_count += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# ------------------------------------------------------------------------
# Translation API (DeepSeek, Anthropic, OpenAI, Gemini)
# ------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_text(
    text: str,
    target_lang: str,
    model: str,
    provider: str,
    logger: logging.Logger
) -> Tuple[str, Dict]:
    """
    Translates a block of text using the DeepSeek, Anthropic, OpenAI, or Gemini API.

    Args:
        text (str): Original text to translate.
        target_lang (str): Target language code (e.g., 'pt', 'en').
        model (str): Model name (e.g., 'deepseek-chat', 'claude-3-sonnet', 'gpt-4o-mini', 'gemini-2.0-flash').
        provider (str): API provider ('deepseek', 'anthropic', 'openai', or 'gemini').
        logger (logging.Logger): Logger for diagnostic messages.

    Returns:
        Tuple[str, Dict]: (translated_text, usage_info)
    """
    system_prompt = (
        "You are an elite academic translator with expertise in scientific and technical documentation, committed to preserving the integrity of academic works while ensuring precise translation.\n\n"
        "ABSOLUTE PRESERVATION RULES (NÃO DEVE SER TRADUZIDO):\n"
        "1. Bibliography and references\n"
        "2. Author names and affiliations\n"
        "3. URLs, DOIs, and digital identifiers\n"
        "4. Journal names and conference proceedings\n"
        "5. Software names, packages, and tools\n"
        "6. Dataset identifiers and names\n"
        "7. Standard scientific abbreviations\n"
        "8. Variable names in code or equations\n"
        "9. Statistical test names (e.g., t-test, ANOVA)\n\n"
        "MATHEMATICAL AND TECHNICAL NOTATION:\n"
        "1. Equations:\n"
        "   - Inline math: $...$\n"
        "   - Display math: $$...$$\n"
        "   - Preserve all operators, symbols, spacing, and LaTeX formatting, including equation numbering and references\n"
        "   - Find out all greek letters CAPITAL or not and write with inline symbol (eg. λ -> $\lambda$, ϕ -> $\phi$, τ -> $\tau$, and so on)\n"

        "2. Code:\n"
        "   - Preserve all code blocks with original syntax\n"
        "   - Maintain comments in target language\n"
        "   - Keep variable names and function calls unchanged\n"
        "3. Technical Elements:\n"
        "   - Preserve SI units and measurements\n"
        "   - Maintain statistical notation conventions\n"
        "   - Keep chemical formulas and nomenclature intact\n\n"
        "DOCUMENT STRUCTURE:\n"
        "1. Formatting:\n"
        "   - Identify and preserve titles, subtitles, lists, bullets, and paragraphs using Markdown syntax (**, *, #, ##)\n"
        "   - Maintain table structure, alignment, figure and table numbering, and cross-references\n"
        "2. Organization:\n"
        "   - Preserve original paragraph structure, section and subsection hierarchy, footnotes, endnotes, and appendix formatting\n\n"
        "ACADEMIC CONVENTIONS:\n"
        "1. Terminology:\n"
        "   - Use field-appropriate technical vocabulary and maintain term consistency\n"
        "2. Style:\n"
        "   - Maintain academic register, formality, hedging language, and citation styles\n\n"
        "QUALITY CONTROL:\n"
        "1. Consistency and Accuracy:\n"
        "   - Ensure uniform terminology and preserve all technical formatting\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Provide only the translated content without any explanations or extra text\n"
        "- Do not include any part of the original text unless explicitly specified\n"
        "Final Check:\n"
        "Before outputting, verify that all formatting rules have been strictly followed."
    )

    if provider == "deepseek":
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data: Dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
            ],
            "temperature": 0,
            "max_tokens": 4000
        }
        logger.debug(f"Sending request to DeepSeek: model={model}, target_lang={target_lang}")
        resp = requests.post(BASE_URLS["deepseek"], json=data, headers=headers, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        translated_text = resp_json["choices"][0]["message"]["content"]
        usage_info = resp_json.get("usage", {})

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.debug(f"Sending request to Anthropic: model={model}, target_lang={target_lang}")
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Translate to {target_lang}:\n{text}"}]
        )
        translated_text = message.content[0].text
        usage_info = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens
        }

    elif provider == "openai":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.debug(f"Sending request to OpenAI: model={model}, target_lang={target_lang}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
            ],
            max_tokens=4000,
            temperature=0
        )
        translated_text = response.choices[0].message.content or ""
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0
        }

    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.debug(f"Sending request to Gemini: model={model}, target_lang={target_lang}")
        response = client.models.generate_content(
            model=model,
            contents=f"Translate to {target_lang}:\n{text}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=4000,
            )
        )
        translated_text = response.text
        usage_info = {
            "prompt_tokens": 0,  # Gemini API does not return token counts
            "completion_tokens": 0
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    logger.debug("API response received successfully.")
    return translated_text, usage_info

# ------------------------------------------------------------------------
# Parallel Translation of All Chunks
# ------------------------------------------------------------------------
def chunk_all_pages(
    page_texts: List[str],
    max_chunk_tokens: int,
    logger: logging.Logger
) -> List[List[str]]:
    """
    Splits the text from each page into chunks if needed.
    
    Args:
        page_texts (List[str]): Raw text of each page.
        max_chunk_tokens (int): Approximate token limit per chunk.
        logger (logging.Logger): Logger for diagnostic messages.
    
    Returns:
        List[List[str]]: A list of lists where each sublist contains the chunks for that page.
    """
    logger.info("Splitting all pages into smaller segments...")
    chunked_pages: List[List[str]] = []
    for idx, text in enumerate(page_texts):
        if text.strip():
            chunks = chunk_text(text, max_chunk_tokens)
            logger.debug(f"Page {idx+1} has {len(chunks)} segment(s).")
        else:
            chunks = []
            logger.debug(f"Page {idx+1} has no extractable text.")
        chunked_pages.append(chunks)
    return chunked_pages

def translate_document_by_chunks(
    chunked_pages: List[List[str]],
    target_lang: str,
    model: str,
    provider: str,
    concurrency: int,
    logger: logging.Logger
) -> Tuple[List[List[str]], List[List[Dict]]]:
    """
    Translates all chunks from all pages in parallel.
    
    Args:
        chunked_pages (List[List[str]]): Each sublist contains the chunks for a page.
        target_lang (str): Target language code.
        model (str): Model name.
        provider (str): API provider.
        concurrency (int): Maximum number of threads.
        logger (logging.Logger): For diagnostic messages.
    
    Returns:
        Tuple[List[List[str]], List[List[Dict]]]:
            (translated_chunks_per_page, usage_data_per_chunk)
    """
    tasks = []  # (page_idx, chunk_idx, chunk_text)
    for i, page_chunks in enumerate(chunked_pages):
        for j, chunk_text_ in enumerate(page_chunks):
            tasks.append((i, j, chunk_text_))

    num_pages = len(chunked_pages)
    per_page_translated_chunks: List[List[Optional[str]]] = [
        [None] * len(chunked_pages[i]) for i in range(num_pages)
    ]
    per_page_usage_data: List[List[Optional[Dict]]] = [
        [None] * len(chunked_pages[i]) for i in range(num_pages)
    ]

    logger.info(
        f"Submitting {len(tasks)} segment(s) distributed across {num_pages} page(s) to a thread pool with size={concurrency}."
    )

    def worker(page_idx: int, chunk_idx: int, chunk_text_: str):
        return translate_text(chunk_text_, target_lang, model, provider, logger)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(worker, p, c, txt): (p, c)
            for (p, c, txt) in tasks
        }

        future_iter = tqdm(as_completed(future_to_task), total=len(tasks), desc="Translating segments") if TQDM_AVAILABLE else as_completed(future_to_task)
        for future in future_iter:
            p_idx, c_idx = future_to_task[future]
            try:
                translated_chunk, usage_info = future.result()
                per_page_translated_chunks[p_idx][c_idx] = translated_chunk
                per_page_usage_data[p_idx][c_idx] = usage_info
            except Exception as e:
                logger.error(f"Segment (page={p_idx+1}, chunk={c_idx+1}) failed: {e}")
                per_page_translated_chunks[p_idx][c_idx] = "[CHUNK FAILED]"
                per_page_usage_data[p_idx][c_idx] = {}

    return per_page_translated_chunks, per_page_usage_data

# ------------------------------------------------------------------------
# Usage & Cost Calculation
# ------------------------------------------------------------------------
def compute_usage_stats(
    usage_data: List[List[Optional[Dict]]]
) -> Dict[str, int]:
    """
    Aggregates usage statistics from all chunks.

    Args:
        usage_data (List[List[Optional[Dict]]]): List of usage dictionaries per chunk.
    
    Returns:
        Dict[str, int]: Aggregated statistics (prompt, completion, and total tokens).
    """
    prompt_sum = 0
    completion_sum = 0

    for page_usage_list in usage_data:
        for usage_info in page_usage_list:
            if not usage_info:
                continue
            p = usage_info.get("prompt_tokens", 0)
            c = usage_info.get("completion_tokens", 0)
            prompt_sum += p
            completion_sum += c

    total_sum = prompt_sum + completion_sum
    return {
        "prompt_tokens_total": prompt_sum,
        "completion_tokens_total": completion_sum,
        "total_tokens_total": total_sum
    }

def estimate_deepseek_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache: str
) -> float:
    """
    Estimates the cost for DeepSeek usage given the model, token counts, and cache scenario.

    Args:
        model (str): e.g., 'deepseek-chat' or 'deepseek-reasoner'.
        prompt_tokens (int)
        completion_tokens (int)
        cache (str): 'hit' or 'miss'.
    
    Returns:
        float: Cost in USD.
    """
    if model not in DEEPSEEK_PRICING:
        raise ValueError(f"Unknown model '{model}'. Check DEEPSEEK_PRICING.")
    if cache not in ("hit", "miss"):
        raise ValueError(f"Cache must be 'hit' or 'miss'. Received '{cache}'.")

    pricing = DEEPSEEK_PRICING[model][f"cache_{cache}"]
    cost_input_per_mil = pricing["input"]
    cost_output_per_mil = pricing["output"]

    input_millions = prompt_tokens / 1_000_000
    output_millions = completion_tokens / 1_000_000

    cost_input = input_millions * cost_input_per_mil
    cost_output = output_millions * cost_output_per_mil
    return cost_input + cost_output

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main() -> None:
    """
    CLI entry point para traduzir documentos PDF via DeepSeek, Anthropic, OpenAI ou Gemini.
    
    Etapas:
      1) Validação do formato do arquivo.
      2) Conversão para PDF (se necessário).
      3) Extração do texto do PDF usando a conversão avançada (convert_pdf_to_text).
      4) (Opcional) Envio do texto completo em uma única requisição para tradução.
         Caso contrário, dividir o texto em páginas (usando os marcadores inseridos)
         e posteriormente em segmentos.
      5) Tradução de todos os segmentos (paralelamente, se aplicável).
      6) Reassemble e salvamento da tradução final.
      7) Cálculo do uso e custo.
    """
    parser = argparse.ArgumentParser(
        description="Traduz documentos PDF via DeepSeek, Anthropic, OpenAI ou Gemini (com concorrência ao nível de segmento)."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Caminho para o arquivo de entrada (pdf/doc/docx/odt).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Código da língua alvo (ex.: 'pt', 'en').",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "anthropic", "openai", "gemini"],
        help="Provedor da API para tradução.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo específico a ser utilizado. Valor padrão: modelo padrão do provedor.",
    )
    parser.add_argument(
        "--cache",
        default="miss",
        choices=["hit", "miss"],
        help="Assume que os tokens de entrada são cache HIT ou MISS (somente para DeepSeek). Padrão: 'miss'.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=2000,
        help="Limite aproximado de tokens permitidos por segmento. Padrão=2000.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Número de requisições paralelas para tradução dos segmentos. Padrão=4.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        default=False,
        help="Se definido, o texto extraído completo será enviado para tradução em uma única requisição, em vez de ser dividido em partes.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Ativa o log em nível DEBUG.",
    )
    args = parser.parse_args()

    # Define os modelos padrão se não especificados
    if not args.model:
        args.model = {
            "deepseek": "deepseek-chat",
            "anthropic": "claude-3-5-haiku-20241022",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash"
        }[args.provider]

    logger = setup_logger(trace=args.trace)
    
    file_path = Path(args.file).resolve()
    try:
        # 1) Valida e converte para PDF, se necessário.
        validate_file(file_path, logger=logger)
        pdf_path = convert_to_pdf(file_path, logger=logger)

        # 2) Extrai o texto do PDF usando a extração avançada.
        extracted_text = convert_pdf_to_text(pdf_path, logger=logger)

        # Nova opção: tradução via texto completo em uma única requisição.
        if args.full_text:
            logger.info("Opção de tradução em texto completo selecionada. Traduzindo o documento completo em uma única requisição.")
            translated_text, usage_info = translate_text(extracted_text, args.target, args.model, args.provider, logger)
            output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(translated_text)
            logger.info(f"Tradução completa. Arquivo de saída salvo em '{output_file}'")

            # Cálculo do uso e custo (apenas DeepSeek)
            if args.provider == "deepseek":
                # Envolve usage_info em uma estrutura aninhada para reutilizar a função compute_usage_stats.
                usage_agg = compute_usage_stats([[usage_info]])
                total_prompt = usage_agg["prompt_tokens_total"]
                total_completion = usage_agg["completion_tokens_total"]
                total_tokens = usage_agg["total_tokens_total"]
                estimated_cost = estimate_deepseek_cost(
                    model=args.model,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    cache=args.cache
                )
                logger.info("Resumo de Uso:")
                logger.info(f"  Prompt Tokens:     {total_prompt}")
                logger.info(f"  Completion Tokens: {total_completion}")
                logger.info(f"  Total Tokens:      {total_tokens}")
                logger.info(f"Custo Estimado: ${estimated_cost:.4f}")
            else:
                logger.info("A estimativa de custo está disponível somente para DeepSeek.")
            return

        # Fluxo atual: dividir o texto em páginas usando o marcador inserido.
        page_texts = [page.strip() for page in extracted_text.split("<<<PAGE_BREAK>>>") if page.strip()]
        num_pages = len(page_texts)
        if num_pages == 0:
            raise ValueError(f"'{pdf_path.name}' não contém texto extraível.")

        logger.info(f"Extração avançada resultou em {num_pages} página(s).")

        # Divisão: cada página é segmentada conforme o limite de tokens.
        chunked_pages = chunk_all_pages(page_texts, max_chunk_tokens=args.max_chunk_tokens, logger=logger)

        # Tradução paralela dos segmentos.
        logger.info(
            f"Iniciando tradução de {num_pages} página(s) com concurrency={args.concurrency} e max_chunk_tokens={args.max_chunk_tokens}."
        )
        per_page_translated_chunks, per_page_usage_data = translate_document_by_chunks(
            chunked_pages=chunked_pages,
            target_lang=args.target,
            model=args.model,
            provider=args.provider,
            concurrency=args.concurrency,
            logger=logger
        )

        # Reúne a tradução final e salva em arquivo.
        output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"
        with open(output_file, "w", encoding="utf-8") as out:
            for i, translated_chunks in enumerate(per_page_translated_chunks):
                page_translation = "\n\n".join(translated_chunks)
                out.write(page_translation)
                if i < num_pages - 1:
                    out.write("\n\n<!-- NEXT PAGE -->\n\n")

        logger.info(f"Tradução completa. Arquivo de saída salvo em '{output_file}'")

        # Cálculo do uso e custo (apenas DeepSeek)
        if args.provider == "deepseek":
            usage_agg = compute_usage_stats(per_page_usage_data)
            total_prompt = usage_agg["prompt_tokens_total"]
            total_completion = usage_agg["completion_tokens_total"]
            total_tokens = usage_agg["total_tokens_total"]
            estimated_cost = estimate_deepseek_cost(
                model=args.model,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                cache=args.cache
            )
            logger.info("Resumo de Uso:")
            logger.info(f"  Prompt Tokens:     {total_prompt}")
            logger.info(f"  Completion Tokens: {total_completion}")
            logger.info(f"  Total Tokens:      {total_tokens}")
            logger.info(f"Custo Estimado: ${estimated_cost:.4f}")
        else:
            logger.info("A estimativa de custo está disponível somente para DeepSeek.")

    except Exception as e:
        logger.error(f"Tradução falhou: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
