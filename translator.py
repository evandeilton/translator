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

# For advanced retry logic (handles transient API errors)
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional progress bar; install via: pip install tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

BASE_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.doc,.docx,.odt").split(",")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pricing table for DeepSeek (per 1M tokens), simplified
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
    Sets up and returns a logger instance with console output.
    
    Args:
        trace (bool): If True, set logger level to DEBUG, else INFO.
    
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
    Validate if a file exists and is in a supported format.

    Args:
        file_path (Path): Path to the file to validate
        logger (logging.Logger): Logger for diagnostic messages

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        allowed_formats = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Format '{file_path.suffix}' not supported. "
            f"Allowed: {allowed_formats}"
        )
    logger.debug(f"File '{file_path.name}' is valid with suffix '{file_path.suffix}'.")


def convert_to_pdf(file_path: Path, logger: logging.Logger) -> Path:
    """
    Convert a document to PDF format if needed using `unoconv`.

    Args:
        file_path (Path): Path to the input file
        logger (logging.Logger): Logger for diagnostic messages

    Returns:
        Path: Path to the converted PDF file

    Raises:
        RuntimeError: If the `unoconv` command fails
    """
    if file_path.suffix.lower() == ".pdf":
        logger.info(f"'{file_path.name}' is already a PDF. No conversion needed.")
        return file_path

    output_path = TEMP_DIR / f"{file_path.stem}.pdf"
    logger.info(f"Converting '{file_path.name}' to PDF...")

    try:
        result = subprocess.run(
            ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
            check=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"unoconv failed (code {result.returncode}).")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode().strip() if e.stderr else "Unknown error."
        logger.error(f"Failed converting '{file_path}'. Error:\n{err_msg}")
        raise RuntimeError(f"Failed to convert '{file_path}' to PDF.") from e

    logger.info(f"Conversion complete. PDF saved at '{output_path}'.")
    return output_path


# ------------------------------------------------------------------------
# PDF -> Page Text Extraction (in memory)
# ------------------------------------------------------------------------
def extract_all_page_texts(pdf_path: Path, logger: logging.Logger) -> List[str]:
    """
    Extract the text content of each page in the given PDF, returning
    a list of page-level strings.

    Args:
        pdf_path (Path): Path to a PDF file
        logger (logging.Logger): For diagnostic messages

    Returns:
        List[str]: One string per page (even if empty).
    """
    logger.info(f"Extracting text from '{pdf_path.name}'...")
    page_texts: List[str] = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        logger.debug(f"Found {num_pages} pages in '{pdf_path.name}'.")

    # pdfplumber is often better at text extraction than PyPDF2
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            page_texts.append(raw_text)

    logger.info(f"Extraction complete. {len(page_texts)} pages read.")
    return page_texts


# ------------------------------------------------------------------------
# Token Handling / Chunk Logic
# ------------------------------------------------------------------------
def approximate_token_count(text: str) -> int:
    """
    Estimate the token count of a text. 
    For production, consider a dedicated tokenizer (e.g. 'tiktoken').

    Args:
        text (str): The text to analyze

    Returns:
        int: Estimated token count
    """
    # A very rough approximation: 4 characters per token
    return math.ceil(len(text) / 4)


def chunk_text(text: str, max_chunk_tokens: int) -> List[str]:
    """
    Safely chunk the text into segments that are each <= max_chunk_tokens.

    Args:
        text (str): Original text to be chunked
        max_chunk_tokens (int): Max tokens allowed per chunk

    Returns:
        List[str]: List of text chunks, each within the token limit
    """
    lines = text.split("\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_count = 0

    for line in lines:
        line_tokens = approximate_token_count(line)

        # If a single line alone is too big, forcibly break it by words
        if line_tokens >= max_chunk_tokens:
            # Push the existing chunk first
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

        # Otherwise, see if we can append to current chunk
        if current_count + line_tokens > max_chunk_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_count = line_tokens
        else:
            current_chunk.append(line)
            current_count += line_tokens

    # Flush remainder
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


# ------------------------------------------------------------------------
# DeepSeek Translation API
# ------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_text(
    text: str, 
    target_lang: str, 
    model: str, 
    logger: logging.Logger
) -> Tuple[str, Dict]:
    """
    Translate a block of text using DeepSeek. Returns (translated_text, usage_info).

    Args:
        text (str): The raw text to translate
        target_lang (str): The target language code (e.g. 'pt', 'en')
        model (str): DeepSeek model name (e.g. 'deepseek-chat', 'deepseek-reasoner')
        logger (logging.Logger): Logger for diagnostic messages

    Returns:
        Tuple[str, Dict]: (translated_text, usage_info)
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    data: Dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a technical translator specialized in academic/scientific content.\n"
                    "Instructions:\n"
                    "1. All equations must be in LaTeX notation.\n"
                    "2. All Greek letters and inline equations must be in LaTeX notation (e.g. $\alpha$, $x + 1 = 0$). Long Equations must be in LaTeX blocks (e.g. '\\[ x+2\\alpha \\]') according to the original text.\n"
                    "3. Maintain all Markdown formatting.\n"
                    "4. Keep proper names, citations, references.\n"
                    "5. Ensure terminology consistency.\n"
                    "6. Preserve paragraph structure and line breaks.\n"
                    "7. Return only translated text, no explanations.\n"
                    "8. Format tables in Markdown.\n"
                    "9. For images, place its legend in text."
                )
            },
            {
                "role": "user",
                "content": f"Translate to {target_lang}:\n{text}"
            }
        ],
        "temperature": 0,
        "max_tokens": 4000
    }

    logger.debug(f"Sending API request to DeepSeek: model={model}, target_lang={target_lang}")
    resp = requests.post(BASE_URL, json=data, headers=headers, timeout=60)
    resp.raise_for_status()

    resp_json = resp.json()
    translated_text = resp_json["choices"][0]["message"]["content"]
    usage_info = resp_json.get("usage", {})
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
    Given a list of page-level texts, chunk each page's text if needed.
    Returns a list of lists of chunks, where each sub-list corresponds
    to a particular page.

    Args:
        page_texts (List[str]): Raw text from each PDF page
        max_chunk_tokens (int): The chunk size (approx tokens) limit
        logger (logging.Logger): For debug information

    Returns:
        List[List[str]]: chunked_pages[page_idx] = list_of_chunk_strings
    """
    logger.info("Chunking all pages into smaller segments...")
    chunked_pages: List[List[str]] = []
    for idx, text in enumerate(page_texts):
        if text.strip():
            chunks = chunk_text(text, max_chunk_tokens)
            logger.debug(f"Page {idx+1} has {len(chunks)} chunk(s).")
        else:
            # Page is blank or has no extractable text
            chunks = []
            logger.debug(f"Page {idx+1} has no text.")
        chunked_pages.append(chunks)
    return chunked_pages


def translate_document_by_chunks(
    chunked_pages: List[List[str]],
    target_lang: str,
    model: str,
    concurrency: int,
    logger: logging.Logger
) -> Tuple[List[List[str]], List[List[Dict]]]:
    """
    Translate *all chunks from all pages* in parallel.

    Args:
        chunked_pages (List[List[str]]): Each sub-list is all chunks for that page
        target_lang (str): Target language code
        model (str): DeepSeek model
        concurrency (int): Max concurrency (threads)
        logger (logging.Logger): For diagnostic messages

    Returns:
        Tuple[List[List[str]], List[List[Dict]]]]:
            (
               per_page_translated_chunks,  # shape = same as chunked_pages
               per_page_usage_data          # shape = same as chunked_pages
            )
    """
    # Flatten all chunks for concurrency
    # We'll keep an index so we can place results in correct order
    tasks = []  # (page_idx, chunk_idx, chunk_text)
    for i, page_chunks in enumerate(chunked_pages):
        for j, chunk_text_ in enumerate(page_chunks):
            tasks.append((i, j, chunk_text_))

    # Prepare the structures to hold the results
    num_pages = len(chunked_pages)
    per_page_translated_chunks: List[List[Optional[str]]] = [
        [None] * len(chunked_pages[i]) for i in range(num_pages)
    ]
    per_page_usage_data: List[List[Optional[Dict]]] = [
        [None] * len(chunked_pages[i]) for i in range(num_pages)
    ]

    logger.info(
        f"Submitting {len(tasks)} total chunk(s) across {num_pages} page(s) "
        f"to a thread pool of size={concurrency}."
    )

    def worker(page_idx: int, chunk_idx: int, chunk_text_: str):
        return translate_text(chunk_text_, target_lang, model, logger)

    # Translate all chunks in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(worker, p, c, txt): (p, c)
            for (p, c, txt) in tasks
        }

        if TQDM_AVAILABLE:
            future_iter = tqdm(
                as_completed(future_to_task),
                total=len(tasks),
                desc="Translating chunks"
            )
        else:
            future_iter = as_completed(future_to_task)

        for future in future_iter:
            p_idx, c_idx = future_to_task[future]
            try:
                translated_chunk, usage_info = future.result()
                per_page_translated_chunks[p_idx][c_idx] = translated_chunk
                per_page_usage_data[p_idx][c_idx] = usage_info
            except Exception as e:
                logger.error(
                    f"Chunk (page={p_idx+1}, chunk={c_idx+1}) failed: {e}"
                )
                # Store placeholders so we don't lose the structure
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
    Aggregate usage stats across all pages and chunks.

    Args:
        usage_data (List[List[Optional[Dict]]]): 
            usage_data[page_idx] = list of usage dicts for each chunk

    Returns:
        Dict[str, int]: {
            'prompt_tokens_total': ...,
            'completion_tokens_total': ...,
            'total_tokens_total': ...
        }
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
    Estimate cost for DeepSeek usage, given model, tokens, and cache scenario.

    Args:
        model (str): e.g. 'deepseek-chat', 'deepseek-reasoner'
        prompt_tokens (int)
        completion_tokens (int)
        cache (str): either 'hit' or 'miss'

    Returns:
        float: Cost in USD
    """
    if model not in DEEPSEEK_PRICING:
        raise ValueError(f"Unknown model '{model}'. Check DEEPSEEK_PRICING.")
    if cache not in ("hit", "miss"):
        raise ValueError(f"cache must be 'hit' or 'miss'. Received '{cache}'.")

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
    Main CLI entry point for PDF translation using DeepSeek.

    Steps:
      1) Validate file format
      2) Convert doc to PDF (if needed)
      3) Extract text from each page (in memory)
      4) Chunk each page's text
      5) Translate all chunks in parallel
      6) Reassemble & save final translation
      7) Compute usage & cost
    """
    parser = argparse.ArgumentParser(
        description="Translate PDF documents via DeepSeek API (faster chunk-level concurrency)."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input document file (pdf/doc/docx/odt).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target language code (e.g. 'pt', 'en').",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        choices=["deepseek-chat", "deepseek-reasoner"],
        help="DeepSeek model. Default: 'deepseek-chat'.",
    )
    parser.add_argument(
        "--cache",
        default="miss",
        choices=["hit", "miss"],
        help="Assume input tokens are cache HIT or MISS. Default: 'miss'.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=2000,
        help="Approx. max tokens allowed per chunk. Default=2000.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel requests for chunk translation. Default=4.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger(trace=args.trace)
    
    file_path = Path(args.file).resolve()
    try:
        # 1) Validate & Convert to PDF (if needed)
        validate_file(file_path, logger=logger)
        pdf_path = convert_to_pdf(file_path, logger=logger)

        # 2) Extract text from each page in memory
        page_texts = extract_all_page_texts(pdf_path, logger=logger)
        num_pages = len(page_texts)
        if num_pages == 0:
            raise ValueError(f"'{pdf_path.name}' has 0 pages with extractable text.")

        # 3) Chunk each page
        chunked_pages = chunk_all_pages(page_texts, max_chunk_tokens=args.max_chunk_tokens, logger=logger)

        # 4) Translate all chunks in parallel
        logger.info(
            f"Starting translations for {num_pages} page(s) with concurrency={args.concurrency} "
            f"and max_chunk_tokens={args.max_chunk_tokens}."
        )
        per_page_translated_chunks, per_page_usage_data = translate_document_by_chunks(
            chunked_pages=chunked_pages,
            target_lang=args.target,
            model=args.model,
            concurrency=args.concurrency,
            logger=logger
        )

        # 5) Reassemble final text
        #    For each page, join the translated chunks
        output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"
        with open(output_file, "w", encoding="utf-8") as out:
            for i, translated_chunks in enumerate(per_page_translated_chunks):
                # Join all chunk translations
                page_translation = "\n\n".join(translated_chunks)
                out.write(page_translation)
                # Add a separator for the next page
                if i < num_pages - 1:
                    out.write("\n\n<!-- NEXT PAGE -->\n\n")

        logger.info(f"Translation complete. Output saved to '{output_file}'")

        # 6) Compute usage & cost
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

        logger.info("Usage Summary:")
        logger.info(f"  Prompt Tokens:     {total_prompt}")
        logger.info(f"  Completion Tokens: {total_completion}")
        logger.info(f"  Overall Tokens:    {total_tokens}")
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



# import os
# import sys
# import logging
# import subprocess
# import argparse
# import requests
# import pdfplumber
# import PyPDF2
# import math

# from pathlib import Path
# from typing import List, Dict, Tuple, Optional
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # For advanced retry logic (handles transient API errors)
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Optional progress bar; install via: pip install tqdm
# try:
#     from tqdm import tqdm
#     TQDM_AVAILABLE = True
# except ImportError:
#     TQDM_AVAILABLE = False

# # ------------------------------------------------------------------------
# # Global Configuration
# # ------------------------------------------------------------------------
# API_KEY = os.getenv("DEEPSEEK_API_KEY")
# if not API_KEY:
#     raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

# BASE_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

# SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.doc,.docx,.odt").split(",")
# TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
# OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

# TEMP_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Pricing table for DeepSeek (per 1M tokens), simplified
# DEEPSEEK_PRICING = {
#     "deepseek-chat": {
#         "cache_hit":  {"input": 0.07,  "output": 1.10},
#         "cache_miss": {"input": 0.14,  "output": 1.10},
#     },
#     "deepseek-reasoner": {
#         "cache_hit":  {"input": 0.14,  "output": 2.19},
#         "cache_miss": {"input": 0.55,  "output": 2.19},
#     }
# }


# def setup_logger(trace: bool) -> logging.Logger:
#     """
#     Sets up and returns a logger instance with console output.
    
#     Args:
#         trace (bool): If True, set logger level to DEBUG, else INFO.
    
#     Returns:
#         logging.Logger: Configured logger.
#     """
#     level = logging.DEBUG if trace else logging.INFO
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level)

#     # If there are no handlers, add console handler:
#     if not logger.handlers:
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setLevel(level)
#         formatter = logging.Formatter(
#             "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
#         )
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)

#     return logger


# # ------------------------------------------------------------------------
# # File Validation & Conversion
# # ------------------------------------------------------------------------
# def validate_file(file_path: Path, logger: logging.Logger) -> None:
#     """
#     Validate if a file exists and is in a supported format.

#     Args:
#         file_path (Path): Path to the file to validate
#         logger (logging.Logger): Logger for diagnostic messages

#     Raises:
#         FileNotFoundError: If the file does not exist
#         ValueError: If the file format is not supported
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"File '{file_path}' not found.")

#     if file_path.suffix.lower() not in SUPPORTED_FORMATS:
#         allowed_formats = ", ".join(SUPPORTED_FORMATS)
#         raise ValueError(
#             f"Format '{file_path.suffix}' not supported. "
#             f"Allowed: {allowed_formats}"
#         )
#     logger.debug(f"File '{file_path.name}' is validated and supported.")


# def convert_to_pdf(file_path: Path, logger: logging.Logger) -> Path:
#     """
#     Convert a document to PDF format if needed using `unoconv`.

#     Args:
#         file_path (Path): Path to the input file
#         logger (logging.Logger): Logger for diagnostic messages

#     Returns:
#         Path: Path to the converted PDF file

#     Raises:
#         RuntimeError: If the `unoconv` command fails
#     """
#     if file_path.suffix.lower() == ".pdf":
#         logger.info(f"'{file_path.name}' is already a PDF. No conversion needed.")
#         return file_path

#     output_path = TEMP_DIR / f"{file_path.stem}.pdf"
#     logger.info(f"Converting '{file_path.name}' to PDF...")

#     try:
#         result = subprocess.run(
#             ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
#             check=True, capture_output=True
#         )
#         # Although check=True raises CalledProcessError for non-zero exit,
#         # we double check the return code:
#         if result.returncode != 0:
#             raise RuntimeError(f"unoconv failed (code {result.returncode}).")
#     except subprocess.CalledProcessError as e:
#         err_msg = e.stderr.decode().strip() if e.stderr else "Unknown error."
#         logger.error(f"Failed converting '{file_path}'. Error:\n{err_msg}")
#         raise RuntimeError(f"Failed to convert '{file_path}' to PDF.") from e

#     logger.info(f"Conversion complete. PDF saved at '{output_path}'.")
#     return output_path


# def split_pdf(pdf_path: Path, logger: logging.Logger) -> List[Path]:
#     """
#     Split a PDF document into individual pages. Each page is saved as
#     a separate PDF file in a 'pages' subdirectory of TEMP_DIR.

#     Args:
#         pdf_path (Path): Path to the PDF file to split
#         logger (logging.Logger): Logger for diagnostic messages

#     Returns:
#         List[Path]: List of per-page PDF paths
#     """
#     pages_dir = TEMP_DIR / "pages"
#     pages_dir.mkdir(exist_ok=True)

#     logger.info(f"Splitting '{pdf_path.name}' into pages...")
#     page_paths: List[Path] = []

#     with open(pdf_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for i, page in enumerate(reader.pages, start=1):
#             output_path = pages_dir / f"page_{i}.pdf"
#             writer = PyPDF2.PdfWriter()
#             writer.add_page(page)

#             with open(output_path, "wb") as out:
#                 writer.write(out)
#             page_paths.append(output_path)

#     logger.info(f"Splitting complete. Generated {len(page_paths)} page(s).")
#     return page_paths


# # ------------------------------------------------------------------------
# # Token Handling / Chunk Logic
# # ------------------------------------------------------------------------
# def approximate_token_count(text: str) -> int:
#     """
#     Estimate the token count of a text. 
#     For production, consider a dedicated tokenizer (e.g. 'tiktoken').

#     Args:
#         text (str): The text to analyze

#     Returns:
#         int: Estimated token count
#     """
#     # A simplistic approach: assume ~4 characters per token
#     return math.ceil(len(text) / 4)


# def chunk_text(text: str, max_chunk_tokens: int) -> List[str]:
#     """
#     Safely chunk the text into segments that are each <= max_chunk_tokens.

#     Args:
#         text (str): Original text to be chunked
#         max_chunk_tokens (int): Max tokens allowed per chunk

#     Returns:
#         List[str]: List of text chunks, each within the token limit
#     """
#     lines = text.split("\n")
#     chunks = []
#     current_chunk: List[str] = []
#     current_count = 0

#     for line in lines:
#         line_tokens = approximate_token_count(line)
        
#         # If line alone is too big, forcibly break into sub-chunks by words
#         if line_tokens >= max_chunk_tokens:
#             # Push whatever is in current_chunk first
#             if current_chunk:
#                 chunks.append("\n".join(current_chunk))
#                 current_chunk = []
#                 current_count = 0

#             words = line.split()
#             subchunk: List[str] = []
#             subcount = 0
#             for word in words:
#                 wcount = approximate_token_count(word + " ")
#                 if subcount + wcount <= max_chunk_tokens:
#                     subchunk.append(word)
#                     subcount += wcount
#                 else:
#                     chunks.append(" ".join(subchunk))
#                     subchunk = [word]
#                     subcount = wcount

#             # Append remainder
#             if subchunk:
#                 chunks.append(" ".join(subchunk))
#             continue

#         # If adding this line exceeds limit, finalize current_chunk
#         if current_count + line_tokens > max_chunk_tokens:
#             chunks.append("\n".join(current_chunk))
#             current_chunk = [line]
#             current_count = line_tokens
#         else:
#             current_chunk.append(line)
#             current_count += line_tokens

#     # Append any leftover
#     if current_chunk:
#         chunks.append("\n".join(current_chunk))

#     return chunks


# # ------------------------------------------------------------------------
# # DeepSeek Translation API
# # ------------------------------------------------------------------------
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def translate_text(
#     text: str, 
#     target_lang: str, 
#     model: str, 
#     logger: logging.Logger
# ) -> Tuple[str, Dict]:
#     """
#     Translate a block of text using DeepSeek. Returns (translated_text, usage_info).

#     Args:
#         text (str): The raw text to translate
#         target_lang (str): The target language code (e.g. 'pt', 'en')
#         model (str): DeepSeek model name (e.g. 'deepseek-chat', 'deepseek-reasoner')
#         logger (logging.Logger): Logger for diagnostic messages

#     Returns:
#         Tuple[str, Dict]: (translated_text, usage_info)

#     Raises:
#         requests.exceptions.RequestException: If the API request fails
#     """
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json",
#         "Accept": "application/json"
#     }

#     # The system instruction ensures the translator style and formatting
#     data: Dict = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a technical translator specialized in academic/scientific content.\n"
#                     "Instructions:\n"
#                     "1. All equations must be in LaTeX.\n"
#                     "2. All Greek letters and equations must be in LaTeX blocks (e.g. '\\[ x+2\\alpha \\]').\n"
#                     "3. Maintain all Markdown formatting.\n"
#                     "4. Keep proper names, citations, references.\n"
#                     "5. Ensure terminology consistency.\n"
#                     "6. Preserve paragraph structure and line breaks.\n"
#                     "7. Return only translated text, no explanations.\n"
#                     "8. Format tables in Markdown.\n"
#                     "9. For images, place its legend in text."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"Translate to {target_lang}:\n{text}"
#             }
#         ],
#         "temperature": 0,
#         "max_tokens": 4000
#     }

#     logger.debug(f"Sending API request to DeepSeek model={model}, target_lang={target_lang}")
#     resp = requests.post(BASE_URL, json=data, headers=headers, timeout=60)
#     resp.raise_for_status()

#     resp_json = resp.json()
#     translated_text = resp_json["choices"][0]["message"]["content"]
#     usage_info = resp_json.get("usage", {})
#     logger.debug("API response received successfully.")
#     return translated_text, usage_info


# def translate_page(
#     page_path: Path,
#     target_lang: str,
#     model: str,
#     max_chunk_tokens: int,
#     logger: logging.Logger
# ) -> Tuple[str, List[Dict]]:
#     """
#     Translate one PDF page. Splits the page into chunks if needed.
#     Returns:
#       - combined translation of the entire page
#       - a list of usage dictionaries (one per chunk)

#     Args:
#         page_path (Path): Path to a single-page PDF
#         target_lang (str): Target language
#         model (str): DeepSeek model name
#         max_chunk_tokens (int): Approx. max tokens per chunk
#         logger (logging.Logger): Logger for diagnostic messages

#     Returns:
#         Tuple[str, List[Dict]]:
#             (page_translation, usage_list)
#     """
#     if not page_path.exists():
#         raise FileNotFoundError(f"Page file '{page_path}' not found.")

#     with pdfplumber.open(page_path) as pdf:
#         raw_text = pdf.pages[0].extract_text() or ""

#     if not raw_text.strip():
#         logger.info(f"No text on page '{page_path.name}'. Skipping.")
#         return "", []

#     chunks = chunk_text(raw_text, max_chunk_tokens)
#     logger.debug(f"Page '{page_path.name}' split into {len(chunks)} chunk(s).")

#     page_translation_parts: List[str] = []
#     usage_list: List[Dict] = []

#     for i, chunk_text_segment in enumerate(chunks, start=1):
#         try:
#             translated_chunk, usage_info = translate_text(
#                 chunk_text_segment, target_lang, model, logger
#             )
#             page_translation_parts.append(translated_chunk)
#             usage_list.append(usage_info)
#         except Exception as exc:
#             logger.error(
#                 f"Chunk {i}/{len(chunks)} of page '{page_path.name}' failed: {exc}"
#             )
#             page_translation_parts.append("[TRANSLATION FAILED FOR THIS CHUNK]")
#             usage_list.append({})

#     full_page_translation = "\n\n".join(page_translation_parts)
#     return full_page_translation, usage_list


# def translate_pages(
#     page_paths: List[Path],
#     target_lang: str,
#     model: str,
#     max_chunk_tokens: int,
#     concurrency: int,
#     logger: logging.Logger
# ) -> List[Tuple[str, List[Dict]]]:
#     """
#     Translate multiple PDF pages concurrently. Each page might produce multiple
#     chunks internally, but concurrency is at the page level.

#     Args:
#         page_paths (List[Path]): A list of single-page PDF file paths
#         target_lang (str): The target language code
#         model (str): DeepSeek model to use
#         max_chunk_tokens (int): Approx. max tokens per chunk
#         concurrency (int): Number of pages to process concurrently
#         logger (logging.Logger): Logger for diagnostic messages

#     Returns:
#         List[Tuple[str, List[Dict]]]: A list of (page_translation, usage_list) pairs.
#                                        The order of pages is preserved.
#     """
#     results = [("", []) for _ in range(len(page_paths))]

#     # Prepare the executor
#     with ThreadPoolExecutor(max_workers=concurrency) as executor:
#         future_to_index = {
#             executor.submit(
#                 translate_page, p, target_lang, model, max_chunk_tokens, logger
#             ): i
#             for i, p in enumerate(page_paths)
#         }

#         # If tqdm is installed, show a progress bar
#         future_iter = (
#             tqdm(as_completed(future_to_index), total=len(page_paths), desc="Translating pages")
#             if TQDM_AVAILABLE
#             else as_completed(future_to_index)
#         )

#         for future in future_iter:
#             idx = future_to_index[future]
#             try:
#                 page_translation, usage_list = future.result()
#                 results[idx] = (page_translation, usage_list)
#             except Exception as e:
#                 logger.error(f"Page {idx+1} translation failed entirely: {e}")
#                 results[idx] = ("[PAGE TRANSLATION FAILED]", [])

#     return results


# # ------------------------------------------------------------------------
# # Usage & Cost Calculation
# # ------------------------------------------------------------------------
# def compute_usage_stats(
#     per_page_results: List[Tuple[str, List[Dict]]]
# ) -> Dict[str, int]:
#     """
#     Aggregate usage stats across all pages and chunks.

#     Args:
#         per_page_results (List[Tuple[str, List[Dict]]]):
#             Each element is (page_text, [usage_dict_chunk1, usage_dict_chunk2, ...])

#     Returns:
#         Dict[str, int]: {
#             'prompt_tokens_total': ...,
#             'completion_tokens_total': ...,
#             'total_tokens_total': ...
#         }
#     """
#     prompt_sum = 0
#     completion_sum = 0
#     total_sum = 0

#     for _, usage_list in per_page_results:
#         for usage_info in usage_list:
#             p = usage_info.get("prompt_tokens", 0)
#             c = usage_info.get("completion_tokens", 0)
#             t = p + c
#             prompt_sum += p
#             completion_sum += c
#             total_sum += t

#     return {
#         "prompt_tokens_total": prompt_sum,
#         "completion_tokens_total": completion_sum,
#         "total_tokens_total": total_sum
#     }


# def estimate_deepseek_cost(
#     model: str,
#     prompt_tokens: int,
#     completion_tokens: int,
#     cache: str
# ) -> float:
#     """
#     Estimate cost for DeepSeek usage, given model, tokens, and cache scenario.

#     Args:
#         model (str): e.g. 'deepseek-chat', 'deepseek-reasoner'
#         prompt_tokens (int)
#         completion_tokens (int)
#         cache (str): either 'hit' or 'miss'

#     Returns:
#         float: Cost in USD
#     """
#     if model not in DEEPSEEK_PRICING:
#         raise ValueError(f"Unknown model '{model}'. Check DEEPSEEK_PRICING.")
#     if cache not in ("hit", "miss"):
#         raise ValueError(f"cache must be 'hit' or 'miss'. Received '{cache}'.")

#     pricing = DEEPSEEK_PRICING[model][f"cache_{cache}"]
#     cost_input_per_mil = pricing["input"]
#     cost_output_per_mil = pricing["output"]

#     input_millions = prompt_tokens / 1_000_000
#     output_millions = completion_tokens / 1_000_000

#     cost_input = input_millions * cost_input_per_mil
#     cost_output = output_millions * cost_output_per_mil
#     return cost_input + cost_output


# # ------------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------------
# def main() -> None:
#     """
#     Main CLI entry point for PDF translation using DeepSeek.

#     Steps:
#       1) Validate file format
#       2) Convert doc to PDF (if needed)
#       3) Split PDF into pages
#       4) Translate pages -> handle chunking, concurrency
#       5) Combine translations
#       6) Compute usage & cost
#       7) (Optional) Cleanup page files
#     """
#     parser = argparse.ArgumentParser(
#         description="Translate PDF documents via DeepSeek API."
#     )
#     parser.add_argument(
#         "--file",
#         required=True,
#         help="Path to the input document file (pdf/doc/docx/odt).",
#     )
#     parser.add_argument(
#         "--target",
#         required=True,
#         help="Target language code (e.g. 'pt', 'en').",
#     )
#     parser.add_argument(
#         "--model",
#         default="deepseek-chat",
#         choices=["deepseek-chat", "deepseek-reasoner"],
#         help="DeepSeek model. Default: 'deepseek-chat'.",
#     )
#     parser.add_argument(
#         "--cache",
#         default="miss",
#         choices=["hit", "miss"],
#         help="Assume input tokens are cache HIT or MISS. Default: 'miss'.",
#     )
#     parser.add_argument(
#         "--max-chunk-tokens",
#         type=int,
#         default=2000,
#         help="Approx. max tokens allowed per chunk. Default=2000.",
#     )
#     parser.add_argument(
#         "--concurrency",
#         type=int,
#         default=1,
#         help="Number of pages to translate concurrently. Default=1.",
#     )
#     parser.add_argument(
#         "--clean",
#         action="store_true",
#         default=False,
#         help="If set, remove split PDF pages after completion.",
#     )
#     parser.add_argument(
#         "--trace",
#         action="store_true",
#         default=False,
#         help="Enable DEBUG-level logging.",
#     )
#     args = parser.parse_args()

#     # Set up logger
#     logger = setup_logger(trace=args.trace)
    
#     file_path = Path(args.file).resolve()

#     try:
#         # 1) Validate & Convert
#         validate_file(file_path, logger=logger)
#         pdf_path = convert_to_pdf(file_path, logger=logger)

#         # 2) Split into pages
#         page_paths = split_pdf(pdf_path, logger=logger)
#         total_pages = len(page_paths)
#         if total_pages == 0:
#             raise ValueError(f"'{pdf_path.name}' has 0 pages to translate.")

#         logger.info(
#             f"Translating {total_pages} page(s) to '{args.target}' "
#             f"using model='{args.model}', concurrency={args.concurrency}, "
#             f"max_chunk_tokens={args.max_chunk_tokens}."
#         )

#         # 3) Translate pages
#         results = translate_pages(
#             page_paths=page_paths,
#             target_lang=args.target,
#             model=args.model,
#             max_chunk_tokens=args.max_chunk_tokens,
#             concurrency=args.concurrency,
#             logger=logger
#         )

#         # 4) Combine translations
#         page_texts = [r[0] for r in results]
#         output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"
#         with open(output_file, "w", encoding="utf-8") as out:
#             out.write("\n\n<!-- NEXT PAGE -->\n\n".join(page_texts))
#         logger.info(f"Translation complete. Output saved to '{output_file}'")

#         # 5) Compute usage & cost
#         usage_agg = compute_usage_stats(results)
#         total_prompt = usage_agg["prompt_tokens_total"]
#         total_completion = usage_agg["completion_tokens_total"]
#         total_tokens = usage_agg["total_tokens_total"]
#         estimated_cost = estimate_deepseek_cost(
#             model=args.model,
#             prompt_tokens=total_prompt,
#             completion_tokens=total_completion,
#             cache=args.cache
#         )

#         logger.info("Usage Summary:")
#         logger.info(f"  Prompt Tokens:     {total_prompt}")
#         logger.info(f"  Completion Tokens: {total_completion}")
#         logger.info(f"  Overall Tokens:    {total_tokens}")
#         logger.info(f"Estimated cost: ${estimated_cost:.4f}")

#         # 6) Optional cleanup of page files
#         if args.clean:
#             for p in page_paths:
#                 try:
#                     p.unlink(missing_ok=True)
#                 except Exception as e:
#                     logger.warning(f"Failed to remove '{p}': {e}")

#     except Exception as e:
#         logger.error(f"Translation failed: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()



# import os
# import logging
# import subprocess
# import argparse
# import requests
# import pdfplumber
# import PyPDF2
# import math

# from pathlib import Path
# from typing import List, Dict, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # For advanced retry logic (handles transient API errors)
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Optional progress bar; install via: pip install tqdm
# try:
#     from tqdm import tqdm
#     TQDM_AVAILABLE = True
# except ImportError:
#     TQDM_AVAILABLE = False

# # ------------------------------------------------------------------------------
# # Global Configuration
# # ------------------------------------------------------------------------------
# API_KEY = os.getenv('DEEPSEEK_API_KEY')
# if not API_KEY:
#     raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

# BASE_URL = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/v1/chat/completions")

# SUPPORTED_FORMATS = os.getenv('SUPPORTED_FORMATS', '.pdf,.doc,.docx,.odt').split(',')
# TEMP_DIR = Path(os.getenv('TEMP_DIR', 'temp'))
# OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'output'))

# TEMP_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Pricing table for DeepSeek (per 1M tokens), simplified:
# DEEPSEEK_PRICING = {
#     "deepseek-chat": {
#         "cache_hit":  {"input": 0.07,  "output": 1.10},
#         "cache_miss": {"input": 0.14,  "output": 1.10},
#     },
#     "deepseek-reasoner": {
#         "cache_hit":  {"input": 0.14,  "output": 2.19},
#         "cache_miss": {"input": 0.55,  "output": 2.19},
#     }
# }


# # ------------------------------------------------------------------------------
# # Logging Control
# # ------------------------------------------------------------------------------
# def set_logging_level(trace: bool) -> None:
#     """
#     Enable or disable tracing (debug-level logging).

#     Args:
#         trace (bool): If True, set log level to DEBUG; else INFO.
#     """
#     level = logging.DEBUG if trace else logging.INFO
#     logging.basicConfig(
#         level=level,
#         format='[%(asctime)s] %(levelname)s: %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )


# # ------------------------------------------------------------------------------
# # File Validation & Conversion
# # ------------------------------------------------------------------------------
# def validate_file(file_path: Path) -> None:
#     """
#     Validate if a file exists and is in a supported format.

#     Args:
#         file_path (Path): Path to the file to validate

#     Raises:
#         FileNotFoundError: If the file does not exist
#         ValueError: If the file format is not supported
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"File '{file_path}' not found.")
    
#     if file_path.suffix.lower() not in SUPPORTED_FORMATS:
#         raise ValueError(
#             f"Format '{file_path.suffix}' not supported. "
#             f"Allowed: {', '.join(SUPPORTED_FORMATS)}"
#         )


# def convert_to_pdf(file_path: Path) -> Path:
#     """
#     Convert a document to PDF format if needed.

#     Args:
#         file_path (Path): Path to the input file

#     Returns:
#         Path: Path to the converted PDF file

#     Raises:
#         RuntimeError: If the `unoconv` command fails
#     """
#     if file_path.suffix.lower() == '.pdf':
#         logging.info(f"'{file_path.name}' is already PDF. No conversion needed.")
#         return file_path
    
#     output_path = TEMP_DIR / f"{file_path.stem}.pdf"
#     logging.info(f"Converting '{file_path.name}' to PDF...")

#     try:
#         result = subprocess.run(
#             ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
#             check=True, capture_output=True
#         )
#         if result.returncode != 0:
#             raise RuntimeError(f"unoconv failed (code {result.returncode}).")
#     except subprocess.CalledProcessError as e:
#         err_msg = e.stderr.decode().strip() if e.stderr else "Unknown error."
#         logging.error(f"Failed converting '{file_path}'. Error:\n{err_msg}")
#         raise RuntimeError(f"Failed to convert '{file_path}' to PDF.") from e

#     logging.info(f"Conversion complete. PDF saved at '{output_path}'.")
#     return output_path


# def split_pdf(pdf_path: Path) -> List[Path]:
#     """
#     Split a PDF document into individual pages. Each page is saved as
#     a separate PDF file in a 'pages' subdirectory of TEMP_DIR.

#     Args:
#         pdf_path (Path): Path to the PDF file to split

#     Returns:
#         List[Path]: List of page PDF paths
#     """
#     pages_dir = TEMP_DIR / 'pages'
#     pages_dir.mkdir(exist_ok=True)

#     logging.info(f"Splitting '{pdf_path.name}' into pages...")
#     page_paths: List[Path] = []

#     with open(pdf_path, 'rb') as f:
#         reader = PyPDF2.PdfReader(f)
#         for i, page in enumerate(reader.pages, start=1):
#             output_path = pages_dir / f"page_{i}.pdf"
#             writer = PyPDF2.PdfWriter()
#             writer.add_page(page)
#             with open(output_path, 'wb') as out:
#                 writer.write(out)
#             page_paths.append(output_path)

#     logging.info(f"Splitting complete. Generated {len(page_paths)} page(s).")
#     return page_paths


# # ------------------------------------------------------------------------------
# # Token Handling / Chunk Logic
# # ------------------------------------------------------------------------------
# def approximate_token_count(text: str) -> int:
#     """
#     Estimate the token count of a text. In production, consider a dedicated
#     tokenizer (e.g., 'tiktoken') for more accurate counts.

#     Args:
#         text (str): The text to analyze

#     Returns:
#         int: Estimated token count
#     """
#     # Basic approximation: 1 token ~= ~4 chars. 
#     # Or use: len(text.split()) * 1.3 as a naive approach.
#     # Here, we do a simplistic approach: ~4 chars per token
#     # This is purely an approximation; adapt as needed.
#     return math.ceil(len(text) / 4)


# def chunk_text(text: str, max_chunk_tokens: int) -> List[str]:
#     """
#     Safely chunk the text into segments that are each <= max_chunk_tokens.

#     Args:
#         text (str): Original text to be chunked
#         max_chunk_tokens (int): Max tokens allowed per chunk

#     Returns:
#         List[str]: List of text chunks, each within the token limit
#     """
#     lines = text.split("\n")
#     chunks = []
#     current_chunk = []

#     current_count = 0
#     for line in lines:
#         line_tokens = approximate_token_count(line)
#         # If line alone too big, forcibly chunk it at a smaller granularity
#         # This approach is simplistic. For more accuracy, you'd break by word.
#         if line_tokens >= max_chunk_tokens:
#             # If we have content in current_chunk, store it first
#             if current_chunk:
#                 chunks.append("\n".join(current_chunk))
#                 current_chunk = []
#                 current_count = 0

#             # Break the line itself into subchunks if needed
#             words = line.split()
#             subchunk = []
#             subcount = 0
#             for word in words:
#                 wcount = approximate_token_count(word + " ")
#                 if subcount + wcount <= max_chunk_tokens:
#                     subchunk.append(word)
#                     subcount += wcount
#                 else:
#                     chunks.append(" ".join(subchunk))
#                     subchunk = [word]
#                     subcount = wcount

#             if subchunk:
#                 chunks.append(" ".join(subchunk))
#             continue

#         # If adding this line exceeds the limit, finalize current_chunk
#         if current_count + line_tokens > max_chunk_tokens:
#             chunks.append("\n".join(current_chunk))
#             current_chunk = [line]
#             current_count = line_tokens
#         else:
#             current_chunk.append(line)
#             current_count += line_tokens

#     # Append remainder
#     if current_chunk:
#         chunks.append("\n".join(current_chunk))

#     return chunks


# # ------------------------------------------------------------------------------
# # API Translation
# # ------------------------------------------------------------------------------
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def translate_text(text: str, target_lang: str, model: str) -> Tuple[str, Dict]:
#     """
#     Translate a block of text using DeepSeek. Returns (translated_text, usage_info).

#     Args:
#         text (str): The raw text to translate
#         target_lang (str): The target language code (e.g. 'pt', 'en')
#         model (str): DeepSeek model name (e.g. 'deepseek-chat', 'deepseek-reasoner')

#     Returns:
#         Tuple[str, Dict]: (translated_text, usage_info)

#     Raises:
#         requests.exceptions.RequestException: If the API request fails
#         Exception: Other unexpected errors
#     """
#     headers = {
#         'Authorization': f'Bearer {API_KEY}',
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }

#     data: Dict = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a technical translator specialized in academic/scientific content. "
#                     "Instructions:\n"
#                     "1. All equations must be in LaTeX.\n"
#                     "2. All Greek letters and equations must be in LaTeX blocks (e.g. '\\[x+2\\alpha\\]').\n"
#                     "3. Maintain all Markdown formatting.\n"
#                     "4. Keep proper names, citations, references.\n"
#                     "5. Ensure terminology consistency.\n"
#                     "6. Preserve paragraph structure and line breaks.\n"
#                     "7. Return only translated text, no explanations.\n"
#                     "8. Format tables in Markdown.\n"
#                     "9. For images, place its legend in text."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"Translate to {target_lang}:\n{text}"
#             }
#         ],
#         "temperature": 0,
#         "max_tokens": 4000
#     }

#     resp = requests.post(BASE_URL, json=data, headers=headers, timeout=60)
#     resp.raise_for_status()

#     resp_json = resp.json()
#     translated_text = resp_json["choices"][0]["message"]["content"]
#     usage_info = resp_json.get("usage", {})
#     return translated_text, usage_info


# def translate_page(page_path: Path, target_lang: str, model: str, max_chunk_tokens: int) -> Tuple[str, List[Dict]]:
#     """
#     Translate one PDF page. Splits the page into chunks if needed. Returns:
#       - combined translation of the entire page
#       - a list of usage dictionaries (one per chunk)

#     Args:
#         page_path (Path): Path to a single-page PDF
#         target_lang (str): Target language
#         model (str): DeepSeek model name
#         max_chunk_tokens (int): Approx. max tokens per chunk

#     Returns:
#         Tuple[str, List[Dict]]:
#             page_translation, usage_list
#             (usage_list has usage dict for each chunk)
#     """
#     if not page_path.exists():
#         raise FileNotFoundError(f"Page file '{page_path}' not found.")

#     # Extract page text
#     with pdfplumber.open(page_path) as pdf:
#         raw_text = pdf.pages[0].extract_text() or ""
#         if not raw_text.strip():
#             logging.info(f"No text on page '{page_path.name}'. Skipping.")
#             return "", []

#     # Chunk the text if it's large
#     chunks = chunk_text(raw_text, max_chunk_tokens)

#     page_translation_parts: List[str] = []
#     usage_list: List[Dict] = []

#     for i, chunk_text_segment in enumerate(chunks, start=1):
#         try:
#             translated_chunk, usage_info = translate_text(chunk_text_segment, target_lang, model)
#             page_translation_parts.append(translated_chunk)
#             usage_list.append(usage_info)
#         except Exception as exc:
#             logging.error(f"Chunk {i}/{len(chunks)} of page '{page_path.name}' failed: {exc}")
#             # We add a placeholder to avoid losing page structure
#             page_translation_parts.append("[TRANSLATION FAILED FOR THIS CHUNK]")
#             usage_list.append({})

#     # Combine all chunk translations for the final page text
#     full_page_translation = "\n\n".join(page_translation_parts)
#     return full_page_translation, usage_list


# def translate_pages(
#     page_paths: List[Path],
#     target_lang: str,
#     model: str,
#     max_chunk_tokens: int,
#     concurrency: int
# ) -> List[Tuple[str, List[Dict]]]:
#     """
#     Translate multiple PDF pages concurrently. Each page might produce multiple
#     chunks internally, but concurrency is at the page level.

#     Args:
#         page_paths (List[Path]): A list of single-page PDF file paths
#         target_lang (str): The target language code
#         model (str): DeepSeek model to use
#         max_chunk_tokens (int): Approx. max tokens per chunk
#         concurrency (int): Number of pages to process concurrently

#     Returns:
#         List[Tuple[str, List[Dict]]]: A list of (page_translation, usage_list) pairs,
#         preserving the order of pages.
#     """
#     results = [("", []) for _ in range(len(page_paths))]

#     # Prepare the executor
#     with ThreadPoolExecutor(max_workers=concurrency) as executor:
#         future_to_index = {
#             executor.submit(translate_page, path, target_lang, model, max_chunk_tokens): i
#             for i, path in enumerate(page_paths)
#         }

#         # If tqdm is installed, show page-level progress
#         if TQDM_AVAILABLE:
#             future_iter = tqdm(as_completed(future_to_index), total=len(page_paths), desc="Translating pages")
#         else:
#             future_iter = as_completed(future_to_index)

#         for future in future_iter:
#             idx = future_to_index[future]
#             try:
#                 page_translation, usage_list = future.result()
#                 results[idx] = (page_translation, usage_list)
#             except Exception as e:
#                 logging.error(f"Page {idx+1} translation failed entirely: {e}")
#                 results[idx] = ("[PAGE TRANSLATION FAILED]", [])

#     return results


# # ------------------------------------------------------------------------------
# # Usage & Cost Calculation
# # ------------------------------------------------------------------------------
# def compute_usage_stats(
#     per_page_results: List[Tuple[str, List[Dict]]]
# ) -> Dict[str, int]:
#     """
#     Aggregate usage stats across all pages and chunks.

#     Args:
#         per_page_results (List[Tuple[str, List[Dict]]]):
#             Each element is (page_text, [usage_dict_chunk1, usage_dict_chunk2, ...])

#     Returns:
#         Dict[str, int]: e.g.
#             {
#               'prompt_tokens_total': ...,
#               'completion_tokens_total': ...,
#               'total_tokens_total': ...
#             }
#     """
#     prompt_sum, completion_sum, total_sum = 0, 0, 0

#     for _, usage_list in per_page_results:
#         for usage_info in usage_list:
#             p = usage_info.get('prompt_tokens', 0)
#             c = usage_info.get('completion_tokens', 0)
#             t = usage_info.get('total_tokens', p + c)
#             prompt_sum += p
#             completion_sum += c
#             total_sum += t

#     return {
#         'prompt_tokens_total': prompt_sum,
#         'completion_tokens_total': completion_sum,
#         'total_tokens_total': total_sum
#     }


# def estimate_deepseek_cost(
#     model: str,
#     prompt_tokens: int,
#     completion_tokens: int,
#     cache: str
# ) -> float:
#     """
#     Estimate cost for DeepSeek usage, given model, tokens, and cache scenario.

#     Args:
#         model (str): e.g. 'deepseek-chat', 'deepseek-reasoner'
#         prompt_tokens (int)
#         completion_tokens (int)
#         cache (str): either 'hit' or 'miss'

#     Returns:
#         float: Cost in USD
#     """
#     if model not in DEEPSEEK_PRICING:
#         raise ValueError(f"Unknown model '{model}'. Check DEEPSEEK_PRICING.")
#     if cache not in ('hit', 'miss'):
#         raise ValueError(f"cache must be 'hit' or 'miss'. Received '{cache}'.")

#     pricing = DEEPSEEK_PRICING[model][f"cache_{cache}"]
#     cost_input_per_mil = pricing["input"]
#     cost_output_per_mil = pricing["output"]

#     # Convert to millions
#     input_millions = prompt_tokens / 1_000_000
#     output_millions = completion_tokens / 1_000_000

#     cost_input = input_millions * cost_input_per_mil
#     cost_output = output_millions * cost_output_per_mil

#     return cost_input + cost_output


# # ------------------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------------------
# def main() -> None:
#     """
#     Main CLI entry point for PDF translation using DeepSeek.

#     Steps:
#       1) Validate file format
#       2) Convert doc to PDF (if needed)
#       3) Split PDF into pages
#       4) Translate pages -> handle chunking, concurrency
#       5) Combine translations
#       6) Compute usage & cost
#       7) (Optional) Cleanup page files
#     """
#     parser = argparse.ArgumentParser(description="Translate PDF documents via DeepSeek API.")
#     parser.add_argument("--file", required=True, help="Path to the input document file (pdf/doc/docx/odt).")
#     parser.add_argument("--target", required=True, help="Target language code (e.g. 'pt', 'en').")
#     parser.add_argument("--model", default="deepseek-chat",
#                         choices=["deepseek-chat", "deepseek-reasoner"],
#                         help="DeepSeek model. Default: 'deepseek-chat'.")
#     parser.add_argument("--cache", default="miss", choices=["hit", "miss"],
#                         help="Assume input tokens are cache HIT or MISS. Default: 'miss'.")
#     parser.add_argument("--max-chunk-tokens", type=int, default=2000,
#                         help="Approx. max tokens allowed per chunk. Default=2000.")
#     parser.add_argument("--concurrency", type=int, default=1,
#                         help="Number of pages to translate concurrently. Default=1.")
#     parser.add_argument("--clean", action="store_true", default=False,
#                         help="If set, remove split PDF pages after completion.")
#     parser.add_argument("--trace", action="store_true", default=False,
#                         help="Enable DEBUG-level logging.")
#     args = parser.parse_args()

#     # Set logging level
#     set_logging_level(args.trace)

#     file_path = Path(args.file).resolve()

#     try:
#         # 1) Validate & convert
#         validate_file(file_path)
#         pdf_path = convert_to_pdf(file_path)

#         # 2) Split
#         page_paths = split_pdf(pdf_path)
#         total_pages = len(page_paths)
#         if total_pages == 0:
#             raise ValueError(f"'{pdf_path.name}' has 0 pages to translate.")

#         # 3) Translate pages
#         logging.info(
#             f"Translating {total_pages} pages to '{args.target}' using model='{args.model}', "
#             f"concurrency={args.concurrency}, max_chunk_tokens={args.max_chunk_tokens}."
#         )

#         results = translate_pages(
#             page_paths,
#             target_lang=args.target,
#             model=args.model,
#             max_chunk_tokens=args.max_chunk_tokens,
#             concurrency=args.concurrency
#         )

#         # 4) Combine translations
#         page_texts = [r[0] for r in results]
#         output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"

#         with open(output_file, 'w', encoding='utf-8') as out:
#             out.write("\n\n<!-- NEXT PAGE -->\n\n".join(page_texts))

#         logging.info(f"Translation complete. Output saved to '{output_file}'")

#         # 5) Compute usage & cost
#         usage_agg = compute_usage_stats(results)
#         total_prompt = usage_agg['prompt_tokens_total']
#         total_completion = usage_agg['completion_tokens_total']
#         total_tokens = usage_agg['total_tokens_total']

#         estimated_cost = estimate_deepseek_cost(
#             model=args.model,
#             prompt_tokens=total_prompt,
#             completion_tokens=total_completion,
#             cache=args.cache
#         )

#         logging.info("Usage Summary:")
#         logging.info(f"  Prompt Tokens:     {total_prompt}")
#         logging.info(f"  Completion Tokens: {total_completion}")
#         logging.info(f"  Overall Tokens:    {total_tokens}")
#         logging.info(f"Estimated cost: ${estimated_cost:.4f}")

#         # 6) Optional cleanup
#         if args.clean:
#             for p in page_paths:
#                 try:
#                     p.unlink(missing_ok=True)
#                 except Exception as e:
#                     logging.warning(f"Failed to remove '{p}': {e}")

#     except Exception as e:
#         logging.error(f"Translation failed: {e}")
#         exit(1)


# if __name__ == "__main__":
#     main()






# import os
# import sys
# import time
# import PyPDF2
# import pdfplumber
# import requests
# from pathlib import Path
# from typing import Optional, Dict
# from tenacity import retry, stop_after_attempt, wait_exponential
# import logging

# # Configuration
# API_KEY = os.getenv('DEEPSEEK_API_KEY')
# if not API_KEY:
#     raise ValueError("DEEPSEEK_API_KEY environment variable not set")
# BASE_URL = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/v1/chat/completions")
# SUPPORTED_FORMATS = os.getenv('SUPPORTED_FORMATS', '.pdf,.doc,.docx,.odt').split(',')
# TEMP_DIR = Path(os.getenv('TEMP_DIR', 'temp'))
# INPUT_DIR = Path(os.getenv('INPUT_DIR', 'input'))
# OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'output'))

# def validate_file(file_path: Path) -> None:
#     """Validate if file exists and is in supported format.
    
#     Args:
#         file_path (Path): Path to the file to validate
        
#     Raises:
#         FileNotFoundError: If the file does not exist
#         ValueError: If the file format is not supported
        
#     Example:
#         >>> validate_file(Path('document.pdf'))
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"File {file_path} not found")
    
#     if file_path.suffix.lower() not in SUPPORTED_FORMATS:
#         raise ValueError(f"Format {file_path.suffix} not supported")

# def convert_to_pdf(file_path: Path) -> Path:
#     """Convert document to PDF format if needed.
    
#     If the file is already a PDF, returns the original path.
#     Otherwise converts the file to PDF using unoconv.
    
#     Args:
#         file_path (Path): Path to the input file
        
#     Returns:
#         Path: Path to the converted PDF file
        
#     Example:
#         >>> convert_to_pdf(Path('document.docx'))
#         Path('temp/document.pdf')
#     """
#     if file_path.suffix.lower() == '.pdf':
#         return file_path
    
#     output_path = TEMP_DIR / f"{file_path.stem}.pdf"
#     os.system(f"unoconv -f pdf -o {output_path} {file_path}")
#     return output_path

# def split_pdf(file_path: Path) -> list[Path]:
#     """Split PDF document into individual pages.
    
#     Args:
#         file_path (Path): Path to the PDF file to split
        
#     Returns:
#         list[Path]: List of paths to individual page PDFs
        
#     Example:
#         >>> split_pdf(Path('document.pdf'))
#         [Path('temp/pages/page_1.pdf'), Path('temp/pages/page_2.pdf')]
#     """
#     pages_dir = TEMP_DIR / 'pages'
#     pages_dir.mkdir(exist_ok=True)
    
#     with open(file_path, 'rb') as input_pdf:
#         reader = PyPDF2.PdfReader(input_pdf)
        
#         for i, page in enumerate(reader.pages):
#             output_path = pages_dir / f"page_{i+1}.pdf"
#             writer = PyPDF2.PdfWriter()
#             writer.add_page(page)
            
#             with open(output_path, 'wb') as output_pdf:
#                 writer.write(output_pdf)
    
#     return list(pages_dir.glob('*.pdf'))

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def translate_page(page_path: Path, target_lang: str) -> str:
#     """Translate a single PDF page using DeepSeek API.
    
#     Args:
#         page_path (Path): Path to the PDF page to translate
#         target_lang (str): Target language code (e.g., 'pt', 'en')
        
#     Returns:
#         str: Translated text content
        
#     Raises:
#         FileNotFoundError: If PDF page cannot be found
#         requests.exceptions.RequestException: If API request fails
#         Exception: For any other unexpected errors
        
#     Example:
#         >>> translate_page(Path('page_1.pdf'), 'pt')
#         'Texto traduzido...'
#     """
#     try:
#         with pdfplumber.open(page_path) as pdf:
#             text = pdf.pages[0].extract_text()
#             if not text:
#                 return ""
            
#             headers = {
#                 'Authorization': f'Bearer {API_KEY}',
#                 'Content-Type': 'application/json',
#                 'Accept': 'application/json'
#             }
            
#             # Enhanced translation payload
#             data = {
#                 "model": "deepseek-chat",
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a technical translator specialized in academic and scientific content. "
#                             "Instructions:\n"
#                             "1. All equations found in text MUST BE WRITEN in LaTeX.\n"
#                             "2. All greek letters and equations must be in LaTeX blocks and start exactly next to '\[' and '\]' operators. eg '\[x+2\alpha\]' unless '\[ x+2\alpha \]'. \n"
#                             "3. Maintain all Markdown formatting.\n"
#                             "4. Keep proper names, citations, and references unchanged.\n"
#                             "5. Ensure terminology consistency.\n"
#                             "6. Preserve paragraph structure and line breaks.\n"
#                             "7. Return only the translated text without explanations or comments.\n"
#                             "8. Format tables in Markdown respeting original layout.\n"
#                             "9. For images, write its legend in corresponding places in text."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": f"Translate to {target_lang}:\n{text}"
#                     }
#                 ],
#                 "temperature": 0,
#                 "max_tokens": 4000
#             }

#             response = requests.post(
#                 BASE_URL,
#                 json=data,
#                 headers=headers,
#                 timeout=30
#             )
#             response.raise_for_status()
#             return response.json()['choices'][0]['message']['content']
            
#     except Exception as e:
#         logging.error(f"Error translating page {page_path}: {str(e)}")
#         raise

# def main() -> None:
#     """Main function to execute document translation workflow.
    
#     Usage:
#         python translator.py <file> <target_language>
        
#     Args:
#         file (str): Path to input document
#         target_language (str): Target language code (e.g., 'pt', 'en')
        
#     Raises:
#         SystemExit: If incorrect number of arguments provided
#         Exception: If any error occurs during translation process
#     """
#     if len(sys.argv) != 3:
#         print("Usage: python translator.py <file> <target_language>")
#         sys.exit(1)
    
#     file_path = Path(sys.argv[1])
#     target_lang = sys.argv[2]
    
#     try:
#         # File validation and conversion
#         validate_file(file_path)
#         pdf_path = convert_to_pdf(file_path)
        
#         # PDF splitting
#         pages = split_pdf(pdf_path)
        
#         # Page translation
#         translations = []
#         for i, page in enumerate(pages):
#             print(f"Translating page {i+1}/{len(pages)}...")
#             translation = translate_page(page, target_lang)
#             translations.append(translation)
#             time.sleep(1)  # Rate limit
        
#         # Combine translations
#         output_file = OUTPUT_DIR / f"{file_path.stem}_{target_lang}.md"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write("\n\n".join(translations))
        
#         print(f"Translation complete! File saved at: {output_file}")
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
