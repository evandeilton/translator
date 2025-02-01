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
import anthropic  # Added Anthropic library
import openai  # Added OpenAI library

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
# Support multiple API providers
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not DEEPSEEK_API_KEY and not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
    raise ValueError("No API keys set. Set one of DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")

BASE_URLS = {
    "deepseek": os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
    "anthropic": "https://api.anthropic.com/v1/messages",  # Anthropic Messages API endpoint
    "openai": "https://api.openai.com/v1/chat/completions"
}

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
    provider: str,
    logger: logging.Logger
) -> Tuple[str, Dict]:
    """
    Translate a block of text using DeepSeek, Anthropic, or OpenAI API.

    Args:
        text (str): The raw text to translate
        target_lang (str): The target language code (e.g. 'pt', 'en')
        model (str): Model name (e.g. 'deepseek-chat', 'claude-3-sonnet', 'gpt-4o-mini')
        provider (str): API provider ('deepseek', 'anthropic', or 'openai')
        logger (logging.Logger): Logger for diagnostic messages

    Returns:
        Tuple[str, Dict]: (translated_text, usage_info)
    """
    system_prompt = (
        "You are an elite academic translator with expertise in scientific and technical documentation, committed to preserving the integrity of academic works while ensuring precise translation.\n\n"
        
        "ABSOLUTE PRESERVATION RULES:\n"
        "1. Never Translate:\n"
        "   - Bibliography and references\n"
        "   - Author names and affiliations\n"
        "   - URLs, DOIs, and digital identifiers\n"
        "   - Journal names and conference proceedings\n"
        "   - Software names, packages, and tools\n"
        "   - Dataset identifiers and names\n"
        "   - Standard scientific abbreviations\n"
        "   - Variable names in code or equations\n"
        "   - Statistical test names (e.g., t-test, ANOVA)\n\n"
        
        "MATHEMATICAL AND TECHNICAL NOTATION:\n"
        "1. Equations:\n"
        "   - Inline math: $...$ (e.g., $\\alpha$, $\\beta$)\n"
        "   - Display math: $$...$$\n"
        "   - Preserve all operators, symbols, and spacing\n"
        "   - Maintain equation numbering and references\n"
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
        "   - Preserve all Markdown syntax (**, *, #, ##)\n"
        "   - Maintain table structure and alignment\n"
        "   - Keep figure and table numbering systems\n"
        "   - Preserve cross-references and internal links\n"
        "2. Organization:\n"
        "   - Maintain original paragraph structure\n"
        "   - Preserve section and subsection hierarchy\n"
        "   - Keep footnotes and endnotes formatting\n"
        "   - Maintain appendix structure and labeling\n\n"
        
        "ACADEMIC CONVENTIONS:\n"
        "1. Terminology:\n"
        "   - Use field-appropriate technical vocabulary\n"
        "   - Maintain term consistency throughout\n"
        "   - Preserve standard field abbreviations\n"
        "   - Keep measurement units unchanged\n"
        "2. Style:\n"
        "   - Maintain academic register and formality\n"
        "   - Preserve hedging language appropriately\n"
        "   - Keep passive voice where used\n"
        "   - Maintain citation styles and formats\n"
        "3. Field-Specific:\n"
        "   - Respect discipline-specific conventions\n"
        "   - Maintain standard field terminology\n"
        "   - Preserve specialized notation systems\n\n"
        
        "QUALITY CONTROL:\n"
        "1. Consistency:\n"
        "   - Maintain uniform terminology\n"
        "   - Ensure consistent formatting\n"
        "   - Keep coherent style throughout\n"
        "2. Accuracy:\n"
        "   - Preserve technical precision\n"
        "   - Maintain numerical accuracy\n"
        "   - Keep methodological clarity\n\n"
        
        "OUTPUT REQUIREMENTS:\n"
        "- Provide only translated content\n"
        "- No explanations or comments\n"
        "- No original text unless specified\n"
        "- Maintain all formatting and structure\n"
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
                {
                    "role": "system",
                    "content": system_prompt
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
        resp = requests.post(BASE_URLS["deepseek"], json=data, headers=headers, timeout=60)
        resp.raise_for_status()

        resp_json = resp.json()
        translated_text = resp_json["choices"][0]["message"]["content"]
        usage_info = resp_json.get("usage", {})

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        logger.debug(f"Sending API request to Anthropic: model={model}, target_lang={target_lang}")
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate to {target_lang}:\n{text}"
                }
            ]
        )

        translated_text = message.content[0].text
        usage_info = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens
        }

    elif provider == "openai":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        logger.debug(f"Sending API request to OpenAI: model={model}, target_lang={target_lang}")
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
    provider: str,
    concurrency: int,
    logger: logging.Logger
) -> Tuple[List[List[str]], List[List[Dict]]]:
    """
    Translate *all chunks from all pages* in parallel.

    Args:
        chunked_pages (List[List[str]]): Each sub-list is all chunks for that page
        target_lang (str): Target language code
        model (str): Model name
        provider (str): API provider ('deepseek' or 'anthropic')
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
        return translate_text(chunk_text_, target_lang, model, provider, logger)

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
    Main CLI entry point for PDF translation using DeepSeek or Anthropic.

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
        description="Translate PDF documents via DeepSeek or Anthropic API (faster chunk-level concurrency)."
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
        "--provider",
        default="deepseek",
        choices=["deepseek", "anthropic", "openai"],
        help="API provider to use for translation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model to use. Defaults to provider's default model.",
    )
    parser.add_argument(
        "--cache",
        default="miss",
        choices=["hit", "miss"],
        help="Assume input tokens are cache HIT or MISS (DeepSeek only). Default: 'miss'.",
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

    # Set default models if not specified
    if not args.model:
        args.model = {
            "deepseek": "deepseek-chat",
            "anthropic": "claude-3-5-haiku-20241022",
            "openai": "gpt-4o-mini"
        }[args.provider]

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
            provider=args.provider,
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

        # 6) Compute usage & cost (only for DeepSeek currently)
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

            logger.info("Usage Summary:")
            logger.info(f"  Prompt Tokens:     {total_prompt}")
            logger.info(f"  Completion Tokens: {total_completion}")
            logger.info(f"  Overall Tokens:    {total_tokens}")
            logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        else:
            logger.info("Cost estimation is currently available only for DeepSeek.")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
