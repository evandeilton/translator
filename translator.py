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
import anthropic  # Biblioteca Anthropic
import openai     # Biblioteca OpenAI
import google.generativeai as genai  # Biblioteca Google Gemini

# Para retry avançado (trata erros transitórios de API)
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional progress bar; instale com: pip install tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------
# Suporte a múltiplos provedores de API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validação das chaves de API
if not any([DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Nenhuma chave de API configurada. Configure uma das variáveis: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY ou GEMINI_API_KEY.")

# Configurar Gemini (caso haja a chave)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_URLS = {
    "deepseek": os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
    "anthropic": "https://api.anthropic.com/v1/messages",  # Endpoint Anthropic
    "openai": "https://api.openai.com/v1/chat/completions"
}

SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.doc,.docx,.odt").split(",")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tabela de preços para DeepSeek (por 1M tokens), simplificada
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
    Configura e retorna um objeto logger com saída para o console.
    
    Args:
        trace (bool): Se True, define o nível DEBUG, caso contrário INFO.
    
    Returns:
        logging.Logger: Logger configurado.
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
    Valida se um arquivo existe e se está em um formato suportado.

    Args:
        file_path (Path): Caminho para o arquivo a validar.
        logger (logging.Logger): Logger para mensagens de diagnóstico.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o formato do arquivo não for suportado.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo '{file_path}' não encontrado.")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        allowed_formats = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Formato '{file_path.suffix}' não suportado. Permitidos: {allowed_formats}"
        )
    logger.debug(f"Arquivo '{file_path.name}' é válido com o sufixo '{file_path.suffix}'.")

def convert_to_pdf(file_path: Path, logger: logging.Logger) -> Path:
    """
    Converte um documento para PDF, se necessário, utilizando o `unoconv`.

    Args:
        file_path (Path): Caminho para o arquivo de entrada.
        logger (logging.Logger): Logger para mensagens de diagnóstico.

    Returns:
        Path: Caminho para o arquivo PDF convertido.

    Raises:
        RuntimeError: Se o comando `unoconv` falhar.
    """
    if file_path.suffix.lower() == ".pdf":
        logger.info(f"'{file_path.name}' já é um PDF. Conversão não necessária.")
        return file_path

    output_path = TEMP_DIR / f"{file_path.stem}.pdf"
    logger.info(f"Convertendo '{file_path.name}' para PDF...")

    try:
        result = subprocess.run(
            ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
            check=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"unoconv falhou (código {result.returncode}).")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode().strip() if e.stderr else "Erro desconhecido."
        logger.error(f"Falha ao converter '{file_path}'. Erro:\n{err_msg}")
        raise RuntimeError(f"Falha na conversão de '{file_path}' para PDF.") from e

    logger.info(f"Conversão completa. PDF salvo em '{output_path}'.")
    return output_path

# ------------------------------------------------------------------------
# PDF -> Extração de Texto por Página (em memória)
# ------------------------------------------------------------------------
def extract_all_page_texts(pdf_path: Path, logger: logging.Logger) -> List[str]:
    """
    Extrai o conteúdo textual de cada página de um PDF, retornando uma lista
    de strings, uma por página.

    Args:
        pdf_path (Path): Caminho para o arquivo PDF.
        logger (logging.Logger): Logger para mensagens de diagnóstico.

    Returns:
        List[str]: Uma string por página (mesmo que vazia).
    """
    logger.info(f"Extraindo texto de '{pdf_path.name}'...")
    page_texts: List[str] = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        logger.debug(f"Foram encontradas {num_pages} páginas em '{pdf_path.name}'.")

    # O pdfplumber geralmente extrai melhor o texto que o PyPDF2
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            page_texts.append(raw_text)

    logger.info(f"Extração completa. {len(page_texts)} páginas lidas.")
    return page_texts

# ------------------------------------------------------------------------
# Contagem Aproximada de Tokens / Lógica de Chunking
# ------------------------------------------------------------------------
def approximate_token_count(text: str) -> int:
    """
    Estima a quantidade de tokens de um texto.  
    Para produção, considere um tokenizer dedicado (por exemplo, 'tiktoken').

    Args:
        text (str): Texto a ser analisado.

    Returns:
        int: Estimativa do número de tokens.
    """
    # Aproximação simples: 4 caracteres por token
    return math.ceil(len(text) / 4)

def chunk_text(text: str, max_chunk_tokens: int) -> List[str]:
    """
    Divide o texto em segmentos que não excedam o número máximo de tokens.

    Args:
        text (str): Texto original a ser dividido.
        max_chunk_tokens (int): Número máximo de tokens permitidos por segmento.

    Returns:
        List[str]: Lista de segmentos de texto, cada um dentro do limite.
    """
    lines = text.split("\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_count = 0

    for line in lines:
        line_tokens = approximate_token_count(line)

        # Se uma única linha for muito grande, quebra-a por palavras
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
# API de Tradução (DeepSeek, Anthropic, OpenAI, Gemini)
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
    Traduz um bloco de texto utilizando a API DeepSeek, Anthropic, OpenAI ou Gemini.

    Args:
        text (str): Texto original a ser traduzido.
        target_lang (str): Código da língua alvo (ex.: 'pt', 'en').
        model (str): Nome do modelo (ex.: 'deepseek-chat', 'claude-3-sonnet', 'gpt-4o-mini', 'gemini-2.0-flash').
        provider (str): Provedor da API ('deepseek', 'anthropic', 'openai' ou 'gemini').
        logger (logging.Logger): Logger para mensagens de diagnóstico.

    Returns:
        Tuple[str, Dict]: (texto_traduzido, informações_de_uso)
    """
    # Instrução do sistema para orientar o tradutor acadêmico
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
            ],
            "temperature": 0,
            "max_tokens": 4000
        }

        logger.debug(f"Enviando requisição para DeepSeek: model={model}, target_lang={target_lang}")
        resp = requests.post(BASE_URLS["deepseek"], json=data, headers=headers, timeout=60)
        resp.raise_for_status()

        resp_json = resp.json()
        translated_text = resp_json["choices"][0]["message"]["content"]
        usage_info = resp_json.get("usage", {})

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.debug(f"Enviando requisição para Anthropic: model={model}, target_lang={target_lang}")
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
        logger.debug(f"Enviando requisição para OpenAI: model={model}, target_lang={target_lang}")
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
        logger.debug(f"Enviando requisição para Gemini: model={model}, target_lang={target_lang}")
        # Utilizando o método do client configurado para gerar conteúdo.
        response = genai.models.generate_content(
            model=model,
            config=genai.types.GenerationConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=4000,
            ),
            contents=[f"Translate to {target_lang}:\n{text}"]
        )
        translated_text = response.text
        usage_info = {
            "prompt_tokens": 0,  # Atualmente, a API Gemini não retorna contagem de tokens.
            "completion_tokens": 0
        }
    else:
        raise ValueError(f"Provedor não suportado: {provider}")

    logger.debug("Resposta da API recebida com sucesso.")
    return translated_text, usage_info

# ------------------------------------------------------------------------
# Tradução Paralela de Todos os Chunks
# ------------------------------------------------------------------------
def chunk_all_pages(
    page_texts: List[str],
    max_chunk_tokens: int,
    logger: logging.Logger
) -> List[List[str]]:
    """
    Divide o texto de cada página em chunks (segmentos) se necessário.
    
    Args:
        page_texts (List[str]): Texto bruto de cada página do PDF.
        max_chunk_tokens (int): Limite aproximado de tokens por chunk.
        logger (logging.Logger): Para mensagens de debug.
    
    Returns:
        List[List[str]]: Uma lista de listas onde cada sublista contém os chunks da respectiva página.
    """
    logger.info("Dividindo todas as páginas em segmentos menores...")
    chunked_pages: List[List[str]] = []
    for idx, text in enumerate(page_texts):
        if text.strip():
            chunks = chunk_text(text, max_chunk_tokens)
            logger.debug(f"Página {idx+1} possui {len(chunks)} segmento(s).")
        else:
            chunks = []
            logger.debug(f"Página {idx+1} não possui texto extraível.")
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
    Traduz todos os chunks de todas as páginas em paralelo.
    
    Args:
        chunked_pages (List[List[str]]): Cada sublista contém todos os chunks de uma página.
        target_lang (str): Código da língua alvo.
        model (str): Nome do modelo.
        provider (str): Provedor da API.
        concurrency (int): Número máximo de threads.
        logger (logging.Logger): Para mensagens de diagnóstico.
    
    Returns:
        Tuple[List[List[str]], List[List[Dict]]]:
            (chunks_traduzidos_por_página, dados_de_uso_por_chunk)
    """
    tasks = []  # (page_idx, chunk_idx, texto_do_chunk)
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
        f"Submetendo {len(tasks)} segmento(s) distribuídos em {num_pages} página(s) para uma thread pool com tamanho={concurrency}."
    )

    def worker(page_idx: int, chunk_idx: int, chunk_text_: str):
        return translate_text(chunk_text_, target_lang, model, provider, logger)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(worker, p, c, txt): (p, c)
            for (p, c, txt) in tasks
        }

        future_iter = tqdm(as_completed(future_to_task), total=len(tasks), desc="Traduzindo segmentos") if TQDM_AVAILABLE else as_completed(future_to_task)
        for future in future_iter:
            p_idx, c_idx = future_to_task[future]
            try:
                translated_chunk, usage_info = future.result()
                per_page_translated_chunks[p_idx][c_idx] = translated_chunk
                per_page_usage_data[p_idx][c_idx] = usage_info
            except Exception as e:
                logger.error(f"Segmento (página={p_idx+1}, chunk={c_idx+1}) falhou: {e}")
                per_page_translated_chunks[p_idx][c_idx] = "[CHUNK FAILED]"
                per_page_usage_data[p_idx][c_idx] = {}

    return per_page_translated_chunks, per_page_usage_data

# ------------------------------------------------------------------------
# Cálculo de Uso & Custo
# ------------------------------------------------------------------------
def compute_usage_stats(
    usage_data: List[List[Optional[Dict]]]
) -> Dict[str, int]:
    """
    Agrega as estatísticas de uso de todos os chunks.

    Args:
        usage_data (List[List[Optional[Dict]]]): Lista com os dicionários de uso por chunk.
    
    Returns:
        Dict[str, int]: Estatísticas agregadas (prompt, completion e total de tokens).
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
    Estima o custo para o uso da DeepSeek, dado o modelo, tokens e cenário de cache.

    Args:
        model (str): Por exemplo, 'deepseek-chat' ou 'deepseek-reasoner'.
        prompt_tokens (int)
        completion_tokens (int)
        cache (str): 'hit' ou 'miss'.
    
    Returns:
        float: Custo em USD.
    """
    if model not in DEEPSEEK_PRICING:
        raise ValueError(f"Modelo desconhecido '{model}'. Verifique DEEPSEEK_PRICING.")
    if cache not in ("hit", "miss"):
        raise ValueError(f"cache deve ser 'hit' ou 'miss'. Recebido '{cache}'.")

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
    Ponto de entrada CLI para tradução de documentos PDF utilizando DeepSeek, Anthropic, OpenAI ou Gemini.

    Passos:
      1) Validar o formato do arquivo.
      2) Converter para PDF (se necessário).
      3) Extrair o texto de cada página.
      4) Dividir cada página em chunks.
      5) Traduzir todos os chunks em paralelo.
      6) Reagrupar e salvar a tradução final.
      7) Calcular uso e custo.
    """
    parser = argparse.ArgumentParser(
        description="Traduz documentos PDF via DeepSeek, Anthropic, OpenAI ou Gemini (com concorrência em nível de chunks)."
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
        help="Modelo específico a ser usado. Padrão: modelo do provedor.",
    )
    parser.add_argument(
        "--cache",
        default="miss",
        choices=["hit", "miss"],
        help="Assuma que os tokens de entrada são cache HIT ou MISS (somente DeepSeek). Padrão: 'miss'.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=2000,
        help="Máximo aproximado de tokens permitidos por chunk. Padrão=2000.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Número de requisições paralelas para tradução dos chunks. Padrão=4.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Habilitar logging em nível DEBUG.",
    )
    args = parser.parse_args()

    # Definir modelos padrão se não especificados
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
        # 1) Validar e converter para PDF (se necessário)
        validate_file(file_path, logger=logger)
        pdf_path = convert_to_pdf(file_path, logger=logger)

        # 2) Extrair texto de cada página
        page_texts = extract_all_page_texts(pdf_path, logger=logger)
        num_pages = len(page_texts)
        if num_pages == 0:
            raise ValueError(f"'{pdf_path.name}' não contém páginas com texto extraível.")

        # 3) Dividir cada página em chunks
        chunked_pages = chunk_all_pages(page_texts, max_chunk_tokens=args.max_chunk_tokens, logger=logger)

        # 4) Traduzir todos os chunks em paralelo
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

        # 5) Reagrupar o texto final
        output_file = OUTPUT_DIR / f"{file_path.stem}_{args.target}.md"
        with open(output_file, "w", encoding="utf-8") as out:
            for i, translated_chunks in enumerate(per_page_translated_chunks):
                page_translation = "\n\n".join(translated_chunks)
                out.write(page_translation)
                if i < num_pages - 1:
                    out.write("\n\n<!-- NEXT PAGE -->\n\n")

        logger.info(f"Tradução completa. Saída salva em '{output_file}'")

        # 6) Calcular uso e custo (apenas para DeepSeek)
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
            logger.info(f"  Tokens de Prompt:     {total_prompt}")
            logger.info(f"  Tokens de Completion: {total_completion}")
            logger.info(f"  Tokens Totais:        {total_tokens}")
            logger.info(f"Custo estimado: ${estimated_cost:.4f}")
        else:
            logger.info("Estimativa de custo disponível somente para DeepSeek.")

    except Exception as e:
        logger.error(f"Tradução falhou: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
