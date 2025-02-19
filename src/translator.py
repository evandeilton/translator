#!/usr/bin/env python3
"""
translator.py

This module implements classes for PDF document processing and translation orchestration.

São definidas:
  - PDFProcessor: Responsável por validação, conversão para PDF e extração de texto.
  - DocumentTranslator: Coordena o particionamento do texto, tradução paralela dos segmentos e
    agregação dos resultados.

O script também disponibiliza um ponto de entrada (main) para uso via CLI.
"""

import os
import sys
import math
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF para extração nativa de PDF
from pdf2image import convert_from_path
import pytesseract

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Importa a função que retorna a instância do provedor
from providers import get_provider, DeepSeekProvider


class PDFProcessor:
    """
    Responsável pela validação do arquivo, conversão para PDF (se necessário) e extração de texto.
    """
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.doc,.docx,.odt").split(",")
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def validate_file(self, file_path: Path) -> None:
        """
        Valida se o arquivo existe e se o formato é suportado.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo '{file_path}' não encontrado.")
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            allowed = ", ".join(self.SUPPORTED_FORMATS)
            raise ValueError(f"Formato '{file_path.suffix}' não suportado. Permitidos: {allowed}.")
        self.logger.debug(f"Arquivo '{file_path.name}' é válido.")

    def convert_to_pdf(self, file_path: Path) -> Path:
        """
        Converte o arquivo para PDF utilizando 'unoconv', se necessário.
        """
        if file_path.suffix.lower() == ".pdf":
            self.logger.info(f"'{file_path.name}' já é um PDF.")
            return file_path

        output_path = self.TEMP_DIR / f"{file_path.stem}.pdf"
        self.logger.info(f"Convertendo '{file_path.name}' para PDF...")
        try:
            result = subprocess.run(
                ["unoconv", "-f", "pdf", "-o", str(output_path), str(file_path)],
                check=True, capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"unoconv falhou (código {result.returncode}).")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode().strip() if e.stderr else "Erro desconhecido."
            self.logger.error(f"Erro ao converter '{file_path}': {err}")
            raise RuntimeError(f"Conversão para PDF falhou: {file_path}") from e

        self.logger.info(f"Conversão concluída: '{output_path}'.")
        return output_path

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extrai texto do PDF utilizando múltiplos métodos (PyMuPDF, OCR e, opcionalmente, MarkItDown).

        Insere um marcador de página (<<<PAGE_BREAK>>>) para facilitar o particionamento posterior.
        """
        self.logger.info(f"Iniciando extração de texto de '{pdf_path.name}'...")
        extraction_results: Dict[str, str] = {}

        # Método 1: Extração com PyMuPDF
        def extract_with_pymupdf() -> str:
            text = ""
            try:
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    page_text = page.get_text().strip()
                    text += page_text + "\n<<<PAGE_BREAK>>>\n"
                doc.close()
            except Exception as e:
                self.logger.error(f"Erro no PyMuPDF: {e}")
            return text

        self.logger.info("Extraindo com PyMuPDF...")
        extraction_results['pymupdf'] = extract_with_pymupdf()
        self.logger.debug(f"PyMuPDF extraiu {len(extraction_results['pymupdf'])} caracteres.")

        # Método 2: Extração via OCR (pdf2image + pytesseract)
        def extract_with_ocr() -> str:
            text = ""
            try:
                images = convert_from_path(str(pdf_path), dpi=300)
                for image in images:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    text += page_text.strip() + "\n<<<PAGE_BREAK>>>\n"
            except Exception as e:
                self.logger.error(f"Erro no OCR: {e}")
            return text

        self.logger.info("Extraindo com OCR...")
        extraction_results['ocr'] = extract_with_ocr()
        self.logger.debug(f"OCR extraiu {len(extraction_results['ocr'])} caracteres.")

        # Método 3: Extração com MarkItDown (se disponível)
        def extract_with_markitdown() -> str:
            if not MARKITDOWN_AVAILABLE:
                self.logger.warning("MarkItDown não disponível.")
                return ""
            try:
                md = MarkItDown()
                result = md.convert(str(pdf_path))
                return result.text_content
            except Exception as e:
                self.logger.error(f"Erro no MarkItDown: {e}")
                return ""

        if MARKITDOWN_AVAILABLE:
            self.logger.info("Extraindo com MarkItDown...")
            extraction_results['markitdown'] = extract_with_markitdown()
            self.logger.debug(f"MarkItDown extraiu {len(extraction_results['markitdown'])} caracteres.")
        else:
            extraction_results['markitdown'] = ""

        # Seleciona o método que retornou mais conteúdo
        best_method = max(extraction_results, key=lambda k: len(extraction_results[k].strip()))
        best_text = extraction_results[best_method]
        self.logger.info(f"Método selecionado: {best_method} ({len(best_text)} caracteres).")

        if not best_text.strip():
            raise Exception("Falha ao extrair texto do PDF.")
        return best_text


class DocumentTranslator:
    """
    Coordena a tradução do documento: particionamento do texto, tradução paralela dos segmentos e reagrupamento.
    """
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self, provider_name: str, model: str, max_chunk_tokens: int, concurrency: int, logger: logging.Logger) -> None:
        self.logger = logger
        self.provider = get_provider(provider_name, model, logger)
        self.max_chunk_tokens = max_chunk_tokens
        self.concurrency = concurrency

    @staticmethod
    def approximate_token_count(text: str) -> int:
        """
        Estima a quantidade de tokens do texto.
        """
        return math.ceil(len(text) / 4)

    def chunk_text(self, text: str) -> List[str]:
        """
        Divide o texto em segmentos que não excedam o limite de tokens definido.
        """
        lines = text.split("\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_count = 0

        for line in lines:
            tokens_line = self.approximate_token_count(line)
            # If the line is too long, break it into words
            if tokens_line >= self.max_chunk_tokens:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_count = 0
                words = line.split()
                subchunk: List[str] = []
                subcount = 0
                for word in words:
                    wcount = self.approximate_token_count(word + " ")
                    if subcount + wcount <= self.max_chunk_tokens:
                        subchunk.append(word)
                        subcount += wcount
                    else:
                        chunks.append(" ".join(subchunk))
                        subchunk = [word]
                        subcount = wcount
                if subchunk:
                    chunks.append(" ".join(subchunk))
                continue

            if current_count + tokens_line > self.max_chunk_tokens:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_count = tokens_line
            else:
                current_chunk.append(line)
                current_count += tokens_line

        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def chunk_all_pages(self, page_texts: List[str]) -> List[List[str]]:
        """
        For each page, divides the text into segments.
        """
        self.logger.info("Dividindo páginas em segmentos...")
        chunked_pages: List[List[str]] = []
        for idx, page in enumerate(page_texts):
            if page.strip():
                chunks = self.chunk_text(page)
                self.logger.debug(f"Página {idx + 1}: {len(chunks)} segmento(s).")
            else:
                chunks = []
                self.logger.debug(f"Página {idx + 1}: sem texto.")
            chunked_pages.append(chunks)
        return chunked_pages

    def translate_chunks(self, chunked_pages: List[List[str]], target_lang: str) -> Tuple[List[List[str]], List[List[Dict]]]:
        """
        Parallel translation of all segments.

        Args:
            chunked_pages (List[List[str]]): Segmentos por página.
            target_lang (str): Código da língua alvo (ex.: 'en-US').

        Returns:
            Tuple[List[List[str]], List[List[Dict]]]: (segmentos traduzidos, dados de uso)
        """
        tasks = []  # Cada tarefa: (page_idx, chunk_idx, texto do segmento)
        for i, page_chunks in enumerate(chunked_pages):
            for j, chunk in enumerate(page_chunks):
                tasks.append((i, j, chunk))

        num_pages = len(chunked_pages)
        translated_pages: List[List[Optional[str]]] = [
            [None] * len(page) for page in chunked_pages
        ]
        usage_pages: List[List[Optional[Dict]]] = [
            [None] * len(page) for page in chunked_pages
        ]

        self.logger.info(f"Submetendo {len(tasks)} segmento(s) com concurrency={self.concurrency}...")

        def worker(page_idx: int, chunk_idx: int, text: str, lang: str) -> Tuple[str, Dict]:
            return self.provider.translate(text, lang)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_task = {
                executor.submit(worker, p, c, t, target_lang): (p, c)
                for (p, c, t) in tasks
            }
            future_iter = tqdm(as_completed(future_to_task), total=len(tasks), desc="Traduzindo") if TQDM_AVAILABLE else as_completed(future_to_task)
            for future in future_iter:
                p_idx, c_idx = future_to_task[future]
                try:
                    translated, usage = future.result()
                    translated_pages[p_idx][c_idx] = translated
                    usage_pages[p_idx][c_idx] = usage
                except Exception as e:
                    self.logger.error(f"Falha no segmento (página {p_idx+1}, segmento {c_idx+1}): {e}")
                    translated_pages[p_idx][c_idx] = "[CHUNK FAILED]"
                    usage_pages[p_idx][c_idx] = {}
        return translated_pages, usage_pages

    def aggregate_usage(self, usage_data: List[List[Optional[Dict]]]) -> Dict[str, int]:
        """
        Agrega os dados de uso de todos os segmentos.
        """
        prompt_total = 0
        completion_total = 0
        for page in usage_data:
            for usage in page:
                if usage:
                    prompt_total += usage.get("prompt_tokens", 0)
                    completion_total += usage.get("completion_tokens", 0)
        return {
            "prompt_tokens_total": prompt_total,
            "completion_tokens_total": completion_total,
            "total_tokens_total": prompt_total + completion_total
        }

    def translate_document(self, pdf_text: str, target_lang: str, full_text: bool = False) -> Tuple[str, Dict]:
        """
        Realiza a tradução do documento.
        Se full_text for True, envia o texto completo em uma única requisição.
        Caso contrário, divide o texto em páginas e segmentos, traduz e reagrupa os resultados.

        Args:
            pdf_text (str): Texto completo extraído do PDF.
            target_lang (str): Código da língua alvo (ex.: 'en-US').
            full_text (bool): Se True, traduz o documento completo de uma vez.

        Returns:
            Tuple[str, Dict]: (texto traduzido, informações de uso)
        """
        if full_text:
            self.logger.info("Traduzindo documento completo em uma única requisição...")
            translated, usage = self.provider.translate(pdf_text, target_lang)
            return translated, usage

        # Divide o texto em páginas utilizando o marcador <<<PAGE_BREAK>>>
        pages = [page.strip() for page in pdf_text.split("<<<PAGE_BREAK>>>") if page.strip()]
        if not pages:
            raise ValueError("Nenhuma página com texto extraído.")
        self.logger.info(f"Extração resultou em {len(pages)} página(s).")
        chunked_pages = self.chunk_all_pages(pages)
        translated_pages, usage_pages = self.translate_chunks(chunked_pages, target_lang)
        # Reagrupa as páginas traduzidas
        final_text = "\n\n".join(["\n\n".join(page) for page in translated_pages])
        aggregated_usage = self.aggregate_usage(usage_pages)
        return final_text, aggregated_usage

    def save_translation(self, file_stem: str, target_lang: str, content: str) -> Path:
        """
        Salva o texto traduzido em um arquivo.
        """
        output_file = self.OUTPUT_DIR / f"{file_stem}_{target_lang}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        self.logger.info(f"Tradução salva em '{output_file}'.")
        return output_file


def setup_logger(trace: bool) -> logging.Logger:
    """
    Configura e retorna um logger.
    """
    level = logging.DEBUG if trace else logging.INFO
    logger = logging.getLogger("DocumentTranslator")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="PDF Translator")
    parser.add_argument("--file", required=True, help="Input file")
    parser.add_argument("--target", required=True, help="Target language")
    parser.add_argument("--provider", type=str, default="openai", help="Translation provider")
    parser.add_argument("--model", type=str, default="", help="Model to use")
    parser.add_argument("--trace", action="store_true", help="Enable detailed logging")
    parser.add_argument("--cache", default="miss", choices=["hit", "miss"],
                        help="Indica se tokens de entrada são cache HIT ou MISS (apenas DeepSeek).")
    parser.add_argument("--max-chunk-tokens", type=int, default=2000,
                        help="Limite aproximado de tokens por segmento. Padrão=2000.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Número de requisições paralelas para tradução dos segmentos. Padrão=4.")
    parser.add_argument("--full-text", action="store_true", default=False,
                        help="Se definido, envia o texto completo para tradução em uma única requisição.")
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    
    # Mapeia alias "openrouter" para "gemini"
    if args.provider.lower() == "openrouter":
        print("Mapping provider 'openrouter' to 'gemini'")
        args.provider = "gemini"

    try:
        logger = setup_logger(True)  # Força debug logging
        logger.debug("Argumentos recebidos:")
        logger.debug(f"  file: {args.file}")
        logger.debug(f"  target: {args.target}")
        logger.debug(f"  provider: {args.provider}")
        logger.debug(f"  model: {args.model}")
        
        # Define o modelo padrão se não for especificado
        if not args.model:
            defaults = {
                "openrouter": "google/gemini-2.0-pro-exp-02-05:free",
                "deepseek": "deepseek-chat", 
                "anthropic": "claude-3-5-haiku-20241022",
                "openai": "gpt-4o-mini",
                "gemini": "gemini-2.0-flash"
            }
            args.model = defaults[args.provider]
            logger.debug(f"Usando modelo padrão: {args.model}")

        logger.debug("Iniciando processamento do documento...")
        pdf_processor = PDFProcessor(logger)
        doc_translator = DocumentTranslator(args.provider, args.model, args.max_chunk_tokens, args.concurrency, logger)

        file_path = Path(args.file).resolve()
        logger.debug(f"Caminho completo do arquivo: {file_path}")
        
        pdf_processor.validate_file(file_path)
        pdf_path = pdf_processor.convert_to_pdf(file_path)
        logger.debug(f"PDF para processamento: {pdf_path}")
        
        extracted_text = pdf_processor.extract_text(pdf_path)
        logger.debug(f"Texto extraído: {len(extracted_text)} caracteres")

        # Traduz o documento
        logger.debug("Iniciando tradução...")
        translated_text, usage_info = doc_translator.translate_document(extracted_text, args.target, args.full_text)
        
        logger.debug("Salvando resultado...")
        output_file = doc_translator.save_translation(file_path.stem, args.target, translated_text)
        logger.info(f"Tradução concluída: {output_file}")

        # Se o provedor for DeepSeek, exibe a estimativa de custo
        if args.provider == "deepseek":
            # Se full_text for True, o usage_info é um dicionário simples; caso contrário, já foi agregado.
            if args.full_text:
                usage_agg = doc_translator.aggregate_usage([[usage_info]])
            else:
                usage_agg = doc_translator.aggregate_usage(usage_info if isinstance(usage_info, list) else [usage_info])
            total_prompt = usage_agg["prompt_tokens_total"]
            total_completion = usage_agg["completion_tokens_total"]
            total_tokens = usage_agg["total_tokens_total"]
            cost = DeepSeekProvider.estimate_cost(args.model, total_prompt, total_completion, args.cache)
            logger.info("Resumo de Uso:")
            logger.info(f"  Prompt Tokens:     {total_prompt}")
            logger.info(f"  Completion Tokens: {total_completion}")
            logger.info(f"  Total Tokens:      {total_tokens}")
            logger.info(f"Custo Estimado: ${cost:.4f}")
        else:
            logger.info("Estimativa de custo disponível apenas para DeepSeek.")
    except Exception as e:
        if args.trace:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
