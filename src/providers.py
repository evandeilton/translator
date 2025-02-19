#!/usr/bin/env python3
"""
providers.py

Este módulo implementa classes orientadas a objetos para acesso às APIs de tradução:
DeepSeek, Anthropic, OpenAI e Gemini.

Cada classe herda de BaseProvider e implementa o método translate().
Também há uma função factory para instanciar o provedor adequado.
"""

import os
import logging
import requests
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any, List

import anthropic  # Biblioteca para Anthropic
import openai     # Biblioteca para OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Para Gemini (opcional)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # Caso Gemini não seja utilizado
    types = None
    GEMINI_AVAILABLE = False

# Configuração dos preços para o DeepSeek (por 1M de tokens)
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

#: SYSTEM PROMPT ÚNICO PARA TODOS OS PROVIDERS
SYSTEM_PROMPT_TRANSLATOR = (
    "You are an elite academic translator with extensive expertise in scientific and technical documentation. "
    "Your task is to translate academic texts while preserving the original formatting, structure, and technical accuracy. "
    "You must ensure that all mathematical expressions, LaTeX code, Greek letters, equations, and technical symbols remain intact.\n\n"

    "ABSOLUTE PRESERVATION RULES (NÃO DEVE SER TRADUZIDO):\n"
    "1. Bibliographic references, citations, and reference lists\n"
    "2. Author names, affiliations, and titles\n"
    "3. URLs, DOIs, and digital object identifiers\n"
    "4. Journal names, conference titles, and publisher details\n"
    "5. Software names, library names, and tool names\n"
    "6. Dataset identifiers, codes, and names\n"
    "7. Standard scientific abbreviations and measurement units\n"
    "8. Variable names, function names, and any programming identifiers\n"
    "9. Statistical test names (e.g., t-test, ANOVA, etc.)\n\n"

    "MATHEMATICAL AND TECHNICAL NOTATION:\n"
    "1. Equations:\n"
    "   - Inline math (e.g., $a^2 + b^2 = c^2$) and display math (e.g., $$E=mc^2$$) must be preserved exactly as written, "
    "including all LaTeX formatting, spacing, equation numbering, and references.\n"
    "   - All mathematical operators, symbols, and special characters must remain unchanged.\n"
    "   - Greek letters, whether uppercase or lowercase, must be preserved and represented using appropriate LaTeX commands (e.g., "
    "λ -> $\\lambda$, ϕ -> $\\phi$, τ -> $\\tau$, etc.).\n"
    "2. Code and Programming Elements:\n"
    "   - Preserve all code blocks with their original syntax, indentation, and comments.\n"
    "   - Do not translate code, variable names, function names, or technical identifiers within code segments.\n"
    "3. Technical Elements:\n"
    "   - Maintain SI units and measurements exactly as presented.\n"
    "   - Preserve statistical notation, chemical formulas, and technical nomenclature without alterations.\n\n"

    "DOCUMENT STRUCTURE AND FORMATTING:\n"
    "1. Formatting:\n"
    "   - Identify and preserve titles, subtitles, headings, lists, bullets, and paragraphs using the original formatting "
    "or Markdown syntax (e.g., **, *, #, ##).\n"
    "   - Maintain table structures, alignments, figure and table numbering, and cross-references as in the source document.\n"
    "2. Organization:\n"
    "   - Retain the original paragraph structure, section and subsection hierarchy, footnotes, endnotes, and appendix formatting.\n\n"

    "ACADEMIC CONVENTIONS:\n"
    "1. Terminology and Vocabulary:\n"
    "   - Use precise, domain-specific technical vocabulary and maintain consistent terminology throughout the translation.\n"
    "2. Style and Register:\n"
    "   - Preserve the formal academic tone, use hedging language when appropriate, and maintain the original citation style.\n\n"

    "QUALITY CONTROL AND FINAL CHECK:\n"
    "1. Verify that all technical formatting, mathematical expressions, and LaTeX syntax are preserved exactly.\n"
    "2. Ensure that all Greek letters and special symbols are correctly represented using proper inline formatting.\n"
    "3. Provide only the translated content without any added explanations, commentary, or deviations from the original text.\n"
    "4. Do not include any portion of the source text unless explicitly instructed.\n\n"

    "FINAL OUTPUT REQUIREMENT:\n"
    "Before finalizing the output, perform a thorough check to ensure that all formatting, technical elements, and content integrity "
    "have been maintained. The final output must be impeccably formatted and ready for academic use."
)


class BaseProvider(ABC):
    """
    Classe base abstrata para provedores de tradução.
    """
    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor base.
        
        Args:
            model: Nome do modelo a ser utilizado.
            logger: Instância do logger para registrar atividades.
        """
        self.model = model
        self.logger = logger

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, Any]]:
        """
        Traduza o texto para a língua de destino.

        Args:
            text: Texto original.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').

        Returns:
            Tuple com o texto traduzido e informações de uso.
        """
        pass


class DeepSeekProvider(BaseProvider):
    """
    Implementa a tradução via API DeepSeek.
    """
    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor DeepSeek.
        
        Args:
            model: Nome do modelo DeepSeek a ser utilizado.
            logger: Instância do logger para registrar atividades.
            
        Raises:
            ValueError: Se a chave de API não estiver configurada.
        """
        super().__init__(model, logger)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY não está definida.")
        self.base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

    @staticmethod
    def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, cache: str) -> float:
        """
        Estima o custo de tradução via DeepSeek.

        Args:
            model: Ex.: 'deepseek-chat' ou 'deepseek-reasoner'.
            prompt_tokens: Tokens de entrada.
            completion_tokens: Tokens de saída.
            cache: 'hit' ou 'miss'.

        Returns:
            Custo estimado em USD.
            
        Raises:
            ValueError: Se o modelo for desconhecido ou o cache não for 'hit' ou 'miss'.
        """
        if model not in DEEPSEEK_PRICING:
            raise ValueError(f"Modelo desconhecido '{model}'.")
        if cache not in ("hit", "miss"):
            raise ValueError(f"Cache deve ser 'hit' ou 'miss'. Recebido: '{cache}'.")

        pricing = DEEPSEEK_PRICING[model][f"cache_{cache}"]
        cost_input_per_mil = pricing["input"]
        cost_output_per_mil = pricing["output"]

        input_millions = prompt_tokens / 1_000_000
        output_millions = completion_tokens / 1_000_000
        return input_millions * cost_input_per_mil + output_millions * cost_output_per_mil

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, Any]]:
        """
        Traduz texto via API DeepSeek com retry automático em caso de falha.
        
        Args:
            text: Texto original a ser traduzido.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').
            
        Returns:
            Tuple contendo o texto traduzido e informações de uso.
            
        Raises:
            requests.RequestException: Se houver falha na comunicação com a API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
            ],
            "temperature": 0,
            "max_tokens": 8000
        }
        self.logger.debug(f"Enviando requisição para DeepSeek com model={self.model} e target_lang={target_lang}...")
        response = requests.post(self.base_url, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        resp_json = response.json()
        translated_text = resp_json["choices"][0]["message"]["content"]
        usage_info = resp_json.get("usage", {})
        return translated_text, usage_info


class AnthropicProvider(BaseProvider):
    """
    Implementa a tradução via API da Anthropic.
    """
    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor Anthropic.
        
        Args:
            model: Nome do modelo Anthropic a ser utilizado.
            logger: Instância do logger para registrar atividades.
            
        Raises:
            ValueError: Se a chave de API não estiver configurada.
        """
        super().__init__(model, logger)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY não está definida.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, int]]:
        """
        Traduz texto via API Anthropic com retry automático em caso de falha.
        
        Args:
            text: Texto original a ser traduzido.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').
            
        Returns:
            Tuple contendo o texto traduzido e informações de uso.
        """
        self.logger.debug(f"Enviando requisição para Anthropic com model={self.model} e target_lang={target_lang}...")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            temperature=0,
            system=SYSTEM_PROMPT_TRANSLATOR,
            messages=[{"role": "user", "content": f"Translate to {target_lang}:\n{text}"}]
        )
        
        translated_text = message.content[0].text
        usage_info = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens
        }
        return translated_text, usage_info


class OpenAIProvider(BaseProvider):
    """
    Implementa a tradução via API da OpenAI.
    """
    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor OpenAI.
        
        Args:
            model: Nome do modelo OpenAI a ser utilizado.
            logger: Instância do logger para registrar atividades.
            
        Raises:
            ValueError: Se a chave de API não estiver configurada.
        """
        super().__init__(model, logger)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não está definida.")
        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, int]]:
        """
        Traduz texto via API OpenAI com retry automático em caso de falha.
        
        Args:
            text: Texto original a ser traduzido.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').
            
        Returns:
            Tuple contendo o texto traduzido e informações de uso.
        """
        self.logger.debug(f"Enviando requisição para OpenAI com model={self.model} e target_lang={target_lang}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
            ],
            max_tokens=8000,
            temperature=0
        )

        translated_text = response.choices[0].message.content or ""
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0
        }
        return translated_text, usage_info


class GeminiProvider(BaseProvider):
    """
    Implementa a tradução via API da Gemini.
    """
    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor Gemini.
        
        Args:
            model: Nome do modelo Gemini a ser utilizado.
            logger: Instância do logger para registrar atividades.
            
        Raises:
            ValueError: Se a chave de API não estiver configurada.
        """
        super().__init__(model, logger)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não está definida.")

        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini API requer o pacote google.genai. Instale-o para usar este provedor.")

        self.genai = genai
        self.types = types
        self.client = self.genai.Client(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, int]]:
        """
        Traduz texto via API Gemini com retry automático em caso de falha.
        
        Args:
            text: Texto original a ser traduzido.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').
            
        Returns:
            Tuple contendo o texto traduzido e informações de uso.
        """
        self.logger.debug(f"Enviando requisição para Gemini com model={self.model} e target_lang={target_lang}...")

        response = self.client.generate_content(
            model=self.model,
            contents=f"Translate to {target_lang}:\n{text}",
            generation_config=self.types.GenerationConfig(
                temperature=0,
                max_output_tokens=8000,
                system_instruction=SYSTEM_PROMPT_TRANSLATOR
            )
        )
        translated_text = response.text
        usage_info = {
            "prompt_tokens": 0,  # Gemini não retorna contagem de tokens
            "completion_tokens": 0
        }
        return translated_text, usage_info


def get_provider(provider_name: str, model: str, logger: logging.Logger) -> BaseProvider:
    """
    Retorna uma instância do provedor de tradução apropriado.

    Args:
        provider_name: Nome do provedor ('deepseek', 'anthropic', 'openai', 'gemini' ou 'openrouter').
        model: Nome do modelo a ser utilizado.
        logger: Instância do logger.

    Returns:
        Instância do provedor selecionado.
        
    Raises:
        ValueError: Se o provedor não for suportado.
    """
    provider_map = {
        "deepseek": DeepSeekProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
    }
    if provider_name not in provider_map:
        raise ValueError(f"Provedor não suportado: {provider_name}")
    return provider_map[provider_name](model, logger)


class OpenRouterProvider(BaseProvider):
    """
    Implementa a tradução via API do OpenRouter, compatível com a interface
    dos outros provedores e reutilizando o SDK do OpenAI.
    
    Documentação: https://openrouter.ai/docs
    """
    
    # Modelos suportados pelo OpenRouter
    MODELS = [
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free",
            # "openai/o3-mini-high",
            # "openai/o3-mini",
            "openai/chatgpt-4o-latest",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "google/gemini-2.0-pro-exp-02-05:free",
            "deepseek/deepseek-r1-distill-llama-70b:free",
            "deepseek/deepseek-r1-distill-qwen-32b",
            "deepseek/deepseek-r1:free",
            "qwen/qwen-plus",
            "qwen/qwen-max",
            "qwen/qwen-turbo",
            "qwen/qwen2.5-vl-72b-instruct:free",        
            "mistralai/codestral-2501",
            "mistralai/mistral-7b-instruct:free",        
            "mistralai/mistral-small-24b-instruct-2501:free",
            "anthropic/claude-3.5-haiku-20241022:beta",
            "anthropic/claude-3.5-sonnet",
            "perplexity/sonar-reasoning",
            "perplexity/sonar",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "perplexity/llama-3.1-sonar-small-128k-chat",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free"        
        ]

    def __init__(self, model: str, logger: logging.Logger) -> None:
        """
        Inicializa o provedor OpenRouter.
        
        Args:
            model: Nome do modelo a ser utilizado.
            logger: Instância do logger para registrar atividades.
            
        Raises:
            ValueError: Se a chave de API não estiver configurada.
        """
        super().__init__(model, logger)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY não está definida.")
            
        # Inicializa o cliente OpenAI com a URL base do OpenRouter
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # Configura headers extras para rastreamento (opcional)
        self.extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Academic Translator")
        }
        
        # Verifica se o modelo solicitado é suportado
        if model not in self.MODELS:
            self.logger.warning(f"Modelo '{model}' não está na lista de modelos conhecidos do OpenRouter")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def translate(self, text: str, target_lang: str) -> Tuple[str, Dict[str, Any]]:
        """
        Traduz texto via API OpenRouter com retry automático em caso de falha.
        
        Args:
            text: Texto original a ser traduzido.
            target_lang: Código da língua alvo (ex.: 'pt', 'en').
            
        Returns:
            Tuple contendo o texto traduzido e informações de uso.
            
        Raises:
            Exception: Se ocorrer um erro na comunicação com a API.
        """
        self.logger.debug(f"Enviando requisição para OpenRouter com model={self.model} e target_lang={target_lang}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                    {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
                ],
                max_tokens=8000,
                temperature=0,
                extra_headers=self.extra_headers
            )
            
            translated_text = response.choices[0].message.content or ""
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
            }
            return translated_text, usage_info
            
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {str(e)}")
            raise
