# PDF Translator

A versatile PDF translation tool supporting multiple AI translation providers with advanced text extraction capabilities.

## Features

- Translate PDF documents using multiple AI providers:
  - DeepSeek
  - Anthropic Claude
  - OpenAI
  - Google Gemini
- Advanced PDF text extraction with multiple methods:
  - PyMuPDF native extraction
  - Tesseract OCR for scanned documents
  - MarkItDown for preserving LaTeX and Markdown
- Chunk-level parallel translation or full-text translation in a single request.
- Support for multiple input document formats
- Configurable translation parameters
- Retry logic for handling transient API errors.
- Approximate token estimation for chunking.

## Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`
- Optional system dependencies:
  - Tesseract OCR: `sudo apt-get install tesseract-ocr`
  - unoconv: `sudo apt-get install unoconv`

## Environment Variables

### Translation Provider API Keys (Required)
Choose at least one:
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Google Gemini API key

### Optional Configuration
- `DEEPSEEK_API_URL`: Custom DeepSeek API endpoint
- `SUPPORTED_FORMATS`: Comma-separated list of supported file formats (default: .pdf,.doc,.docx,.odt)
- `TEMP_DIR`: Temporary directory for file conversions (default: 'temp')
- `OUTPUT_DIR`: Output directory for translated files (default: 'output')

## Usage

### Basic Usage

```bash
# Translate using DeepSeek (default)
python translator.py --file input.pdf --target en

# Translate using Anthropic Claude
python translator.py --file input.pdf --target en --provider anthropic

# Translate using OpenAI
python translator.py --file input.pdf --target en --provider openai

# Translate using Google Gemini
python translator.py --file input.pdf --target en --provider gemini
```

### Advanced Options

```bash
# Specify model, concurrency, and chunk size
python translator.py --file input.pdf --target en \
    --provider openai \
    --model gpt-4o-mini \
    --max-chunk-tokens 1500 \
    --concurrency 6 \
    --full-text # Translate the entire document in a single request
    --trace  # Enable detailed logging
```

**Detailed Explanation of Advanced Options:**

*   `--model`: Allows specifying a specific model from the chosen provider. Different models have varying capabilities and costs.
*   `--max-chunk-tokens`: Controls the approximate maximum number of tokens per chunk. Smaller chunks can improve translation quality for complex documents but may increase the number of API requests.
*   `--concurrency`: Determines the number of parallel translation requests. Higher concurrency can speed up translation but may be limited by API rate limits.
*   `--full-text`: Enables translating the entire document in a single request, instead of chunking. This can be faster but might not be suitable for very large documents or complex layouts.
*   `--trace`: Enables detailed logging for debugging purposes.
*   `--cache`: (DeepSeek only) Specifies whether to assume the input tokens are a cache "hit" or "miss" for cost estimation.


## Supported Providers

### DeepSeek
- Models: 
  - `deepseek-chat` (default)
  - `deepseek-reasoner`

### Anthropic
- Models:
  - `claude-3-5-haiku-20241022` (default)
  - Other Claude 3 models supported

### OpenAI
- Models:
  - `gpt-4o-mini` (default)
  - Other OpenAI models supported

### Google Gemini
- Models:
  - `gemini-2.0-flash` (default)
  - Other Gemini models supported

## Supported Input Formats

- PDF
- DOC
- DOCX
- ODT

## Performance and Cost

- Parallel translation with configurable concurrency
- Token-based chunking to optimize translation. Note that tokenization is approximate; for production use, consider a dedicated tokenizer.
- Cost estimation available for DeepSeek API usage. The `--cache` parameter (DeepSeek only) allows specifying "hit" or "miss" for cost calculations.

## Troubleshooting

- Ensure all required API keys are set.
- Check system dependencies are installed.
- The application includes error handling for file validation, conversion failures, and API errors.
- Use `--trace` flag for detailed debugging information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Output Format

The translated document is saved as a Markdown file (.md).

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## Acknowledgments

- DeepSeek AI
- Anthropic
- OpenAI
- Google AI
- PyMuPDF
- Tesseract OCR
- pdf2image
- MarkItDown
