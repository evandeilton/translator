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
- Chunk-level parallel translation
- Support for multiple input document formats
- Configurable translation parameters

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
    --trace  # Enable detailed logging
```

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
- Token-based chunking to optimize translation
- Cost estimation available for DeepSeek API usage

## Troubleshooting

- Ensure all required API keys are set
- Check system dependencies are installed
- Use `--trace` flag for detailed debugging information

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
