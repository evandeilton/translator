# Automatic Document Translator

This is an AI agent that translates documents between languages using the DeepSeek API. It is designed to be easy to use and supports multiple document formats.

## Introduction

The Automatic Document Translator is a tool that leverages AI to translate documents from one language to another. It is ideal for users who need to quickly translate documents while maintaining the original formatting.

## Installation

### Prerequisites

- Python 3.8 or higher
- A DeepSeek API key

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/evandeilton/translator.git
   cd translator
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your DeepSeek API key in the `.env` file.

## How to Use

### Using the Terminal (Linux)

1. Place the document to be translated in the `input` folder.
2. Run the translator with the following command:

   ```bash
   python translator.py input/my_document.pdf en-US
   ```

   Replace `my_document.pdf` with the name of your file and `en-US` with the target language code.

3. The translated file will be saved in the `output` folder in Markdown format.

### Using Jupyter Notebook

You can also use the translator within a Jupyter Notebook. Here's an example:

```python
import os
from translator import translate_document

# Set the input and output paths
input_path = 'input/my_document.pdf'
output_path = 'output/my_document_translated.md'

# Set the target language
target_language = 'en-US'
# target_language = 'pt-BR' or português do Brasil
# target_language = 'fr-FR'
# target_language = 'it-IT'
# target_language = 'cn-CN'
# target_language = 'ru-RU'
# target_language = '..-..' find out you language. Restricted to DeepSeek suported languages.


# Run the translation
translate_document(input_path, output_path, target_language)
```

## Supported Formats

- PDF (.pdf)
- Word (.doc, .docx)
- OpenDocument Text (.odt)

## Directory Structure

- `input/`: Original documents
- `temp/`: Temporary files
  - `pages/`: Individual PDF pages
  - `translations/`: Temporary translations
- `output/`: Final results

## Limitations

- API costs
- Maximum file size: 10MB
- Maximum number of pages: 100
- Supported languages: Depend on the DeepSeek API

## Contributing

