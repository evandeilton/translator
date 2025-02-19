from setuptools import setup, find_packages

setup(
    name="pdf-translator",
    version="0.1.1",
    description="A versatile PDF translation tool supporting multiple AI providers.",
    author="José Lopes",
    author_email="evandeilton@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "PyMuPDF>=1.18.0",        # Para extração nativa de PDF
        "pdf2image>=1.16.0",      # Para conversão PDF para imagem
        "pytesseract>=0.3.8",     # Para OCR
        "tqdm>=4.62.0",           # Para barra de progresso
        "markitdown>=0.1.0",      # Opcional para extração LaTeX/Markdown
        "anthropic>=0.3.0",       # Para API Anthropic/Claude
        "openai>=1.0.0",          # Para API OpenAI
        "google-generativeai>=0.1.0", # Para API Google/Gemini
        "retry>=0.9.2",           # Para lógica de retry
        "python-dotenv>=0.19.0",  # Para carregar variáveis de ambiente
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'mypy>=0.900',
        ],
    },
    entry_points={
        "console_scripts": [
            "translate=translator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="pdf, translation, ai, nlp, ocr",
    project_urls={
        "Source": "https://github.com/evandeilton/translator.git",
        "Bug Reports": "https://github.com/evandeilton/translator/issues",
    }
)
