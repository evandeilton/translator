# Tradutor Automático de Documentos

Este é um agente de IA que traduz documentos entre idiomas usando a API DeepSeek.

## Como usar

1. Coloque o documento a ser traduzido na pasta `input`
2. Execute o tradutor:

```bash
python translator.py input/meu_documento.pdf pt-BR
```

3. O arquivo traduzido será salvo na pasta `output` no formato Markdown

## Requisitos

- Python 3.8+
- Dependências instaladas: `pip install -r requirements.txt`
- Chave da API DeepSeek configurada no arquivo `.env`

## Formatos suportados

- PDF (.pdf)
- Word (.doc, .docx)
- OpenDocument Text (.odt)

## Estrutura de diretórios

- `input/`: Documentos originais
- `temp/`: Arquivos temporários
  - `pages/`: Páginas individuais do PDF
  - `translations/`: Traduções temporárias
- `output/`: Resultados finais

## Limitações

- Tamanho máximo do arquivo: 10MB
- Número máximo de páginas: 100
- Idiomas suportados: Dependem da API DeepSeek
