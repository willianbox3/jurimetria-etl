name: Jurimetria ETL

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  etl:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run jurimetria pipeline
        env:
          CNJ_API_KEY: ${{ secrets.CNJ_API_KEY }}
        run: |
          python src/jurimetria_pipeline.py \
            --tribunais TJCE \
            --max-processos 10 \
            --log-level INFO

      - name: Debug: verifique arquivos gerados
        run: |
          echo ">>> pwd: $(pwd)"
          echo ">>> Conteúdo raiz:"
          ls -lha .
          echo ">>> Conteúdo de dados_jurimetria:"
          ls -lha dados_jurimetria || echo ">> pasta não existe ou está vazia"

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dados-jurimetria
          path: dados_jurimetria/**
