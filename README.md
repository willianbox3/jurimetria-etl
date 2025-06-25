# Jurimetria – Coleta e Análise de Dados Judiciais

Bem-vindo! Este repositório oferece **um fluxo oficial completo** para baixar, tratar e analisar dados judiciais via API pública do CNJ, além de **um conector opcional** para cenários específicos de integração.

## Fluxo oficial: `jurimetria_pipeline.py`

| Etapa          | O que faz                                                                                     | Saída                     |
|----------------|----------------------------------------------------------------------------------------------|---------------------------|
| **Coleta**     | Consulta a API CNJ para a classe 12729 (ANPP) em um ou vários tribunais (padrão **TJCE**).    | JSON → DataFrame          |
| **Persistência** | Salva em Parquet (`zstd`) e CSV no diretório `dados_jurimetria/`.                              | `jurimetria.parquet`, `jurimetria.csv` |
| **Análises**   | Exibe estatísticas básicas e gera gráfico de horário de ajuizamento.                          | `horario_jurimetria.jpg`  |

### Uso rápido

```bash
# Defina a chave antes:
export CNJ_API_KEY='APIKey …'

# Coletar apenas TJCE (padrão):
python jurimetria_pipeline.py

# Coletar múltiplos tribunais (ex.: TJSP, TJRS):
python jurimetria_pipeline.py TJSP TJRS
```

---

## 🔌 Conector opcional: `legacy_esaj_connector.py`

Este script (antigo `esaj_datajud_connector.py`) é **enxuto** e serve somente para **baixar os documentos “crus”** da API ESAJ sem qualquer pós‑processamento. Útil quando você quer:

* Fazer *debug* da API.
* Integrar a coleta a um *pipeline* externo (Airflow, cron, etc.).
* Explorar outros formatos de saída com scripts próprios.

> Se você **não** precisa desses casos, basta usar o `jurimetria_pipeline.py` e ignorar o conector.

---

## Estrutura do repositório

```
.
├── jurimetria_pipeline.py        # Fluxo oficial (coleta → análises)
├── legacy_esaj_connector.py      # Conector opcional (download puro)
├── dados_jurimetria/             # Saídas (criado em tempo de execução)
└── README.md                     # Este arquivo
```

---

## Contribuindo

1. Crie um *branch* a partir de `main`.
2. Siga o padrão *Black* (`black .`).
3. Instale as dependências antes de rodar os testes:
   `pip install -r requirements.txt`
4. Adicione/atualize *tests* (`python jurimetria_pipeline.py test`).
5. Abra um *pull request*.

---

⚖️ **Licença**: MIT

💡 Dúvidas ou sugestões? Abra uma *issue* ou consulte o autor.
