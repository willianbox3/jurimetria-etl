"""
Esaj & DataJud Connector – versão 0.2
====================================
• Scraping paginado do e‑SAJ/TJCE
• Consulta autenticada ao DataJud (CNJ)
• Logging estruturado

Dependências
------------
```
pip install requests beautifulsoup4 python-dotenv pandas tqdm
```
Coloque seu token DataJud (chave **X-API-KEY**) em `.env` ou variável de
ambiente `DATAJUD_TOKEN`.
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import logging
import argparse
from datetime import date
from typing import List, Dict, Generator

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------------------------------------------------
# Config & logging
# ----------------------------------------------------------------------
load_dotenv()
TOKEN_DATAJUD = os.getenv("DATAJUD_TOKEN", "")
BASE_ESAJ = (
    "https://esaj.tjce.jus.br/cpopg/search.do"  # endpoint GET de pesquisa pública
)
BASE_DATAJUD = "https://www.cnj.jus.br/pesquisas-judiciarias/datajud/api/public"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# e‑SAJ crawler (paginado)
# ----------------------------------------------------------------------

def _parse_esaj_table(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "tabelaResultados"})
    if not table:
        return []
    rows = table.find_all("tr")[1:]  # skip header
    out = []
    for tr in rows:
        cols = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cols) < 6:
            continue
        processo, classe, assunto, orgao, data_decisao, _ = cols[:6]
        out.append(
            {
                "processo": processo,
                "orgao": orgao,
                "data": date.fromisoformat("-".join(reversed(data_decisao.split("/")))).isoformat(),
                "classe": classe,
                "assunto": assunto,
                "ementa": "",  # precisaria clicar; opcional
                "súmulas_aplicadas": [],
            }
        )
    return out


def fetch_esaj_tjce(
    classe: str,
    data_inicio: str = "2024-01-01",
    data_fim: str | None = None,
    max_pages: int | None = None,
    pause: float = 1.0,
) -> List[Dict]:
    """Raspagem paginada do e‑SAJ TJCE.

    Args:
        classe: classe processual (ex.: "Apelação Cível").
        data_inicio: AAAA-MM-DD.
        data_fim: AAAA-MM-DD ou None.
        max_pages: limitar nº de páginas. None = todas.
        pause: delay entre requisições.
    """
    logger.info("Iniciando scraping do e‑SAJ…")
    session = requests.Session()

    # Parâmetros genéricos do e‑SAJ (adaptado)
    query = {
        "conversationId": "",  # sessão
        "dadosConsulta.originados": "N",
        "classe": classe,
        "dataIni": data_inicio,
        "dataFim": data_fim or date.today().isoformat(),
        "paginaConsulta": 1,
        "localPesquisa.cdLocal": 1,
        "tipoNumero": "UNIFICADO",
    }

    all_rows: List[Dict] = []
    page = 1
    pbar = tqdm(total=max_pages or math.inf, desc="Páginas", unit="pg")
    while True:
        query["paginaConsulta"] = page
        resp = session.get(BASE_ESAJ, params=query, timeout=30)
        if resp.status_code != 200:
            logger.warning("Página %s retornou %s", page, resp.status_code)
            break
        rows = _parse_esaj_table(resp.text)
        if not rows:
            break
        all_rows.extend(rows)
        pbar.update(1)
        page += 1
        if max_pages and page > max_pages:
            break
        time.sleep(pause)
    pbar.close()
    logger.info("Total de processos coletados: %s", len(all_rows))
    return all_rows

# ----------------------------------------------------------------------
# DataJud TJCE
# ----------------------------------------------------------------------
HEADERS_DJ = {"X-API-KEY": TOKEN_DATAJUD} if TOKEN_DATAJUD else {}

def fetch_datajud_tjce(
    classe: str,
    ano: int,
    metrica: str = "tempo_julgamento",
) -> Dict:
    """Busca estatísticas da DataJud filtrando TJCE.

    metrica: "tempo_julgamento" | "taxa_provimento".
    """
    logger.info("Consultando DataJud (%s)…", metrica)
    endpoint = f"{BASE_DATAJUD}/estatisticas"
    params = {
        "siglaTribunal": "TJCE",
        "classe": classe,
        "ano": ano,
        "metrica": metrica,
    }
    r = requests.get(endpoint, params=params, headers=HEADERS_DJ, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Garantir formato padrão
    return {
        "classe": classe,
        "ano": ano,
        "tempo_medio_dias": data.get("tempo_medio_dias"),
        "taxa_provimento_percent": data.get("taxa_provimento_percent"),
    }

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser("Connector e‑SAJ/DataJud TJCE")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("esaj", help="Scraping do e‑SAJ")
    s1.add_argument("--classe", required=True)
    s1.add_argument("--data-inicio", default="2024-01-01")
    s1.add_argument("--data-fim")
    s1.add_argument("--max-pages", type=int)
    s1.add_argument("--save", metavar="PATH")

    s2 = sub.add_parser("datajud", help="Estatísticas DataJud")
    s2.add_argument("--classe", required=True)
    s2.add_argument("--ano", type=int, required=True)
    s2.add_argument("--metrica", choices=["tempo_julgamento", "taxa_provimento"], default="tempo_julgamento")
    s2.add_argument("--save", metavar="PATH")

    args = parser.parse_args()

    if args.cmd == "esaj":
        res = fetch_esaj_tjce(
            classe=args.classe,
            data_inicio=args.data_inicio,
            data_fim=args.data_fim,
            max_pages=args.max_pages,
        )
        out = json.dumps(res, ensure_ascii=False, indent=2)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(out)
        else:
            print(out)
    elif args.cmd == "datajud":
        res = fetch_datajud_tjce(args.classe, args.ano, args.metrica)
        out = json.dumps(res, ensure_ascii=False, indent=2)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(out)
        else:
            print(out)

if __name__ == "__main__":
    cli()
