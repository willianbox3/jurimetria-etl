# legacy_datajud_connector.py
from __future__ import annotations

"""Legacy e-SAJ/DataJud connector (TJCE-focado)

⚠️  Mantido apenas para cenários muito específicos de integração
    – o fluxo principal está em *jurimetria_pipeline.py*.

Compatibilidade retroativa
--------------------------
Até então o script era chamado sem sub-comando:

    python legacy_datajud_connector.py --classe "Apelação Cível" …

A CLI abaixo exige o *sub-comando* (`esaj` ou `datajud`).  Para evitar
quebras em pipelines antigos adicionamos um *shim* que assume `esaj`
caso nada seja informado.
"""

import argparse
import json
import logging
import os
import sys
import time          # ➜ corrigido (usado em fetch_esaj_tjce)
from datetime import date
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
load_dotenv()
TOKEN_DATAJUD = os.getenv("DATAJUD_TOKEN", "")
BASE_ESAJ = "https://esaj.tjce.jus.br/cpopg/search.do"
BASE_DATAJUD = "https://www.cnj.jus.br/pesquisas-judiciarias/datajud/api/public"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helpers ESAJ
# ----------------------------------------------------------------------


def _parse_esaj_table(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tabelaResultados")
    if not table:
        return []
    rows = table.find_all("tr")[1:]  # pula cabeçalho
    out: List[Dict] = []
    for tr in rows:
        cols = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cols) < 6:
            continue
        proc, classe, assunto, orgao, data_decisao, _ = cols[:6]
        dia, mes, ano = data_decisao.split("/")
        out.append(
            {
                "processo": proc,
                "classe": classe,
                "assunto": assunto,
                "orgao": orgao,
                "data": f"{ano}-{mes}-{dia}",
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
    """Raspagem paginada do e-SAJ TJCE."""
    logger.info("Scraping e-SAJ (%s)…", classe)
    sess = requests.Session()
    query = {
        "conversationId": "",
        "dadosConsulta.originados": "N",
        "classe": classe,
        "dataIni": data_inicio,
        "dataFim": data_fim or date.today().isoformat(),
        "paginaConsulta": 1,
        "localPesquisa.cdLocal": 1,
        "tipoNumero": "UNIFICADO",
    }

    results: List[Dict] = []
    page = 1
    pbar = tqdm(total=max_pages or float("inf"), desc="Páginas", unit="pg")
    while True:
        query["paginaConsulta"] = page
        r = sess.get(BASE_ESAJ, params=query, timeout=30)
        if r.status_code != 200:
            logger.warning("Página %s retornou %s", page, r.status_code)
            break
        rows = _parse_esaj_table(r.text)
        if not rows:
            break
        results.extend(rows)
        pbar.update(1)
        page += 1
        if max_pages and page > max_pages:
            break
        time.sleep(pause)
    pbar.close()
    logger.info("Total de processos coletados: %s", len(results))
    return results


# ----------------------------------------------------------------------
# Helpers DataJud
# ----------------------------------------------------------------------
HEADERS_DJ = {"X-API-KEY": TOKEN_DATAJUD} if TOKEN_DATAJUD else {}


def fetch_datajud_tjce(classe: str, ano: int, metrica: str = "tempo_julgamento") -> Dict:
    """Busca estatísticas da DataJud filtrando TJCE."""
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
    return {
        "classe": classe,
        "ano": ano,
        "tempo_medio_dias": data.get("tempo_medio_dias"),
        "taxa_provimento_percent": data.get("taxa_provimento_percent"),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

# ➡️  Compat shim: insere "esaj" se nenhum sub-comando explícito for passado
if len(sys.argv) > 1 and sys.argv[1] not in {"esaj", "datajud", "-h", "--help"}:
    sys.argv.insert(1, "esaj")

parser = argparse.ArgumentParser("Connector e-SAJ/DataJud TJCE")
sub = parser.add_subparsers(dest="cmd", required=True)  # <- agora *obrigatório*
parser.set_defaults(cmd="esaj")  # mas continua com default, se o shim não atuar

# --- esaj ---
s1 = sub.add_parser("esaj", help="Scraping do e-SAJ")
s1.add_argument("--classe", required=True)
s1.add_argument("--data-inicio", default="2024-01-01")
s1.add_argument("--data-fim")
s1.add_argument("--max-pages", type=int)

# --- datajud ---
s2 = sub.add_parser("datajud", help="Estatísticas DataJud")
s2.add_argument("--classe", required=True)
s2.add_argument("--ano", type=int, required=True)
s2.add_argument(
    "--metrica",
    choices=["tempo_julgamento", "taxa_provimento"],
    default="tempo_julgamento",
)

args = parser.parse_args()

if args.cmd == "esaj":
    resultado = fetch_esaj_tjce(
        classe=args.classe,
        data_inicio=args.data_inicio,
        data_fim=args.data_fim,
        max_pages=args.max_pages,
    )
else:
    resultado = fetch_datajud_tjce(args.classe, args.ano, args.metrica)

print(json.dumps(resultado, ensure_ascii=False, indent=2))
