# src/jurimetria_pipeline.py
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ─────────────────────────── Configuração ────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CLASSE_CODIGO_DEFAULT = 12729          # ANPP – mantido como fallback
PAGE_SIZE = 1_000
DEFAULT_TRIBUNAIS = ["TJCE"]           # padrão: TJCE
OUT_DIR = Path("dados_jurimetria").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_log_level(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        log.error(f"Invalid log level: {level_name}")
        sys.exit(1)
    logging.getLogger().setLevel(level)

def get_headers() -> Dict[str, str]:
    api_key = os.getenv("CNJ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Defina a variável de ambiente CNJ_API_KEY antes de executar o script.")
    if not api_key.lower().startswith("apikey"):
        api_key = f"APIKey {api_key}"
    return {"Authorization": api_key, "Content-Type": "application/json"}

def build_base_url(sigla: str) -> str:
    return f"https://api-publica.datajud.cnj.jus.br/api_publica_{sigla.lower()}/_search"

def tz_utc_to_sp(dt_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not dt_str:
        return None
    try:
        ts = pd.to_datetime(dt_str, utc=True)
        return ts.tz_convert("America/Sao_Paulo")
    except Exception:
        return None

def lista_assuntos(raw: List[Dict[str, Any]]) -> List[str]:
    # ...
    return nomes

def lista_movimentos(raw: List[Dict[str, Any]]) -> List[List[Any]]:
    # ...
    return movs_sorted

def _build_query(
    classe_codigo: Optional[int],
    classe_nome: Optional[str],
    dt_ini: Optional[str],
    dt_fim: Optional[str],
) -> Dict[str, Any]:
    filtros: List[Dict[str, Any]] = []

    # filtro de classe (usa .keyword para evitar 400)  
    if classe_nome:
        filtros.append({"term": {"classe.nome.keyword": classe_nome}})
    elif classe_codigo:
        filtros.append({"term": {"classe.codigo": classe_codigo}})

    # filtro de período
    if dt_ini or dt_fim:
        range_clause: Dict[str, Any] = {"range": {"dataAjuizamento": {}}}
        if dt_ini:
            range_clause["range"]["dataAjuizamento"]["gte"] = dt_ini
        if dt_fim:
            range_clause["range"]["dataAjuizamento"]["lte"] = dt_fim
        filtros.append(range_clause)

    if filtros:
        return {"bool": {"filter": filtros}}
    else:
        return {"match_all": {}}

def fetch_raw_hits(
    tribunal: str,
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
    dt_ini: Optional[str] = None,
    dt_fim: Optional[str] = None,
    page_size: int = PAGE_SIZE,
    max_total: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    headers = get_headers()
    url = build_base_url(tribunal)
    total = 0

    def do_request() -> Generator[Dict[str, Any], None, None]:
        base_payload = {
            "size": page_size,
            "query": _build_query(classe_codigo, classe_nome, dt_ini, dt_fim),
            "sort": [
                {"dataAjuizamento": {"order": "desc"}},
                {"_id": "asc"}        # garante cursor único
            ],
        }
        # loop de paginação com DEBUG condicional...
        # yield from hits

    yield from do_request()

def parse_hit(hit: Dict[str, Any], tribunal: str) -> Dict[str, Any]:
    # ...

def build_dataframe(
    tribunais: List[str] = DEFAULT_TRIBUNAIS,
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
    dt_ini: Optional[str] = None,
    dt_fim: Optional[str] = None,
    max_proc: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trib in tribunais:
        regs = [
            parse_hit(h, trib)
            for h in fetch_raw_hits(trib, classe_codigo, classe_nome, dt_ini, dt_fim, PAGE_SIZE, max_proc)
        ]
        if regs:
            frames.append(pd.DataFrame(regs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("Nenhum dado para persistir.")
        return

    csv_path = OUT_DIR / "jurimetria.csv"
    parquet_path = OUT_DIR / "jurimetria.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Dados salvos em:\n  • {csv_path}\n  • {parquet_path}")

def plot_horario(df: pd.DataFrame, classe_label: str | int) -> None:
    if df.empty:
        return
    # gera gráfico...
    jpg_path = OUT_DIR / "horario_jurimetria.jpg"
    plt.savefig(jpg_path)
    print(f"Gráfico salvo em {jpg_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de Jurimetria via API pública do CNJ")
    # argumentos...
    args = parser.parse_args()

    set_log_level(args.log_level)

    try:
        df = build_dataframe(
            args.tribunais,
            args.classe_codigo,
            args.classe_nome,
            args.dt_ini,
            args.dt_fim,
            args.max_proc,
        )
    except EnvironmentError as err:
        print(f"⚠️  {err}")
        sys.exit(1)

    print(f"✔️  Total de processos: {len(df):,}")
    persist_df(df)
    plot_horario(df, args.classe_nome or args.classe_codigo or CLASSE_CODIGO_DEFAULT)

if __name__ == "__main__":
    main()
