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
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def set_log_level(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        log.error(f"Invalid log level: {level_name}")
        sys.exit(1)
    logging.getLogger().setLevel(level)

CLASSE_CODIGO_DEFAULT = 12729          # ANPP – mantido como fallback
PAGE_SIZE = 1_000
DEFAULT_TRIBUNAIS = ["TJCE"]           # padrão: TJCE
OUT_DIR = Path("dados_jurimetria").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── Utilidades ─────────────────────────────
def get_headers() -> Dict[str, str]:
    # Hardcoded API key to avoid environment variable error
    api_key = "APIKey cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
    if not api_key:
        raise EnvironmentError("Defina a variável de ambiente CNJ_API_KEY antes de executar o script.")
    if not api_key.lower().startswith("apikey"):
        api_key = f"APIKey {api_key}"
    return {"Authorization": api_key, "Content-Type": "application/json"}


def build_base_url(sigla: str) -> str:
    """Monta o endpoint do índice público de cada tribunal (api_publica_<sigla>)."""
    return f"https://api-publica.datajud.cnj.jus.br/api_publica_{sigla.lower()}/_search"


import pandas._libs.tslibs.np_datetime as np_datetime

def tz_utc_to_sp(dt_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not dt_str:
        return None
    try:
        ts = pd.to_datetime(dt_str, utc=True)
        return ts.tz_convert("America/Sao_Paulo")
    except np_datetime.OutOfBoundsDatetime:
        return None


def lista_assuntos(raw: List[Dict[str, Any]]) -> List[str]:
    nomes = []
    for a in raw:
        if isinstance(a, dict):
            nomes.append(a.get("nome", ""))
        elif isinstance(a, list) and len(a) > 0:
            first = a[0]
            if isinstance(first, dict):
                nomes.append(first.get("nome", ""))
            else:
                nomes.append("")
        else:
            nomes.append("")
    return nomes


def lista_movimentos(raw: List[Dict[str, Any]]) -> List[List[Any]]:
    movs: List[List[Any]] = []
    for m in raw:
        movs.append([m.get("codigo"), m.get("nome"), tz_utc_to_sp(m.get("dataHora"))])
    default = pd.Timestamp("1970-01-01", tz="America/Sao_Paulo")
    return sorted(movs, key=lambda x: x[2] or default)


# ────────────────────── Consulta ao DataJud ──────────────────────────
def _build_query(classe_codigo: Optional[int], classe_nome: Optional[str]) -> Dict[str, Any]:
    if classe_nome:
        # tenta primeiro pelo nome; se der 400 o chamador tratará
        return {"term": {"classe.nome": classe_nome}}
    if classe_codigo:
        return {"term": {"classe.codigo": classe_codigo}}
    # No class filter
    return {"match_all": {}}


def fetch_raw_hits(
    tribunal: str,
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
    page_size: int = PAGE_SIZE,
) -> Generator[Dict[str, Any], None, None]:
    """Paginação `search_after` sobre o índice público de cada tribunal."""
    headers = get_headers()
    url = build_base_url(tribunal)

    def do_request(query_nome: Optional[str], query_codigo: Optional[int]) -> Generator[Dict[str, Any], None, None]:
        base_payload = {
            "size": page_size,
            "query": _build_query(query_codigo, query_nome),
            "sort": [
                {"dataAjuizamento": {"order": "desc"}}
            ],
        }
        search_after: Optional[List[Any]] = None
        while True:
            payload = dict(base_payload)
            if search_after:
                payload["search_after"] = search_after
            log.debug(f"Enviando payload para {tribunal}: {payload}")
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                log.debug(f"Request headers: {resp.request.headers}")
                log.debug(f"Request body: {resp.request.body}")
                log.debug(f"Response status: {resp.status_code}")
                log.debug(f"Response headers: {resp.headers}")
                log.debug(f"Response body: {resp.text}")
            except Exception as e:
                log.error(f"Erro na requisição para {tribunal}: {e}")
                return

            if resp.status_code in (400, 404):
                log.warning("Tribunal %s retornou %s – pulando.", tribunal, resp.status_code)
                return

            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                return

            yield from hits
            new_cursor = hits[-1]["sort"]
            if new_cursor == search_after:
                return
            search_after = new_cursor

    # Try querying by class name first if provided
    if classe_nome:
        results = list(do_request(classe_nome, None))
        if results:
            yield from results
            return
        else:
            log.info(f"Consulta por nome de classe '{classe_nome}' não retornou resultados ou falhou, tentando por código.")
    # Fallback to querying by class code
    if classe_codigo:
        results = list(do_request(None, classe_codigo))
        if results:
            yield from results
            return
        else:
            log.info(f"Consulta por código de classe '{classe_codigo}' não retornou resultados ou falhou, tentando sem filtro de classe.")
    # Fallback to querying without class filter
    yield from do_request(None, None)


def parse_hit(hit: Dict[str, Any], tribunal: str) -> Dict[str, Any]:
    src = hit["_source"]
    return {
        "tribunal": tribunal,
        "numero_processo": src.get("numeroProcesso"),
        "classe": src.get("classe", {}).get("nome"),
        "data_ajuizamento": tz_utc_to_sp(src.get("dataAjuizamento")),
        "ultima_atualizacao": tz_utc_to_sp(src.get("dataHoraUltimaAtualizacao")),
        "formato": src.get("formato", {}).get("nome"),
        "codigo_orgao": src.get("orgaoJulgador", {}).get("codigo"),
        "orgao_julgador": src.get("orgaoJulgador", {}).get("nome"),
        "municipio": src.get("orgaoJulgador", {}).get("codigoMunicipioIBGE"),
        "grau": src.get("grau"),
        "assuntos": lista_assuntos(src.get("assuntos", [])),
        "movimentos": lista_movimentos(src.get("movimentos", [])),
        "sort": hit.get("sort", [None])[0],
    }


def build_dataframe(
    tribunais: List[str],
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trib in tribunais:
        registros = [
            parse_hit(h, trib) for h in fetch_raw_hits(trib, classe_codigo, classe_nome)
        ]
        if registros:
            frames.append(pd.DataFrame(registros))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────── Persistência & gráfico ──────────────────────
def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("Nenhum dado para persistir.")
        return
    parquet = OUT_DIR / "jurimetria.parquet"
    csv = OUT_DIR / "jurimetria.csv"
    df.to_parquet(parquet, compression="zstd", index=False)
    df.to_csv(csv, index=False)
    print(f"Dados salvos em:\n  • {parquet}\n  • {csv}")


def plot_horario(df: pd.DataFrame) -> None:
    if df.empty or "data_ajuizamento" not in df.columns:
        return
    horas = (
        pd.to_datetime(df["data_ajuizamento"], utc=True, errors="coerce")
        .dropna()
        .dt.tz_convert("America/Sao_Paulo")
        .dt.hour
    )
    if horas.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cont = horas.value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    cont.plot(kind="bar")
    plt.title("Horário de ajuizamento")
    plt.xlabel("Hora do dia")
    plt.ylabel("Processos")
    plt.tight_layout()
    out = OUT_DIR / "horario_jurimetria.jpg"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {out}")


# ───────────────────────────── CLI ───────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de Jurimetria via API pública do CNJ")
    parser.add_argument(
        "--tribunais",
        nargs="+",
        metavar="TJXX",
        default=DEFAULT_TRIBUNAIS,
        help="Lista de tribunais (TJSP TJCE …). Padrão: TJCE",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classe-codigo", type=int, help="Código numérico da classe")
    group.add_argument(
        "--classe",
        dest="classe_nome",
        help='Nome da classe (ex.: "Apelação Cível"). Sobrepõe --classe-codigo',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Define o nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL). Padrão: INFO",
    )
    args = parser.parse_args()

    set_log_level(args.log_level)

    try:
        print(
            f"⏳ Coletando dados para {', '.join(args.tribunais)} "
            f"(classe={args.classe_nome or args.classe_codigo or CLASSE_CODIGO_DEFAULT}) …"
        )
        df = build_dataframe(args.tribunais, args.classe_codigo, args.classe_nome)
    except EnvironmentError as err:
        print(f"⚠️  {err}")
        sys.exit(1)

    print(f"✔️  Total de processos: {len(df):,}")
    persist_df(df)
    plot_horario(df)


if __name__ == "__main__":
    main()
