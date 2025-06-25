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
    """Ajusta o nível de log global."""
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        log.error(f"Invalid log level: {level_name}")
        sys.exit(1)
    logging.getLogger().setLevel(level)


def get_headers() -> Dict[str, str]:
    """Lê CNJ_API_KEY do ambiente e monta header Authorization."""
    api_key = os.getenv("CNJ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Defina a variável de ambiente CNJ_API_KEY antes de executar o script.")
    if not api_key.lower().startswith("apikey"):
        api_key = f"APIKey {api_key}"
    return {"Authorization": api_key, "Content-Type": "application/json"}


def build_base_url(sigla: str) -> str:
    """Monta o endpoint do índice público de cada tribunal."""
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
    nomes: List[str] = []
    for a in raw:
        if isinstance(a, dict):
            nomes.append(a.get("nome", ""))
        elif isinstance(a, list) and a:
            first = a[0]
            nomes.append(first.get("nome", "") if isinstance(first, dict) else "")
        else:
            nomes.append("")
    return nomes


def lista_movimentos(raw: List[Dict[str, Any]]) -> List[List[Any]]:
    movs: List[List[Any]] = []
    for m in raw:
        movs.append([m.get("codigo"), m.get("nome"), tz_utc_to_sp(m.get("dataHora"))])
    default = pd.Timestamp("1970-01-01", tz="America/Sao_Paulo")
    return sorted(movs, key=lambda x: x[2] or default)


def _build_query(
    classe_codigo: Optional[int],
    classe_nome: Optional[str],
    dt_ini: Optional[str],
    dt_fim: Optional[str],
) -> Dict[str, Any]:
    filtros: List[Dict[str, Any]] = []

    # filtro de classe
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
    """Paginação `search_after` sobre o índice público de cada tribunal, com contador."""
    headers = get_headers()
    url = build_base_url(tribunal)
    total = 0

    def do_request() -> Generator[Dict[str, Any], None, None]:
        base_payload = {
            "size": page_size,
            "query": _build_query(classe_codigo, classe_nome, dt_ini, dt_fim),
            "sort": [
                {"dataAjuizamento": {"order": "desc"}},
                {"_id": "asc"}
            ],
        }
        search_after: Optional[List[Any]] = None

        while True:
            payload = dict(base_payload)
            if search_after is not None:
                payload["search_after"] = search_after

            # controle de logs
            if log.level == logging.DEBUG:
                log.debug(f"[{tribunal}] Payload: {payload}")
            else:
                log.debug(f"[{tribunal}] Enviando request. (detalhes em DEBUG)")

            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code in (400, 404):
                log.warning("Tribunal %s retornou %s – pulando.", tribunal, resp.status_code)
                return
            resp.raise_for_status()

            data = resp.json()
            if log.level == logging.DEBUG:
                log.debug(f"[{tribunal}] Resp JSON: {data}")
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                return

            yield from hits
            if max_total and (total + len(hits)) >= max_total:
                return

            search_after = hits[-1]["sort"]
            # se veio menos que page_size, acabou
            if len(hits) < page_size:
                return

    # dispara único fluxo (sem fallback por nome/código, pois query já inclui ambos)
    for hit in do_request():
        total += 1
        yield hit
        if max_total and total >= max_total:
            return


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
    tribunais: List[str] = DEFAULT_TRIBUNAIS,
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
    de: Optional[str] = None,
    ate: Optional[str] = None,
    max_processos: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trib in tribunais:
        registros = [
            parse_hit(h, trib)
            for h in fetch_raw_hits(trib, classe_codigo, classe_nome, dt_ini, dt_fim, PAGE_SIZE, max_proc)
        ]
        if registros:
            frames.append(pd.DataFrame(registros))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("Nenhum dado para persistir.")
        return
    parquet = OUT_DIR / "jurimetria.parquet"
    csv = OUT_DIR / "jurimetria.csv"
    df.to_parquet(parquet, compression="zstd", index=False)
    df.to_csv(csv, index=False)
    print(f"Dados salvos em:\n  • {parquet}\n  • {csv}")


def plot_horario(df: pd.DataFrame, classe_label: str | int) -> None:
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
    plt.title(f"Horário de ajuizamento – classe {classe_label}")
    plt.xlabel("Hora do dia")
    plt.ylabel("Processos")
    plt.tight_layout()
    out = OUT_DIR / "horario_jurimetria.jpg"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {out}")


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
    group.add_argument("--classe", dest="classe_nome", help='Nome da classe (ex.: "Apelação Cível")')
    parser.add_argument("--de", dest="dt_ini", help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--ate", dest="dt_fim", help="Data final (YYYY-MM-DD)")
    parser.add_argument("--max-processos", type=int, dest="max_proc", help="Limite de processos a extrair")
    parser.add_argument("--log-level", default="INFO", help="Nível de log (DEBUG, INFO…)")
    args, _ = parser.parse_known_args()

    set_log_level(args.log_level)

    try:
        print(
            f"⏳ Coletando dados para {', '.join(args.tribunais)} "
            f"(classe={args.classe_nome or args.classe_codigo or CLASSE_CODIGO_DEFAULT}"
            f"{', de=' + args.dt_ini if args.dt_ini else ''}"
            f"{', até=' + args.dt_fim if args.dt_fim else ''}"
            f"{', max=' + str(args.max_proc) if args.max_proc else ''}) …"
        )
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
