#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO BÁSICA
# ──────────────────────────────────────────────────────────────
CLASSE_CODIGO = 12729          # ANPP – mantido como padrão
PAGE_SIZE = 1000
DEFAULT_TRIBUNAIS = ['TJCE']   # padrão quando nenhum tribunal é informado
OUT_DIR = Path('dados_jurimetria').resolve()
OUT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────
def get_headers() -> Dict[str, str]:
    api_key = os.getenv('CNJ_API_KEY')
    if not api_key:
        print('⚠️  Defina a variável de ambiente CNJ_API_KEY antes de executar o script.')
        print('⚠️  A variável de ambiente CNJ_API_KEY não está definida.')
        print('Defina‑a antes de executar o script. Ex.:')
        print('  export CNJ_API_KEY="sua_chave_aqui"  # Linux/macOS')
        print('  $env:CNJ_API_KEY="sua_chave_aqui"    # Windows PowerShell')
        sys.exit(1)
    if not api_key.lower().startswith('apikey'):
        api_key = f'APIKey {api_key}'
    return {
        'Authorization': api_key,
        'Content-Type': 'application/json',
    }


def build_base_url(tribunal: str) -> str:
    return f'https://api-publica.datajud.cnj.jus.br/api_publica_{tribunal.lower()}/_search'


def tz_utc_to_sp(dt_str: Optional[str]) -> Optional[pd.Timestamp]:
    """
    Converte string UTC (ISO 8601) para fuso America/Sao_Paulo,
    tolerando datas fora do intervalo suportado pelo pandas.
    Retorna None se a conversão falhar.
    """
    if not dt_str:
        return None

    ts = pd.to_datetime(dt_str, utc=True, errors="coerce")  # evita OutOfBounds
    if ts is pd.NaT:
        logger.warning("Out‑of‑bounds ou data inválida: %s", dt_str)
        return None
    return ts.tz_convert('America/Sao_Paulo')


def lista_assuntos(raw_assuntos: List[Dict[str, Any]]) -> List[str]:
    nomes = []
    for a in raw_assuntos:
        if isinstance(a, dict):
            nomes.append(a.get('nome', ''))
        else:
            nomes.append('')
    return nomes


def lista_movimentos(raw_movs: List[Dict[str, Any]]) -> List[List[Any]]:
    movs: List[List[Any]] = []
    for mov in raw_movs:
        codigo = mov.get('codigo')
        nome = mov.get('nome')
        data_parsed = tz_utc_to_sp(mov.get('dataHora'))
        movs.append([codigo, nome, data_parsed])
    default_ts = pd.Timestamp('1970-01-01', tz='America/Sao_Paulo')
    return sorted(movs, key=lambda x: x[2] or default_ts)

# ──────────────────────────────────────────────────────────────
# COLETA (fetch) E PARSE
# ──────────────────────────────────────────────────────────────
def fetch_raw_hits(
    tribunal: str,
    classe_codigo: Optional[int] = CLASSE_CODIGO,
    classe_nome: Optional[str] = None,
    dt_ini: Optional[str] = None,
    dt_fim: Optional[str] = None,
    page_size: int = PAGE_SIZE,
    max_processos: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    headers = get_headers()
    tribunal_lower = tribunal.lower()
    base_url = build_base_url(tribunal_lower)

    # filtros
    filters: List[Dict[str, Any]] = []
    if classe_codigo is not None:
        filters.append({'term': {'classe.codigo': classe_codigo}})
    if classe_nome:
        filters.append({'term': {'classe.nome.keyword': classe_nome}})
    if dt_ini or dt_fim:
        range_filter: Dict[str, Any] = {}
        if dt_ini:
            range_filter['gte'] = dt_ini
        if dt_fim:
            range_filter['lte'] = dt_fim
        filters.append({'range': {'dataAjuizamento': range_filter}})

    query = {'bool': {'must': filters}} if filters else {'match_all': {}}

    payload_base: Dict[str, Any] = {
        'size': page_size,
        'query': query,
        'sort': [{'dataAjuizamento': {'order': 'desc'}}],
    }

    retrieved = 0
    search_after: Optional[List[Any]] = None

    while True:
        payload = dict(payload_base)
        if search_after is not None:
            payload['search_after'] = search_after

        logger.info("Buscando %d processos em %s…", page_size, tribunal_lower)
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error %s: %s", resp.status_code, resp.text)
            raise e

        hits = resp.json().get('hits', {}).get('hits', [])
        if not hits:
            break

        for hit in hits:
            yield hit
            retrieved += 1
            if max_processos is not None and retrieved >= max_processos:
                return

        new_search_after = hits[-1].get('sort')
        if new_search_after == search_after:
            break
        search_after = new_search_after


def parse_hit(hit: Dict[str, Any], tribunal: str) -> Dict[str, Any]:
    src = hit.get('_source', {})
    data_ajuizamento = tz_utc_to_sp(src.get('dataAjuizamento'))
    ultima_atualizacao = tz_utc_to_sp(src.get('dataHoraUltimaAtualizacao'))

    return {
        'tribunal': tribunal,
        'numero_processo': src.get('numeroProcesso'),
        'classe': src.get('classe', {}).get('nome'),
        'data_ajuizamento': data_ajuizamento,
        'ultima_atualizacao': ultima_atualizacao,
        'formato': src.get('formato', {}).get('nome'),
        'codigo': src.get('orgaoJulgador', {}).get('codigo'),
        'orgao_julgador': src.get('orgaoJulgador', {}).get('nome'),
        'municipio': src.get('orgaoJulgador', {}).get('codigoMunicipioIBGE'),
         nome = MUNICIPIOS.loc[str(cod), "nome_municipio"] if cod and str(cod) in MUNICIPIOS.index else None
    return {
        …
        "municipio_codigo_ibge": cod,
        "municipio": nome,
        …
    }
        'grau': src.get('grau'),
        'assuntos': lista_assuntos(src.get('assuntos', [])),
        'movimentos': lista_movimentos(src.get('movimentos', [])),
        'sort': hit.get('sort', [None])[0],
    }

# ──────────────────────────────────────────────────────────────
# DATAFRAME, PERSISTÊNCIA E PLOT
# ──────────────────────────────────────────────────────────────
def build_dataframe(
    tribunais: List[str] = DEFAULT_TRIBUNAIS,
    classe_codigo: Optional[int] = None,
    classe_nome: Optional[str] = None,
    de: Optional[str] = None,
    ate: Optional[str] = None,
    max_processos: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    total_filtered = 0
    for trib in tribunais:
        registros = [
            parse_hit(h, trib)
            for h in fetch_raw_hits(
                trib, classe_codigo, classe_nome, de, ate, PAGE_SIZE, max_processos
            )
        ]
        # Filter out records with invalid or None data_ajuizamento
        registros_validos = [r for r in registros if r.get('data_ajuizamento') is not None]
        filtered_count = len(registros) - len(registros_validos)
        total_filtered += filtered_count
        if filtered_count > 0:
            logger.warning("Filtrados %d registros com datas inválidas em %s", filtered_count, trib)
        if registros_validos:
            frames.append(pd.DataFrame(registros_validos))
    if total_filtered > 0:
        logger.info("Total de registros filtrados por datas inválidas: %d", total_filtered)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print('Nenhum dado para persistir.')
        return
    parquet_path = OUT_DIR / 'jurimetria.parquet'
    csv_path = OUT_DIR / 'jurimetria.csv'
    # Convert 'movimentos' column to JSON strings to avoid pyarrow serialization errors
    if 'movimentos' in df.columns:
        import json
        df = df.copy()
        # Convert Timestamp objects inside 'movimentos' to string before json.dumps
        def convert_timestamps(obj):
            if isinstance(obj, list):
                return [convert_timestamps(i) for i in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        df['movimentos'] = df['movimentos'].apply(lambda x: json.dumps(convert_timestamps(x)))
    df.to_parquet(parquet_path, compression='zstd', index=False)
    df.to_csv(csv_path, index=False)
    print(f'Dados salvos em:\n  • {parquet_path}\n  • {csv_path}')


def plot_horario(
    df: pd.DataFrame,
    classe_nome: Optional[str] = None,
    classe_codigo: Optional[int] = None,
) -> None:
    if df.empty:
        print('Nenhum dado para plotar.')
        return

    horas = (
        pd.to_datetime(df['data_ajuizamento'], errors='coerce', utc=True)
        .dropna()
        .dt.tz_convert('America/Sao_Paulo')
        .dt.hour
    )
    if horas.empty:
        print('Nenhum dado válido de horário para plotar.')
        return

    contagem = horas.value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    contagem.plot(kind='bar')
    plt.title(f"Horário de ajuizamento – classe {classe_nome or classe_codigo}")
    plt.xlabel('Hora do dia')
    plt.ylabel('Número de ajuizamentos')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    out_path = OUT_DIR / 'horario_jurimetria.jpg'
    plt.savefig(out_path, dpi=150)
    print(f'Gráfico salvo em {out_path}')
    plt.close()

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pipeline de Jurimetria via API pública do CNJ'
    )
    parser.add_argument('--tribunais', nargs='+',
                        help='Lista de tribunais (ex.: TJCE TJSP).')
    parser.add_argument('--classe-codigo', type=int, default=None,
                        help='Código da classe (ex.: 12729).')
    parser.add_argument('--classe', dest='classe_nome', type=str, default=None,
                        help='Nome da classe (ex.: "Apelação Cível").')
    parser.add_argument('--de', type=str, default=None,
                        help='Data inicial (YYYY-MM-DD).')
    parser.add_argument('--ate', type=str, default=None,
                        help='Data final (YYYY-MM-DD).')
    parser.add_argument('--max-processos', dest='max_processos',
                        type=int, default=None,
                        help='Máximo de processos a extrair.')
    parser.add_argument('--log-level', dest='log_level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Nível de log.')
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(levelname)s] %(message)s'
    )

    tribunais = args.tribunais if args.tribunais else DEFAULT_TRIBUNAIS
    print(f'⏳ Coletando dados para: {", ".join(tribunais)} …')

    try:
        df = build_dataframe(
            tribunais=tribunais,
            classe_codigo=args.classe_codigo,
            classe_nome=args.classe_nome,
            de=args.de,
            ate=args.ate,
            max_processos=args.max_processos,
        )
    except EnvironmentError as err:
        print(f'⚠️  {err}')
        sys.exit(1)

    print(f'✔️  Total de processos: {len(df):,}')
    persist_df(df)

    if not df.empty:
        assuntos_top = df['assuntos'].explode().value_counts().head()
        print('\nTop-5 assuntos:\n', assuntos_top, sep='')

    plot_horario(df, args.classe_nome, args.classe_codigo)


if __name__ == '__main__':
    main()
