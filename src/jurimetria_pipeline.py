#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

CLASSE_CODIGO = 12729  # ANPP – mantido como padrão
PAGE_SIZE = 1000
DEFAULT_TRIBUNAIS = ['TJCE']  # padrão quando nenhum tribunal é informado

# Diretório de saída
OUT_DIR = Path('dados_jurimetria').resolve()
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Carrega lookup de municípios IBGE -> nome
MUNICIPIOS_CSV = Path('data/municipios_ibge.csv')
if MUNICIPIOS_CSV.exists():
    try:
        _mun = pd.read_csv(MUNICIPIOS_CSV, dtype={'codigo_municipio_ibge': str})
        _mun.set_index('codigo_municipio_ibge', inplace=True)
    except Exception:
        _mun = pd.DataFrame()
else:
    _mun = pd.DataFrame()

logger = logging.getLogger(__name__)


def get_headers() -> Dict[str, str]:
    api_key = os.getenv('CNJ_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Defina a variável de ambiente CNJ_API_KEY antes de executar o script.'
        )
    if not api_key.lower().startswith('apikey'):
        api_key = f'APIKey {api_key}'
    return {
        'Authorization': api_key,
        'Content-Type': 'application/json',
    }


def build_base_url(tribunal: str) -> str:
    return f'https://api-publica.datajud.cnj.jus.br/api_publica_{tribunal.lower()}/_search'


def tz_utc_to_sp(dt_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not dt_str:
        return None
    ts = pd.to_datetime(dt_str, utc=True)
    return ts.tz_convert('America/Sao_Paulo')


def lista_assuntos(raw_assuntos: List[Dict[str, Any]]) -> List[str]:
    return [a.get('nome', '') for a in raw_assuntos]


def lista_movimentos(raw_movs: List[Dict[str, Any]]) -> List[List[Any]]:
    movs: List[List[Any]] = []
    for mov in raw_movs:
        codigo = mov.get('codigo')
        nome = mov.get('nome')
        data_parsed = tz_utc_to_sp(mov.get('dataHora'))
        movs.append([codigo, nome, data_parsed])
    default_ts = pd.Timestamp('1970-01-01', tz='America/Sao_Paulo')
    return sorted(movs, key=lambda x: x[2] or default_ts)


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
    base_url = build_base_url(tribunal)

    # Monta filtros
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

    if filters:
        query: Dict[str, Any] = {'bool': {'must': filters}}
    else:
        query = {'match_all': {}}

    payload_base: Dict[str, Any] = {
        'size': page_size,
        'query': query,
        'sort': [
            {'dataAjuizamento': {'order': 'desc'}},
            {'_id': 'asc'}
        ],
    }

    retrieved = 0
    search_after: Optional[List[Any]] = None

    while True:
        payload = dict(payload_base)
        if search_after is not None:
            payload['search_after'] = search_after

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Request payload: %s", payload)
        else:
            logger.info("Buscando %d processos em %s...", page_size, tribunal)

        resp = requests.post(base_url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Response [%d]: %s", resp.status_code, resp.text)

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

    cod_ibge = src.get('orgaoJulgador', {}).get('codigoMunicipioIBGE')
    nome_mun: Optional[str] = None
    if cod_ibge is not None:
        cod_str = str(cod_ibge)
        if not _mun.empty and cod_str in _mun.index:
            try:
                nome_mun = _mun.loc[cod_str, 'nome_municipio']
            except Exception:
                nome_mun = None

    return {
        'tribunal': tribunal,
        'numero_processo': src.get('numeroProcesso'),
        'classe': src.get('classe', {}).get('nome'),
        'data_ajuizamento': tz_utc_to_sp(src.get('dataAjuizamento')),
        'ultima_atualizacao': tz_utc_to_sp(src.get('dataHoraUltimaAtualizacao')),
        'formato': src.get('formato', {}).get('nome'),
        'codigo': src.get('orgaoJulgador', {}).get('codigo'),
        'orgao_julgador': src.get('orgaoJulgador', {}).get('nome'),
        'municipio': cod_ibge,
        'municipio_nome': nome_mun,
        'grau': src.get('grau'),
        'assuntos': lista_assuntos(src.get('assuntos', [])),
        'movimentos': lista_movimentos(src.get('movimentos', [])),
        'sort': hit.get('sort', [None])[0],
    }


def build_dataframe(
    tribunais: List[str] = DEFAULT_TRIBUNAIS,
    classe_codigo: Optional[int] = CLASSE_CODIGO,
    classe_nome: Optional[str] = None,
    de: Optional[str] = None,
    ate: Optional[str] = None,
    max_processos: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trib in tribunais:
        registros = [
            parse_hit(h, trib)
            for h in fetch_raw_hits(
                trib, classe_codigo, classe_nome, de, ate, PAGE_SIZE, max_processos
            )
        ]
        if registros:
            frames.append(pd.DataFrame(registros))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print('Nenhum dado para persistir.')
        return
    parquet_path = OUT_DIR / 'jurimetria.parquet'
    csv_path = OUT_DIR / 'jurimetria.csv'
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pipeline de Jurimetria via API pública do CNJ'
    )
    parser.add_argument(
        '--tribunais',
        nargs='+',
        help='Lista de tribunais (ex.: TJCE TJSP). Se omitido, padrão é TJCE.',
    )
    parser.add_argument(
        '--classe-codigo',
        dest='classe_codigo',
        type=int,
        default=CLASSE_CODIGO,
        help=f'Código da classe (padrão: {CLASSE_CODIGO}).',
    )
    parser.add_argument(
        '--classe',
        dest='classe_nome',
        type=str,
        default=None,
        help='Nome da classe (ex.: "Apelação Cível").',
    )
    parser.add_argument(
        '--de',
        dest='de',
        type=str,
        default=None,
        help='Data inicial (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--ate',
        dest='ate',
        type=str,
        default=None,
        help='Data final (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--max-processos',
        dest='max_processos',
        type=int,
        default=None,
        help='Máximo de processos a extrair.',
    )
    parser.add_argument(
        '--log-level',
        dest='log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Nível de log.',
    )
    # ignora flags não reconhecidas (ex.: pytest -q)
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(levelname)s] %(message)s'
    )

    tribunais = args.tribunais or DEFAULT_TRIBUNAIS

    try:
        print(f'⏳ Coletando dados para: {", ".join(tribunais)} …')
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
        return

    print(f'✔️  Total de processos: {len(df):,}')
    persist_df(df)

    if not df.empty:
        assuntos_top = df['assuntos'].explode().value_counts().head()
        print('\nTop-5 assuntos:\n', assuntos_top, sep='')

    plot_horario(df, args.classe_nome, args.classe_codigo)


if __name__ == '__main__':
    main()
