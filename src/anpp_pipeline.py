# anpp_pipeline.py
'''Coleta, tratamento e análises dos ANPPs do TJSP via API‑CNJ

Melhorias incorporadas
----------------------
*   Paginação automática (>10 000 resultados)  
*   Chave da API lida de variável de ambiente (`CNJ_API_KEY`)  
*   Datas "aware" em UTC → convertidas para America/Sao_Paulo  
*   Pipeline modularizado em funções para facilitar testes  
*   Persistência em Parquet (`zstd`) + CSV opcional  
*   Gráfico de horário de ajuizamento

Executar:
    CNJ_API_KEY='APIKey …' python anpp_pipeline.py
'''
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import unittest
from unittest import mock

TRIBUNAL = 'TJSP'
CLASSE_CODIGO = 12729
PAGE_SIZE = 1000
BASE_URL = 'https://api-publica.datajud.cnj.jus.br/api_publica_tjsp/_search'

OUT_DIR = Path('dados_anpp').resolve()
OUT_DIR.mkdir(exist_ok=True, parents=True)


def get_headers() -> Dict[str, str]:
    api_key = os.getenv('CNJ_API_KEY')
    if not api_key:
        raise EnvironmentError('Defina a variável de ambiente CNJ_API_KEY antes de executar o script.')
    if not api_key.lower().startswith('apikey'):
        api_key = f'APIKey {api_key}'
    return {
        'Authorization': api_key,
        'Content-Type': 'application/json',
    }


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


def fetch_raw_hits(classe: int = CLASSE_CODIGO, page_size: int = PAGE_SIZE) -> Generator[Dict[str, Any], None, None]:
    headers = get_headers()
    payload_base = {
        'size': page_size,
        'query': {'match': {'classe.codigo': classe}},
        'sort': [{'dataAjuizamento': {'order': 'desc'}}, {'_id': 'asc'}],
    }
    search_after: Optional[List[Any]] = None
    while True:
        payload = dict(payload_base)
        if search_after is not None:
            payload['search_after'] = search_after
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        hits = resp.json().get('hits', {}).get('hits', [])
        if not hits:
            break
        yield from hits
        new_search_after = hits[-1]['sort']
        if new_search_after == search_after:
            break
        search_after = new_search_after


def parse_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit['_source']
    return {
        'numero_processo': src.get('numeroProcesso'),
        'classe': src.get('classe', {}).get('nome'),
        'data_ajuizamento': tz_utc_to_sp(src.get('dataAjuizamento')),
        'ultima_atualizacao': tz_utc_to_sp(src.get('dataHoraUltimaAtualizacao')),
        'formato': src.get('formato', {}).get('nome'),
        'codigo': src.get('orgaoJulgador', {}).get('codigo'),
        'orgao_julgador': src.get('orgaoJulgador', {}).get('nome'),
        'municipio': src.get('orgaoJulgador', {}).get('codigoMunicipioIBGE'),
        'grau': src.get('grau'),
        'assuntos': lista_assuntos(src.get('assuntos', [])),
        'movimentos': lista_movimentos(src.get('movimentos', [])),
        'sort': hit.get('sort', [None])[0],
    }


def build_dataframe() -> pd.DataFrame:
    registros = [parse_hit(h) for h in fetch_raw_hits()]
    df = pd.DataFrame(registros)
    return df


def persist_df(df: pd.DataFrame) -> None:
    if df.empty:
        print('Nenhum dado para persistir.')
        return
    parquet_path = OUT_DIR / 'anpp.parquet'
    csv_path = OUT_DIR / 'anpp.csv'
    df.to_parquet(parquet_path, compression='zstd', index=False)
    df.to_csv(csv_path, index=False)
    print(f'Dados salvos em:\n  • {parquet_path}\n  • {csv_path}')


def plot_horario(df: pd.DataFrame) -> None:
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
    plt.title('Horário de ajuizamento dos ANPPs (TJSP)')
    plt.xlabel('Hora do dia')
    plt.ylabel('Número de ajuizamentos')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    out_path = OUT_DIR / 'horario_anpp.jpg'
    plt.savefig(out_path, dpi=150)
    print(f'Gráfico salvo em {out_path}')
    plt.close()


def main() -> None:
    try:
        print('⏳ Coletando dados …')
        df = build_dataframe()
    except EnvironmentError as err:
        print(f'⚠️  {err}')
        return
    print(f'✔️  Total de processos: {len(df):,}')
    persist_df(df)
    if not df.empty:
        assuntos_top = df['assuntos'].explode().value_counts().head()
        print('\nTop‑5 assuntos:\n', assuntos_top, sep='')
    plot_horario(df)


class TestHelpers(unittest.TestCase):
    def setUp(self):
        matplotlib.use('Agg')

    def test_tz_utc_to_sp(self):
        ts = tz_utc_to_sp('2020-01-01T03:00:00Z')
        self.assertEqual(ts.tz.zone, 'America/Sao_Paulo')
        self.assertEqual(ts.hour, 0)

    def test_tz_utc_to_sp_none(self):
        self.assertIsNone(tz_utc_to_sp(None))

    def test_lista_assuntos(self):
        raw = [{'nome': 'Penal'}, {'nome': 'Civil'}]
        self.assertEqual(lista_assuntos(raw), ['Penal', 'Civil'])

    def test_lista_movimentos(self):
        raw = [
            {'codigo': 2, 'nome': 'B', 'dataHora': '2020-01-01T12:00:00Z'},
            {'codigo': 1, 'nome': 'A', 'dataHora': '2020-01-02T12:00:00Z'},
        ]
        result = lista_movimentos(raw)
        order = [r[0] for r in result]
        self.assertEqual(order, [2, 1])


class TestMain(unittest.TestCase):
    def test_main_handles_missing_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch('builtins.print') as mocked_print:
                main()
                mocked_print.assert_any_call('⚠️  Defina a variável de ambiente CNJ_API_KEY antes de executar o script.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=sys.argv[:1])
    else:
        main()
