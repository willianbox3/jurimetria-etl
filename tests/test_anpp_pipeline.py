"""
Testes unitários do pipeline ANPP (TJSP)

Comando:
    pytest -q
ou
    python -m unittest
"""
import os
from pathlib import Path
import unittest
from unittest import mock

import pandas as pd
import matplotlib

# Permite importar src/jurimetria_pipeline.py sem instalar como pacote
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from jurimetria_pipeline import (
    tz_utc_to_sp, lista_assuntos, lista_movimentos,
    build_dataframe, main
)

matplotlib.use("Agg")   # evita abrir janelas gráficas em CI


class TestHelpers(unittest.TestCase):
    def test_tz_utc_to_sp(self):
        ts = tz_utc_to_sp("2020-01-01T03:00:00Z")
        self.assertEqual(ts.tz.zone, "America/Sao_Paulo")
        self.assertEqual(ts.hour, 0)

    def test_tz_utc_to_sp_none(self):
        self.assertIsNone(tz_utc_to_sp(None))

    def test_lista_assuntos(self):
        raw = [{"nome": "Penal"}, {"nome": "Civil"}]
        self.assertEqual(lista_assuntos(raw), ["Penal", "Civil"])

    def test_lista_movimentos_ordena_por_data(self):
        raw = [
            {"codigo": 2, "nome": "B", "dataHora": "2020-01-01T12:00:00Z"},
            {"codigo": 1, "nome": "A", "dataHora": "2020-01-02T12:00:00Z"},
        ]
        order = [r[0] for r in lista_movimentos(raw)]
        self.assertEqual(order, [2, 1])


class TestMain(unittest.TestCase):
    def test_main_handles_missing_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.print") as mocked_print:
                main()
                mocked_print.assert_any_call(
                    "⚠️  Defina a variável de ambiente CNJ_API_KEY antes de executar o script."
                )

    def test_build_dataframe_sem_api_real(self):
        """Smoke-test do build_dataframe com request simulado."""
        sample_hit = {
            "_source": {
                "numeroProcesso": "0000001-00.2023.8.26.0000",
                "classe": {"nome": "Acao Penal"},
                "dataAjuizamento": "2023-01-10T12:00:00Z",
                "dataHoraUltimaAtualizacao": "2023-01-11T12:00:00Z",
                "formato": {"nome": "Digital"},
                "orgaoJulgador": {
                    "codigo": "123456",
                    "nome": "1ª Vara",
                    "codigoMunicipioIBGE": "3550308"
                },
                "grau": "1º Grau",
                "assuntos": [{"nome": "Penal"}],
                "movimentos": []
            },
            "sort": [123]
        }

        with mock.patch("jurimetria_pipeline.fetch_raw_hits", return_value=[sample_hit]):
            df = build_dataframe()
            self.assertEqual(len(df), 1)
            self.assertIn("numero_processo", df.columns)


if __name__ == "__main__":
    unittest.main()
