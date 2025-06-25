import json
import sys
import unittest
from pathlib import Path
from unittest import mock
import tempfile

# allow import of legacy_datajud_connector.py
sys.path.append(str(Path(__file__).resolve().parents[1] / "legacy"))
import legacy_datajud_connector as ldc


class TestCLI(unittest.TestCase):
    def test_save_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            sample = [{"ok": True}]
            with mock.patch(
                "legacy_datajud_connector.fetch_esaj_tjce", return_value=sample
            ):
                test_args = [
                    "ldc",
                    "esaj",
                    "--classe",
                    "A",
                    "--save",
                    str(out),
                ]
                with mock.patch.object(sys, "argv", test_args):
                    ldc.cli()
            self.assertTrue(out.exists())
            data = json.loads(out.read_text())
            self.assertEqual(data, sample)


if __name__ == "__main__":
    unittest.main()
