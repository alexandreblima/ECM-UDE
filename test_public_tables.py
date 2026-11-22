import tempfile
import unittest
from pathlib import Path

from paper_v3_tables import tost_rows, write_table


class PublicTableTests(unittest.TestCase):
    def test_tost_table_formats_exact_values_and_excluded_holm(self):
        rows = tost_rows()
        self.assertEqual(len(rows), 60)
        first = next(r for r in rows if r[:3] == ["NCA", "UDDS_25C", "PG"])
        self.assertEqual(first[5:], ["<0.001", "0.020", "--", "--"])
        excluded = next(r for r in rows if r[:3] == ["NMC", "US06_25C", "OR"])
        self.assertEqual(excluded[7:], ["<0.001", "--"])

    def test_tost_latex_renders_small_p_in_math_mode(self):
        with tempfile.TemporaryDirectory() as output:
            path = Path(output)/"table9"
            write_table(path, ["p"], [["<0.001"]])
            self.assertIn(r"$<10^{-3}$", path.with_suffix(".tex").read_text())
