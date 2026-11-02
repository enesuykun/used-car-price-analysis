"""Uçtan uca testler: üret -> analiz et -> çıktı dosyaları ve kurulan etkilerin geri bulunması."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_data as g  # noqa: E402
import run_analysis  # noqa: E402
from analysis import build_features, fit_ols  # noqa: E402


class TestOLSRecoversPlantedEffects(unittest.TestCase):
    """Doğru kurulmuş OLS, veri üreticisindeki gerçek etkileri güven aralığı içinde bulmalı."""

    @classmethod
    def setUpClass(cls):
        df = g.generate(n=12_000, seed=5)
        cls.table = fit_ols(build_features(df, age_by_model=True), np.log(df.price.to_numpy(float)))
        cls.general, cls.body = run_analysis.planted_effects()

    def _coverage(self, planted: dict) -> float:
        hits = [self.table.loc[k, "ci_low"] - 1e-9 <= v <= self.table.loc[k, "ci_high"] + 1e-9
                for k, v in planted.items() if k in self.table.index]
        return np.mean(hits), len(hits)

    def test_general_effects(self):
        cov, n = self._coverage(self.general)
        self.assertGreaterEqual(n, 15)
        self.assertGreaterEqual(cov, 0.85)  # %95 GA: rastgele birkaç kaçırma olabilir

    def test_body_effects(self):
        cov, n = self._coverage(self.body)
        self.assertEqual(n, 21)
        self.assertGreaterEqual(cov, 0.85)

    def test_roof_replacement_is_the_costliest_body_work(self):
        body_cols = [c for c in self.table.index if c.startswith("body_")]
        self.assertEqual(self.table.loc[body_cols, "coef"].idxmin(), "body_tavan_degisen")


class TestRunAnalysisSmoke(unittest.TestCase):
    """Hızlı modda tüm analiz hattı geçici klasöre çalışır ve beklenen dosyaları üretir."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.csv = d / "listings.csv"
        g.generate(n=3000, seed=9).to_csv(cls.csv, index=False)
        cls.res, cls.img = d / "results", d / "images"
        cls.out = run_analysis.main(["--csv", str(cls.csv), "--results-dir", str(cls.res), "--images-dir", str(cls.img),
                                     "--fast", "--synthetic", "yes"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_output_files(self):
        for f in ["model_comparison.csv", "ols_effects.csv", "body_part_effects.csv", "market_index.csv",
                  "permutation_importance.csv", "age_depreciation_curve.csv", "model_depreciation.csv", "test_predictions.csv"]:
            self.assertTrue((self.res / f).exists(), f)
        for f in ["eda.png", "model_comparison.png", "ols_effects.png", "body_part_effects.png", "market_index.png",
                  "permutation_importance.png", "age_depreciation.png"]:
            self.assertGreater((self.img / f).stat().st_size, 5_000, f)

    def test_models_beat_baseline(self):
        s = self.out["test_scores"]["MAPE"]
        for m in run_analysis.MODELS:
            self.assertLess(s[m], s[run_analysis.BASELINE], m)

    def test_market_index_tracks_planted(self):
        idx = self.out["index"]
        err = (np.log(idx.ols_index) - np.log(idx.planted_index)).abs()
        self.assertLess(err.mean(), 0.06)

    def test_effects_tables(self):
        eff = pd.read_csv(self.res / "ols_effects.csv")
        self.assertTrue({"variable", "pct_effect", "pct_low", "pct_high", "planted_pct"} <= set(eff.columns))
        self.assertTrue((eff.pct_low <= eff.pct_effect).all() and (eff.pct_effect <= eff.pct_high).all())
        body = pd.read_csv(self.res / "body_part_effects.csv")
        # 7 bölge x 3 işlem = 21; küçük veride hiç görülmeyen nadir bir durum (ör. değişen tavan) atlanır
        self.assertTrue(18 <= len(body) <= 21, len(body))
        self.assertTrue(body.ci_covers_planted.mean() >= 0.8)

    def test_user_csv_without_optional_columns(self):
        """Yalnızca zorunlu sütunlu kullanıcı verisinde kaporta çıktısı olmadan çalışmalı."""
        from analysis import REQUIRED

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            csv = d / "user.csv"
            pd.read_csv(self.csv)[REQUIRED].to_csv(csv, index=False)
            out = run_analysis.main(["--csv", str(csv), "--results-dir", str(d / "r"), "--images-dir", str(d / "i"), "--fast"])
            self.assertEqual(len(out["body"]), 0)
            self.assertFalse((d / "r" / "body_part_effects.csv").exists())
            self.assertNotIn("planted_index", out["index"].columns)


if __name__ == "__main__":
    unittest.main()
