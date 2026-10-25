"""Birim testler: python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    REQUIRED,
    build_features,
    feature_groups,
    fit_ols,
    grouped_permutation_importance,
    load_listings,
    mae,
    mape,
    median_ape,
    ols_predict,
    partial_dependence_age,
    pct_effect,
    r2,
)


def toy_listings(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "model": rng.choice(["A", "B", "C"], n),
            "year": rng.integers(2014, 2027, n),
            "km": rng.integers(1_000, 200_000, n),
            "fuel": rng.choice(["Benzin", "Dizel", "LPG"], n),
            "gearbox": rng.choice(["Manuel", "Otomatik"], n),
            "damage": rng.integers(0, 2, n),
            "seller_type": rng.choice(["Sahibinden", "Galeri"], n),
            "city": rng.choice(["İstanbul", "Diğer", "Ankara"], n),
            "listing_month": rng.choice(["2026-01", "2026-02", "2026-03"], n),
            "price": rng.integers(300_000, 3_000_000, n),
        }
    )


class TestFeatures(unittest.TestCase):
    def test_reference_levels_are_dropped(self):
        X = build_features(toy_listings())
        for ref in ["fuel_Benzin", "gearbox_Manuel", "seller_Sahibinden", "city_Diğer", "model_A"]:
            self.assertNotIn(ref, X.columns)
        for col in ["fuel_Dizel", "fuel_LPG", "gearbox_Otomatik", "seller_Galeri", "city_İstanbul", "model_B", "model_C"]:
            self.assertIn(col, X.columns)

    def test_missing_reference_level_falls_back_to_first_level(self):
        """Veride 'Diğer' yoksa tüm şehirler dummy olursa sabit terimle doğrusal bağıntılı olur; bu engellenmeli."""
        df = toy_listings()
        df["city"] = df["city"].replace({"Diğer": "Ankara"})  # artık yalnızca İstanbul ve Ankara var
        X = build_features(df)
        city_cols = [c for c in X.columns if c.startswith("city_")]
        self.assertEqual(city_cols, ["city_İstanbul"])  # Ankara (alfabetik ilk) referans oldu
        fit_ols(X, np.log(df["price"].to_numpy(float)))  # tekil matris hatası vermemeli

    def test_numeric_features(self):
        df = toy_listings()
        X = build_features(df, ref_year=2026)
        np.testing.assert_array_equal(X["age"], 2026 - df["year"])
        np.testing.assert_array_equal(X["age_sq"], (2026 - df["year"]) ** 2)
        np.testing.assert_allclose(X["km_10k"], df["km"] / 10_000)
        self.assertEqual(set(X["month_idx"].unique()), {0.0, 1.0, 2.0})

    def test_feature_groups(self):
        groups = feature_groups(["age", "age_sq", "km_10k", "fuel_Dizel", "fuel_LPG", "model_B", "seller_Galeri"])
        self.assertEqual(groups["age (yaş)"], ["age", "age_sq"])
        self.assertEqual(groups["yakıt"], ["fuel_Dizel", "fuel_LPG"])
        self.assertIn("satıcı tipi", groups)


class TestLoad(unittest.TestCase):
    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            toy_listings().drop(columns=["km"]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_listings(path)

    def test_non_positive_price_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            df = toy_listings()
            df.loc[0, "price"] = 0
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_listings(path)

    def test_valid_file_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            toy_listings().to_csv(path, index=False)
            self.assertEqual(list(load_listings(path).columns), REQUIRED)


class TestOLS(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 2000
        self.X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 2, n).astype(float)})
        self.true = {"intercept": 2.0, "a": -0.5, "b": 0.3}
        self.y = 2.0 - 0.5 * self.X["a"] + 0.3 * self.X["b"] + rng.normal(0, 0.1, n)

    def test_recovers_coefficients_within_ci(self):
        t = fit_ols(self.X, self.y.to_numpy())
        for name, value in self.true.items():
            self.assertTrue(t.loc[name, "ci_low"] <= value <= t.loc[name, "ci_high"], name)

    def test_matches_lstsq_and_predicts(self):
        t = fit_ols(self.X, self.y.to_numpy())
        A = np.column_stack([np.ones(len(self.X)), self.X.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, self.y.to_numpy(), rcond=None)
        np.testing.assert_allclose(t["coef"].to_numpy(), beta)
        np.testing.assert_allclose(ols_predict(t, self.X), A @ beta)
        self.assertTrue((t["se"] > 0).all())

    def test_pct_effect(self):
        self.assertAlmostEqual(pct_effect(np.log(1.10)), 0.10)
        self.assertAlmostEqual(pct_effect(0.0), 0.0)
        self.assertLess(pct_effect(-0.12), 0)


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        a = np.array([100.0, 200.0, 300.0])
        self.assertEqual(mape(a, a), 0.0)
        self.assertEqual(mae(a, a), 0.0)
        self.assertEqual(r2(a, a), 1.0)

    def test_values(self):
        self.assertAlmostEqual(mape([100, 200], [110, 180]), 0.10)
        self.assertAlmostEqual(mae([100, 200], [110, 180]), 15.0)
        self.assertAlmostEqual(median_ape([100, 200, 400], [110, 200, 400]), 0.0)


class TestImportanceAndPD(unittest.TestCase):
    def test_important_variable_ranks_first(self):
        rng = np.random.default_rng(2)
        n = 1500
        X = pd.DataFrame({"km_10k": rng.normal(size=n), "damage": rng.integers(0, 2, n).astype(float)})
        y = 3.0 * X["km_10k"].to_numpy()  # damage hiç etkili değil
        predict = lambda Z: 3.0 * Z["km_10k"].to_numpy()  # noqa: E731
        imp = grouped_permutation_importance(predict, X, y, n_repeats=5, seed=0)
        self.assertEqual(imp.iloc[0]["variable"], "km")
        self.assertAlmostEqual(float(imp.set_index("variable").loc["hasar kaydı", "rmse_increase"]), 0.0, places=9)

    def test_partial_dependence_is_monotone_for_decreasing_effect(self):
        X = pd.DataFrame({"age": np.zeros(50), "age_sq": np.zeros(50)})
        pd_curve = partial_dependence_age(lambda Z: 14.0 - 0.1 * Z["age"].to_numpy(), X, range(0, 10))
        self.assertTrue(pd_curve.is_monotonic_decreasing)
        self.assertAlmostEqual(pd_curve.iloc[1] / pd_curve.iloc[0], np.exp(-0.1))


class TestGeneratedData(unittest.TestCase):
    def test_generated_file_schema_if_present(self):
        path = ROOT / "data" / "listings.csv"
        if not path.exists():
            self.skipTest("data/listings.csv yok (önce generate_data.py çalıştırın)")
        df = load_listings(path)
        self.assertTrue(df["damage"].isin([0, 1]).all())
        self.assertTrue((df["km"] > 0).all())
        self.assertTrue(df["year"].between(2000, 2026).all())


if __name__ == "__main__":
    unittest.main()
