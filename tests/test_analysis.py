"""analysis.py birim testleri: python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    BODY_PARTS,
    PART_REGION,
    REGIONS,
    REQUIRED,
    STATE_KEYS,
    body_effects,
    body_loss_breakdown,
    body_region_counts,
    build_features,
    drop_constant,
    feature_groups,
    fit_ols,
    grouped_permutation_importance,
    has_body_parts,
    listing_age,
    load_listings,
    mae,
    mape,
    median_ape,
    model_depreciation,
    ols_predict,
    partial_dependence_age,
    pct_effect,
    period_index,
    r2,
)


def toy_listings(n=300, seed=0, extended=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "model": rng.choice(["A", "B", "C"], n),
            "year": rng.integers(2014, 2026, n),
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
    if extended:
        df["color"] = rng.choice(["Beyaz", "Gri", "Siyah"], n)
        df["engine_cc"] = rng.choice([999, 1332, 1598], n)
        df["trim_level"] = rng.choice(["Temel", "Orta", "Üst"], n)
        for p in BODY_PARTS:
            df[p] = rng.choice(["Orijinal", "Lokal boyalı", "Boyalı", "Değişen"], n, p=[0.7, 0.1, 0.12, 0.08])
    return df


class TestFeatures(unittest.TestCase):
    def test_reference_levels_are_dropped(self):
        X = build_features(toy_listings())
        for ref in ["fuel_Benzin", "gearbox_Manuel", "seller_Sahibinden", "city_Diğer", "model_A", "color_Beyaz",
                    "trim_Temel", "period_2026-01"]:
            self.assertNotIn(ref, X.columns)
        for col in ["fuel_Dizel", "fuel_LPG", "gearbox_Otomatik", "seller_Galeri", "city_İstanbul", "model_B", "model_C",
                    "color_Gri", "color_Siyah", "trim_Orta", "trim_Üst", "period_2026-02", "period_2026-03", "engine_100cc"]:
            self.assertIn(col, X.columns)

    def test_missing_reference_level_falls_back_to_first_level(self):
        """Veride 'Diğer' yoksa tüm şehirler dummy olursa sabit terimle doğrusal bağıntılı olur; bu engellenmeli."""
        df = toy_listings()
        df["city"] = df["city"].replace({"Diğer": "Ankara"})  # artık yalnızca İstanbul ve Ankara var
        X = build_features(df)
        city_cols = [c for c in X.columns if c.startswith("city_")]
        self.assertEqual(city_cols, ["city_İstanbul"])  # Ankara (alfabetik ilk) referans oldu
        fit_ols(X, np.log(df["price"].to_numpy(float)))  # tekil matris hatası vermemeli

    def test_numeric_features_use_listing_year_for_age(self):
        df = toy_listings()
        df["listing_month"] = np.where(np.arange(len(df)) % 2 == 0, "2020-05", "2026-03")
        X = build_features(df)
        expected_age = np.where(np.arange(len(df)) % 2 == 0, 2020, 2026) - df["year"]
        np.testing.assert_array_equal(X["age"], np.clip(expected_age, 0, None))
        np.testing.assert_array_equal(X["age_sq"], X["age"] ** 2)
        np.testing.assert_allclose(X["km_10k"], df["km"] / 10_000)
        np.testing.assert_allclose(X["engine_100cc"], df["engine_cc"] / 100)

    def test_listing_age_never_negative(self):
        df = pd.DataFrame({"year": [2027, 2020], "listing_month": ["2026-10", "2026-10"]})
        self.assertEqual(list(listing_age(df)), [0.0, 6.0])

    def test_time_modes(self):
        df = toy_listings()
        Xn = build_features(df, time_mode="numeric")
        self.assertIn("month_idx", Xn.columns)
        self.assertFalse(any(c.startswith("period_") for c in Xn.columns))
        self.assertEqual(set(Xn["month_idx"].unique()), {0.0, 1.0, 2.0})
        Xd = build_features(df, time_mode="dummies")
        self.assertNotIn("month_idx", Xd.columns)
        with self.assertRaises(ValueError):
            build_features(df, time_mode="weekly")

    def test_month_order_keeps_columns_stable(self):
        """Alt kümede bir ay hiç yoksa bile aynı month_order ile aynı sütunlar üretilmeli."""
        df = toy_listings()
        order = sorted(df["listing_month"].unique())
        sub = df[df["listing_month"] != "2026-03"]
        X_sub = build_features(sub, month_order=order)
        self.assertIn("period_2026-03", X_sub.columns)
        self.assertEqual(X_sub["period_2026-03"].sum(), 0)

    def test_age_by_model_interactions(self):
        df = toy_listings()
        X = build_features(df, age_by_model=True)
        self.assertEqual(sorted(c for c in X.columns if c.startswith("agex_")), ["agex_B", "agex_C"])
        np.testing.assert_allclose(X["agex_B"], X["age"] * X["model_B"])
        self.assertFalse(any(c.startswith("agex_") for c in build_features(df).columns))

    def test_optional_columns_are_optional(self):
        """Yalnızca zorunlu sütunlarla da çalışmalı (eski şema / kullanıcı verisi)."""
        df = toy_listings(extended=False)
        X = build_features(df)
        self.assertFalse(any(c.startswith(("color_", "trim_", "body_")) for c in X.columns))
        self.assertNotIn("engine_100cc", X.columns)
        fit_ols(X, np.log(df["price"].to_numpy(float)))

    def test_feature_matrix_is_float_and_complete(self):
        X = build_features(toy_listings(), age_by_model=True)
        self.assertFalse(X.isna().any().any())
        self.assertTrue(all(np.issubdtype(t, np.floating) for t in X.dtypes))
        self.assertEqual(len(set(X.columns)), X.shape[1])

    def test_feature_groups(self):
        groups = feature_groups(["age", "age_sq", "agex_B", "km_10k", "fuel_Dizel", "fuel_LPG", "model_B", "seller_Galeri",
                                 "period_2026-02", "month_idx", "color_Gri", "trim_Üst", "engine_100cc",
                                 "body_kaput_boyali", "body_tavan_degisen", "damage"])
        self.assertEqual(groups["yaş"], ["age", "age_sq", "agex_B"])
        self.assertEqual(groups["yakıt"], ["fuel_Dizel", "fuel_LPG"])
        self.assertEqual(groups["ilan dönemi"], ["period_2026-02", "month_idx"])
        self.assertEqual(groups["kaporta (boya/değişen)"], ["body_kaput_boyali", "body_tavan_degisen"])
        for g in ["satıcı tipi", "renk", "donanım", "motor hacmi", "tramer kaydı", "model"]:
            self.assertIn(g, groups)


class TestBodyParts(unittest.TestCase):
    def test_part_region_mapping(self):
        self.assertEqual(len(BODY_PARTS), 13)
        self.assertEqual(set(PART_REGION.values()), set(REGIONS))
        self.assertEqual(sum(PART_REGION[p] == "kapi" for p in BODY_PARTS), 4)
        self.assertEqual(sum(PART_REGION[p] == "tampon" for p in BODY_PARTS), 2)

    def test_region_counts(self):
        row = {p: "Orijinal" for p in BODY_PARTS}
        row.update({"sol_on_kapi": "Boyalı", "sag_arka_kapi": "Boyalı", "sol_arka_kapi": "Değişen", "tavan": "Lokal boyalı"})
        c = body_region_counts(pd.DataFrame([row])).iloc[0]
        self.assertEqual(c["body_kapi_boyali"], 2)
        self.assertEqual(c["body_kapi_degisen"], 1)
        self.assertEqual(c["body_tavan_lokal"], 1)
        self.assertEqual(c["body_kaput_boyali"], 0)
        self.assertEqual(len(c), len(REGIONS) * len(STATE_KEYS))
        self.assertEqual(c.sum(), 4)

    def test_has_body_parts(self):
        df = toy_listings()
        self.assertTrue(has_body_parts(df))
        self.assertFalse(has_body_parts(df.drop(columns=["tavan"])))

    def test_body_effects_and_breakdown(self):
        idx = ["intercept", "body_tavan_degisen", "body_kapi_boyali"]
        table = pd.DataFrame({"coef": [10.0, -0.12, -0.015], "ci_low": [9.9, -0.14, -0.02], "ci_high": [10.1, -0.10, -0.01]},
                             index=idx)
        eff = body_effects(table)
        self.assertEqual(set(eff.variable), {"body_tavan_degisen", "body_kapi_boyali"})
        self.assertAlmostEqual(eff.set_index("variable").loc["body_tavan_degisen", "pct_effect"], np.expm1(-0.12))
        parts = {"tavan": "Değişen", "sol_on_kapi": "Boyalı", "sag_on_kapi": "Boyalı", "kaput": "Orijinal"}
        bd = body_loss_breakdown(parts, eff)
        self.assertEqual(list(bd.part), ["tavan", "sol_on_kapi", "sag_on_kapi", "TOPLAM"])
        self.assertAlmostEqual(bd.iloc[-1]["log_effect"], -0.12 - 2 * 0.015)
        self.assertAlmostEqual(bd.iloc[-1]["pct_effect"], np.expm1(-0.15))
        # bilinmeyen parça / durum
        with self.assertRaises(KeyError):
            body_loss_breakdown({"yan_ayna": "Boyalı"}, eff)
        with self.assertRaises(ValueError):
            body_loss_breakdown({"kaput": "Çizik"}, eff)

    def test_clean_car_has_zero_body_loss(self):
        eff = pd.DataFrame({"variable": ["body_kaput_boyali"], "coef": [-0.025]})
        bd = body_loss_breakdown({p: "Orijinal" for p in BODY_PARTS}, eff)
        self.assertEqual(len(bd), 1)
        self.assertEqual(bd.iloc[0]["pct_effect"], 0.0)


class TestLoad(unittest.TestCase):
    def _write(self, df, d):
        path = Path(d) / "x.csv"
        df.to_csv(path, index=False)
        return path

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_listings(self._write(toy_listings().drop(columns=["km"]), d))

    def test_non_positive_price_raises(self):
        with tempfile.TemporaryDirectory() as d:
            df = toy_listings()
            df.loc[0, "price"] = 0
            with self.assertRaises(ValueError):
                load_listings(self._write(df, d))

    def test_non_binary_damage_raises(self):
        with tempfile.TemporaryDirectory() as d:
            df = toy_listings()
            df.loc[0, "damage"] = 2
            with self.assertRaises(ValueError):
                load_listings(self._write(df, d))

    def test_invalid_body_state_raises(self):
        with tempfile.TemporaryDirectory() as d:
            df = toy_listings()
            df.loc[3, "kaput"] = "Çizik"
            with self.assertRaisesRegex(ValueError, "kaput"):
                load_listings(self._write(df, d))

    def test_rows_with_missing_required_values_are_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            df = toy_listings()
            df.loc[[0, 1], "km"] = np.nan
            self.assertEqual(len(load_listings(self._write(df, d))), len(df) - 2)

    def test_valid_files_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(list(load_listings(self._write(toy_listings(extended=False), d)).columns), REQUIRED)
            self.assertTrue(has_body_parts(load_listings(self._write(toy_listings(), d))))


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

    def test_constant_column_rejected(self):
        X = self.X.assign(c=1.0)
        with self.assertRaisesRegex(ValueError, "Sabit"):
            fit_ols(X, self.y.to_numpy())

    def test_drop_constant_and_predict_ignores_dropped(self):
        X = self.X.assign(c=0.0)
        Xd = drop_constant(X)
        self.assertEqual(list(Xd.columns), ["a", "b"])
        t = fit_ols(Xd, self.y.to_numpy())
        np.testing.assert_allclose(ols_predict(t, X), ols_predict(t, Xd))  # fazladan sütun tahmini değiştirmez

    def test_pct_effect(self):
        self.assertAlmostEqual(pct_effect(np.log(1.10)), 0.10)
        self.assertAlmostEqual(pct_effect(0.0), 0.0)
        self.assertLess(pct_effect(-0.12), 0)

    def test_period_index_recovers_planted_months(self):
        rng = np.random.default_rng(3)
        months = ["2024-01", "2024-02", "2024-03", "2024-04"]
        level = {"2024-01": 0.0, "2024-02": 0.10, "2024-03": 0.25, "2024-04": 0.20}
        df = pd.DataFrame({"listing_month": rng.choice(months, 4000)})
        X = pd.DataFrame({f"period_{m}": (df.listing_month == m).astype(float) for m in months[1:]})
        y = 13 + df.listing_month.map(level).to_numpy() + rng.normal(0, 0.02, len(df))
        idx = period_index(fit_ols(X, y), months)
        self.assertEqual(idx.iloc[0], 1.0)
        np.testing.assert_allclose(idx.to_numpy(), np.exp(list(level.values())), rtol=0.01)

    def test_model_depreciation(self):
        table = pd.DataFrame({"coef": [-0.06, 0.01, -0.02]}, index=["age", "agex_B", "agex_C"])
        dep = model_depreciation(table, ["C", "A", "B"])
        self.assertEqual(list(dep.model), ["B", "A", "C"])  # en az değer kaybeden önce
        self.assertAlmostEqual(dep.set_index("model").loc["A", "annual_loss_pct"], -np.expm1(-0.06))
        self.assertAlmostEqual(dep.set_index("model").loc["C", "age_slope"], -0.08)


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

    def test_r2_of_mean_prediction_is_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(r2(a, np.full(4, a.mean())), 0.0)


class TestImportanceAndPD(unittest.TestCase):
    def test_important_variable_ranks_first(self):
        rng = np.random.default_rng(2)
        n = 1500
        X = pd.DataFrame({"km_10k": rng.normal(size=n), "damage": rng.integers(0, 2, n).astype(float)})
        y = 3.0 * X["km_10k"].to_numpy()  # damage hiç etkili değil
        predict = lambda Z: 3.0 * Z["km_10k"].to_numpy()  # noqa: E731
        imp = grouped_permutation_importance(predict, X, y, n_repeats=5, seed=0)
        self.assertEqual(imp.iloc[0]["variable"], "km")
        self.assertAlmostEqual(float(imp.set_index("variable").loc["tramer kaydı", "rmse_increase"]), 0.0, places=9)

    def test_grouped_columns_are_permuted_together(self):
        """Kaporta sütunları tek bir grup olarak değerlendirilmeli."""
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"body_kaput_boyali": rng.integers(0, 2, 500).astype(float),
                          "body_tavan_degisen": rng.integers(0, 2, 500).astype(float)})
        predict = lambda Z: -0.1 * Z["body_tavan_degisen"].to_numpy()  # noqa: E731
        imp = grouped_permutation_importance(predict, X, predict(X), n_repeats=3)
        self.assertEqual(list(imp.variable), ["kaporta (boya/değişen)"])
        self.assertGreater(imp.iloc[0]["rmse_increase"], 0)

    def test_partial_dependence_is_monotone_for_decreasing_effect(self):
        X = pd.DataFrame({"age": np.zeros(50), "age_sq": np.zeros(50)})
        pd_curve = partial_dependence_age(lambda Z: 14.0 - 0.1 * Z["age"].to_numpy(), X, range(0, 10))
        self.assertTrue(pd_curve.is_monotonic_decreasing)
        self.assertAlmostEqual(pd_curve.iloc[1] / pd_curve.iloc[0], np.exp(-0.1))

    def test_partial_dependence_updates_model_age_interactions(self):
        X = pd.DataFrame({"age": [3.0, 3.0], "age_sq": [9.0, 9.0], "model_B": [0.0, 1.0], "agex_B": [0.0, 3.0]})
        seen = []

        def predict(Z):
            seen.append(Z["agex_B"].to_numpy().copy())
            return np.zeros(len(Z))

        partial_dependence_age(predict, X, [5])
        np.testing.assert_array_equal(seen[0], [0.0, 5.0])


if __name__ == "__main__":
    unittest.main()
