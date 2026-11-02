"""generate_data.py testleri: üretilen verinin tutarlılığı, kalibrasyonu ve değer kaybı mantığı."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_data as g  # noqa: E402
from analysis import BODY_PARTS, BODY_STATES, REQUIRED, listing_age, load_listings  # noqa: E402


class GeneratedDataBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = g.generate(n=6000, seed=7)
        cls.age = listing_age(cls.df)


class TestSchemaAndRanges(GeneratedDataBase):
    def test_columns(self):
        for c in [*REQUIRED, "color", "engine_cc", "trim", "trim_level", "segment", *BODY_PARTS, "n_painted", "n_replaced",
                  "body_loss_pct_true"]:
            self.assertIn(c, self.df.columns)
        self.assertFalse(self.df[REQUIRED].isna().any().any())
        self.assertTrue(self.df["listing_id"].is_unique)

    def test_value_ranges(self):
        df = self.df
        self.assertTrue((df.price > 0).all())
        self.assertTrue(df.damage.isin([0, 1]).all())
        self.assertTrue((df.km > 0).all())
        self.assertTrue(df.listing_month.between("2016-01", "2026-10").all())
        self.assertTrue((self.age >= 0).all() and (self.age <= 20).all())
        self.assertTrue(df.fuel.isin(["Benzin", "Dizel", "LPG", "Hibrit"]).all())
        self.assertTrue(df.gearbox.isin(["Manuel", "Otomatik"]).all())
        self.assertTrue(df.color.isin(g.COLORS).all())
        self.assertTrue(df.trim_level.isin(["Temel", "Orta", "Üst"]).all())
        self.assertTrue(df.city.isin(g.CITIES).all())
        self.assertTrue(df[BODY_PARTS].isin(BODY_STATES).all().all())

    def test_model_year_not_before_launch_and_not_after_listing(self):
        launch = self.df.model.map({m: s["launch"] for m, s in g.MODELS.items()})
        self.assertTrue((self.df.year >= launch).all())
        self.assertTrue((self.df.year <= self.df.listing_month.str[:4].astype(int)).all())

    def test_written_file_passes_loader(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "listings.csv"
            self.df.to_csv(path, index=False)
            self.assertEqual(len(load_listings(path)), len(self.df))

    def test_deterministic(self):
        a = g.generate(n=500, seed=3)
        b = g.generate(n=500, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.price.equals(g.generate(n=500, seed=4).price))


class TestConsistency(GeneratedDataBase):
    def test_engine_matches_model_catalogue(self):
        for (model, cc, fuel), _ in self.df.groupby(["model", "engine_cc", "fuel"]):
            catalogue = {(c, f) for c, f, _ in g.MODELS[model]["engines"]}
            base_fuel = "Benzin" if fuel == "LPG" else fuel  # LPG sonradan dönüşüm
            self.assertIn((cc, base_fuel), catalogue, f"{model} {cc} {fuel}")

    def test_hybrid_only_in_hybrid_models_and_years(self):
        hyb = self.df[self.df.fuel == "Hibrit"]
        self.assertGreater(len(hyb), 0)
        self.assertTrue(hyb.model.isin(g.HYBRID_MODELS).all())
        first = {m: min(y for _, f, y in g.MODELS[m]["engines"] if f == "Hibrit") for m in g.HYBRID_MODELS}
        self.assertTrue((hyb.year >= hyb.model.map(first)).all())
        self.assertTrue((hyb.gearbox == "Otomatik").all())

    def test_no_lpg_in_premium(self):
        self.assertFalse(((self.df.segment == "premium") & (self.df.fuel == "LPG")).any())

    def test_trim_name_matches_level(self):
        for (model, level), sub in self.df.groupby(["model", "trim_level"]):
            i = ["Temel", "Orta", "Üst"].index(level)
            self.assertTrue((sub.trim == g.MODELS[model]["trims"][i]).all())

    def test_painted_replaced_counts(self):
        st = self.df[BODY_PARTS]
        np.testing.assert_array_equal(self.df.n_painted, st.isin(["Boyalı", "Lokal boyalı"]).sum(axis=1))
        np.testing.assert_array_equal(self.df.n_replaced, (st == "Değişen").sum(axis=1))

    def test_body_loss_true_matches_penalty_table(self):
        expected = np.round(-np.expm1(-g.body_penalty_log(self.df[BODY_PARTS])) * 100, 2)
        np.testing.assert_allclose(self.df.body_loss_pct_true, expected)
        clean = (self.df.n_painted + self.df.n_replaced) == 0
        self.assertTrue((self.df.loc[clean, "body_loss_pct_true"] == 0).all())


class TestRealismTargets(GeneratedDataBase):
    """Kalibrasyon hedefleri: kaynaklardaki eğilimler (README, Kalibrasyon bölümü) korunmalı."""

    def test_older_cars_have_more_bodywork_and_km(self):
        old, young = self.age >= 10, self.age <= 3
        work = self.df.n_painted + self.df.n_replaced
        self.assertGreater(work[old].mean(), 2 * work[young].mean())
        self.assertGreater(self.df.km[old].median(), 2 * self.df.km[young].median())

    def test_clean_share_is_plausible(self):
        clean = ((self.df.n_painted + self.df.n_replaced) == 0).mean()
        self.assertTrue(0.2 < clean < 0.5, clean)

    def test_damage_record_more_likely_with_serious_bodywork(self):
        serious = self.df[[p for p in BODY_PARTS if "tampon" not in p]].isin(["Boyalı", "Değişen"]).any(axis=1)
        self.assertGreater(self.df.damage[serious].mean(), self.df.damage[~serious].mean() + 0.3)

    def test_roof_is_rarest_bumper_most_common(self):
        touched = (self.df[BODY_PARTS] != "Orijinal").mean()
        self.assertEqual(touched.idxmin(), "tavan")
        self.assertIn(touched.idxmax(), ["on_tampon", "arka_tampon"])

    def test_white_to_grey_shift(self):
        old = self.df[self.df.year <= 2012].color.value_counts(normalize=True)
        new = self.df[self.df.year >= 2022].color.value_counts(normalize=True)
        self.assertEqual(old.idxmax(), "Beyaz")
        self.assertLess(new["Beyaz"], old["Beyaz"] - 0.1)
        self.assertGreater(new["Gri"], old["Gri"] + 0.1)

    def test_diesel_share_falls_hybrid_share_rises(self):
        y = self.df.listing_month.str[:4].astype(int)
        early, late = self.df[y <= 2018], self.df[y >= 2025]
        self.assertGreater((early.fuel == "Dizel").mean(), (late.fuel == "Dizel").mean() + 0.05)
        self.assertGreater((late.fuel == "Hibrit").mean(), (early.fuel == "Hibrit").mean())

    def test_diesel_cars_drive_more(self):
        km_per_year = self.df.km / (self.age + 0.5)
        self.assertGreater(km_per_year[self.df.fuel == "Dizel"].median(), 1.15 * km_per_year[self.df.fuel == "Benzin"].median())

    def test_most_engines_below_1600cc(self):
        self.assertGreater((self.df.engine_cc <= 1600).mean(), 0.85)

    def test_premium_mostly_automatic(self):
        prem = self.df[self.df.segment == "premium"]
        self.assertGreater((prem.gearbox == "Otomatik").mean(), 0.8)
        self.assertLess((self.df[self.df.segment == "ekonomi"].gearbox == "Otomatik").mean(), 0.6)


class TestMarketIndexAndCalibration(unittest.TestCase):
    def test_index_hits_anchors(self):
        idx = g.market_index()
        self.assertEqual(idx.iloc[0], 1.0)
        self.assertEqual(len(idx), len(g.MONTHS))
        base = g.PRICE_ANCHORS["2016-01"]
        for m, v in g.PRICE_ANCHORS.items():
            self.assertAlmostEqual(idx[m] * base, v, delta=1)

    def test_index_known_episodes(self):
        idx = g.market_index()
        self.assertGreater(idx["2022-06"] / idx["2021-06"], 2.0)   # 2021-22 fiyat patlaması
        self.assertLess(idx["2022-11"], idx["2022-06"])            # Haziran 2022 düzenlemesi sonrası gevşeme
        self.assertGreater(idx["2026-10"], idx["2025-10"])
        self.assertTrue(10 < idx.iloc[-1] < 30)                     # 10 yılda ~20 kat nominal

    def test_generated_mean_prices_track_sourced_anchors(self):
        """2023 sonrası kaynaklı çapalarda üretilen ortalama fiyat ±%10 içinde olmalı."""
        df = g.generate()
        for m in ["2023-11", "2024-06", "2025-11", "2026-06"]:
            p = pd.Period(m, "M")
            window = df.listing_month.isin([str(p - 1), m, str(p + 1)])
            ratio = df.loc[window, "price"].mean() / g.PRICE_ANCHORS[m]
            self.assertTrue(0.9 < ratio < 1.1, f"{m}: {ratio:.3f}")

    def test_calibration_constant_is_current(self):
        self.assertAlmostEqual(g._calibrate(), g.PRICE_SCALE, delta=0.01)


class TestDepreciationLogic(unittest.TestCase):
    def test_body_penalty_ordering(self):
        pen = g.BODY_PENALTY
        for region, p in pen.items():
            self.assertGreater(p["Değişen"], p["Boyalı"], region)
        heaviest = max(pen, key=lambda r: pen[r]["Değişen"])
        lightest = min(pen, key=lambda r: pen[r]["Değişen"])
        self.assertEqual(heaviest, "tavan")
        self.assertEqual(lightest, "tampon")
        self.assertTrue(0 < g.LOCAL_SHARE < 1)

    def test_body_penalty_log_single_parts(self):
        def states(**kw):
            row = {p: "Orijinal" for p in BODY_PARTS}
            row.update(kw)
            return pd.DataFrame([row])

        self.assertEqual(g.body_penalty_log(states())[0], 0.0)
        self.assertAlmostEqual(g.body_penalty_log(states(tavan="Değişen"))[0], g.BODY_PENALTY["tavan"]["Değişen"])
        self.assertAlmostEqual(g.body_penalty_log(states(kaput="Lokal boyalı"))[0], g.LOCAL_SHARE * g.BODY_PENALTY["kaput"]["Boyalı"])
        two_doors = g.body_penalty_log(states(sol_on_kapi="Boyalı", sag_on_kapi="Boyalı"))[0]
        self.assertAlmostEqual(two_doors, 2 * g.BODY_PENALTY["kapi"]["Boyalı"])

    def test_bodywork_lowers_price_all_else_equal(self):
        """Aynı seed ile üretilen veride, kaporta cezası yüksek ilanlar (yaş ve dönem içinde) daha ucuz olmalı."""
        df = g.generate(n=8000, seed=11)
        df["age"] = listing_age(df)
        df["rel"] = np.log(df.price) - np.log(df.groupby(["listing_month", "model"]).price.transform("median"))
        heavy = df.body_loss_pct_true > 5
        clean = df.body_loss_pct_true == 0
        same_age = df.age.between(4, 10)
        self.assertLess(df.rel[heavy & same_age].mean(), df.rel[clean & same_age].mean() - 0.03)

    def test_effect_signs(self):
        e = g.TRUE_EFFECTS
        self.assertLess(e["damage"], 0)
        self.assertLess(e["km_10k"], 0)
        self.assertLess(e["fuel_LPG"], 0)
        self.assertGreater(e["engine_100cc"], 0)
        self.assertGreater(e["trim_Üst"], e["trim_Orta"])
        self.assertGreater(e["trim_Orta"], 0)
        self.assertGreater(e["gearbox_Otomatik"], 0)


if __name__ == "__main__":
    unittest.main()
