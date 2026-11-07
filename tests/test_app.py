"""Web arayüzü testleri: dışa aktarılan model güncel mi, JavaScript hesabı Python ile aynı mı, davranış mantıklı mı."""
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import analysis as A  # noqa: E402
import build_app  # noqa: E402
import export_model as X  # noqa: E402

MODEL = json.loads((APP / "model.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((APP / "golden_cases.json").read_text(encoding="utf-8"))


def base_car(**kw):
    car = dict(model="Toyota C-HR", year=2024, km=35_000, fuel="Hibrit", engine_cc=1798, gearbox="Otomatik", trim_level="Orta",
               color="Gri", city="Diğer", seller_type="Sahibinden", damage=0, parts={p: "Orijinal" for p in A.BODY_PARTS})
    car.update(kw)
    return car


def price(car):
    return float(np.exp(X.predict_log(MODEL, car)))


class TestExportedModel(unittest.TestCase):
    def test_model_json_matches_a_fresh_fit(self):
        """Depodaki model.json, data/listings.csv'den şimdi yeniden kestirilen katsayılarla aynı olmalı (dışa aktarmayı unutma)."""
        df = A.load_listings(ROOT / "data" / "listings.csv")
        months, table, _ = X.fit(df)
        for name, value in table["coef"].items():
            self.assertAlmostEqual(MODEL["coef"][name], value, places=9, msg=name)
        self.assertEqual(len(MODEL["coef"]), len(table))

    def test_python_formula_matches_ols_predictions(self):
        for case in GOLDEN:
            self.assertAlmostEqual(X.predict_log(MODEL, case["car"]), case["expected_log"], places=9)

    def test_catalog_covers_all_generated_models(self):
        import generate_data as G

        self.assertEqual(set(MODEL["catalog"]), set(G.MODELS))
        for name, spec in MODEL["catalog"].items():
            self.assertEqual(len(spec["trims"]), 3, name)

    def test_residual_band_is_sensible(self):
        meta = MODEL["meta"]
        self.assertLess(meta["resid_p10"], 0)
        self.assertGreater(meta["resid_p90"], 0)
        self.assertTrue(0.05 < meta["resid_sd"] < 0.15)


class TestEstimatorBehaviour(unittest.TestCase):
    """Arayüzün gösterdiği fiyatın yönleri doğru olmalı (Python karşılığı; JS ile aynı formül)."""

    def test_more_km_is_cheaper(self):
        self.assertGreater(price(base_car(km=20_000)), price(base_car(km=120_000)))

    def test_older_is_cheaper(self):
        self.assertGreater(price(base_car(year=2024)), price(base_car(year=2018)))

    def test_damage_record_lowers_price(self):
        self.assertGreater(price(base_car()), price(base_car(damage=1)))

    def test_body_state_ordering_on_roof(self):
        def with_roof(state):
            car = base_car()
            car["parts"]["tavan"] = state
            return price(car)

        clean, local, painted, replaced = (with_roof(s) for s in A.BODY_STATES)
        self.assertGreater(clean, local)
        self.assertGreater(local, painted)
        self.assertGreater(painted, replaced)

    def test_roof_costs_more_than_bumper(self):
        roof, bumper = base_car(), base_car()
        roof["parts"]["tavan"] = "Değişen"
        bumper["parts"]["on_tampon"] = "Değişen"
        self.assertLess(price(roof), price(bumper))

    def test_effects_add_up_in_log_space(self):
        """İki parça birlikte, tek tek etkilerin log toplamıdır (arayüzdeki 'toplam' satırı buna dayanır)."""
        both, a, b, clean = base_car(), base_car(), base_car(), base_car()
        both["parts"].update(tavan="Boyalı", kaput="Değişen")
        a["parts"]["tavan"] = "Boyalı"
        b["parts"]["kaput"] = "Değişen"
        la, lb, l0, lboth = (X.predict_log(MODEL, c) for c in (a, b, clean, both))
        self.assertAlmostEqual(lboth - l0, (la - l0) + (lb - l0), places=12)

    def test_reference_model_has_no_dummy(self):
        ref = MODEL["meta"]["reference_model"]
        self.assertNotIn(ref, MODEL["meta"]["models_with_dummy"])
        self.assertEqual(ref, sorted(MODEL["catalog"])[0])
        self.assertGreater(price(base_car(model=ref, fuel="Benzin", engine_cc=1998, trim_level="Orta")), 0)

    def test_example_car_is_in_a_plausible_range(self):
        """2024 C-HR 1.8 hibrit, hatasız, 35 bin km: gerçek ilanlar ~2,07-2,25 M TL civarında; model bu bandın ±%20'sinde."""
        self.assertTrue(1_700_000 < price(base_car()) < 2_700_000, price(base_car()))


@unittest.skipUnless(shutil.which("node"), "node yok")
class TestJavaScriptParity(unittest.TestCase):
    def test_js_formula_equals_python(self):
        res = subprocess.run(["node", str(APP / "check_golden.js")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("OK", res.stdout)


class TestBuiltPage(unittest.TestCase):
    def test_index_html_is_up_to_date(self):
        """index.html, şablon + estimator.js + model.json'dan şimdi üretilecek içeriği taşımalı (build_app.py çalıştırıldı mı)."""
        html = (APP / "index.html").read_text(encoding="utf-8")
        body = build_app.assemble().split("<!--BODY-->", 1)[1].strip()
        self.assertIn(body, html)

    def test_pages_copy_matches_app_page(self):
        """GitHub Pages /docs/index.html, app/index.html ile birebir aynı olmalı (build_app.py ikisini birlikte yazar)."""
        self.assertEqual((ROOT / "docs" / "index.html").read_text(encoding="utf-8"), (APP / "index.html").read_text(encoding="utf-8"))
        self.assertTrue((ROOT / "docs" / ".nojekyll").exists())

    def test_page_is_self_contained(self):
        html = (APP / "index.html").read_text(encoding="utf-8")
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("window.Estimator", html + "root.Estimator")
        self.assertNotIn("/*__MODEL__*/", html)
        self.assertNotIn("/*__ESTIMATOR__*/", html)
        for host in ("cdn.", "unpkg", "jsdelivr"):  # üçüncü taraf betik yok; yalnızca Google Fonts stil dosyası
            self.assertNotIn(host, html)

    def test_page_discloses_synthetic_data(self):
        html = (APP / "index.html").read_text(encoding="utf-8")
        self.assertIn("sentetik", html)
        self.assertIn("synthetic", html)


if __name__ == "__main__":
    unittest.main()
