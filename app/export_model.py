"""OLS modelini web arayüzünün okuyacağı JSON'a aktarır.

    python app/export_model.py        # app/model.json ve app/golden_cases.json yazar

Arayüz sunucu gerektirmez: katsayılar tarayıcıda çarpılıp toplanır (log fiyat = sabit + Σ katsayı x özellik).
golden_cases.json, JavaScript hesabının Python ile aynı sonucu verdiğini test etmek içindir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as A  # noqa: E402
import generate_data as G  # noqa: E402

OUT = Path(__file__).parent
REF_MONTHS = 3  # güncel fiyat düzeyi = son 3 ayın ay katsayılarının ortalaması (tek ayın gürültüsünü azaltır)


def fit(df: pd.DataFrame):
    months = sorted(df["listing_month"].unique())
    X = A.build_features(df, month_order=months, time_mode="dummies", age_by_model=True)
    y = np.log(df["price"].to_numpy(float))
    table = A.fit_ols(A.drop_constant(X), y)
    resid = y - A.ols_predict(table, X[[c for c in table.index if c != "intercept"]])
    return months, table, resid


def period_level(table: pd.DataFrame, months: list[str]) -> float:
    last = months[-REF_MONTHS:]
    return float(np.mean([table.loc[f"period_{m}", "coef"] for m in last]))


def predict_log(model: dict, car: dict) -> float:
    """Arayüzdeki (JavaScript) hesabın Python karşılığı. car: arayüzün topladığı girdiler."""
    c = model["coef"]
    age = max(0, model["meta"]["ref_year"] - car["year"])
    z = c["intercept"] + model["meta"]["period_level"]
    z += c["age"] * age + c["age_sq"] * age**2 + c["km_10k"] * car["km"] / 10_000
    z += c["damage"] * car["damage"] + c["engine_100cc"] * car["engine_cc"] / 100
    for prefix, value in [("fuel", car["fuel"]), ("gearbox", car["gearbox"]), ("seller", car["seller_type"]),
                          ("city", car["city"]), ("color", car["color"]), ("trim", car["trim_level"])]:
        z += c.get(f"{prefix}_{value}", 0.0)
    if car["model"] in model["meta"]["models_with_dummy"]:
        z += c[f"model_{car['model']}"] + c[f"agex_{car['model']}"] * age
    for part, state in car["parts"].items():
        if state != "Orijinal":
            region = model["part_region"][part]
            z += c.get(f"body_{region}_{model['state_keys'][state]}", 0.0)
    return z


def main() -> None:
    df = A.load_listings(ROOT / "data" / "listings.csv")
    months, table, resid = fit(df)
    plevel = period_level(table, months)
    age = A.listing_age(df)
    kmy = (df["km"] / (age + 0.5))[age >= 1]
    cmp = pd.read_csv(ROOT / "results" / "model_comparison.csv", index_col=0)
    ols_row = cmp.loc["OLS (log-lineer)"]

    coef = {k: float(v) for k, v in table["coef"].items()}
    se = {k: float(v) for k, v in table["se"].items()}
    model = {
        "meta": {
            "n_listings": int(len(df)),
            "last_month": months[-1],
            "ref_year": int(months[-1][:4]),
            "period_level": plevel,
            "ref_months": months[-REF_MONTHS:],
            "resid_sd": float(np.std(resid)),
            "resid_p10": float(np.quantile(resid, 0.10)),
            "resid_p90": float(np.quantile(resid, 0.90)),
            "test_mape": float(ols_row["MAPE"]),
            "km_per_year_p05": float(kmy.quantile(0.05)),
            "km_per_year_p95": float(kmy.quantile(0.95)),
            "models_with_dummy": sorted(m for m in df["model"].unique() if f"model_{m}" in coef),
            "reference_model": sorted(df["model"].unique())[0],
            "synthetic": True,
        },
        "coef": coef,
        "se": se,
        "part_region": A.PART_REGION,
        "state_keys": A.STATE_KEYS,
        "region_labels": A.REGION_LABELS,
        "catalog": {
            m: {"segment": s["seg"], "launch": s["launch"], "engines": [list(e) for e in s["engines"]], "trims": s["trims"]}
            for m, s in G.MODELS.items()
        },
        "colors": G.COLORS,
        "cities": G.CITIES,
        "year_min": int(df["year"].min()),
        "max_age": 20,
    }
    (OUT / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- altın test örnekleri: Python tahmini (ols_predict) ile arayüz formülü aynı olmalı
    cases = []
    # Özellikler tam veri üzerinde hesaplanır (tek satırda kategorik düzeyler kaybolur); tüm ilanlar son aya taşınır.
    ref = df.assign(listing_month=months[-1])
    Xref = A.build_features(ref, month_order=months, time_mode="dummies", age_by_model=True)
    Xref = Xref[[c for c in table.index if c != "intercept"]]
    pred_ref = A.ols_predict(table, Xref)
    for i in df.sample(12, random_state=3).index:
        r = df.loc[i]
        car = dict(model=r["model"], year=int(r["year"]), km=int(r["km"]), fuel=r["fuel"], engine_cc=int(r["engine_cc"]),
                   gearbox=r["gearbox"], trim_level=r["trim_level"], color=r["color"], city=r["city"],
                   seller_type=r["seller_type"], damage=int(r["damage"]), parts={p: r[p] for p in A.BODY_PARTS})
        expected = float(pred_ref[i]) - coef[f"period_{months[-1]}"] + plevel
        cases.append({"car": car, "expected_log": expected, "expected_price": float(np.exp(expected))})
    (OUT / "golden_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    worst = max(abs(predict_log(model, c["car"]) - c["expected_log"]) for c in cases)
    print(f"model.json ({len(coef)} katsayı), golden_cases.json ({len(cases)} örnek). Python formül farkı: {worst:.2e}")


if __name__ == "__main__":
    main()
