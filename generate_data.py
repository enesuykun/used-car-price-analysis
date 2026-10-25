"""Sentetik ikinci el otomobil ilan verisi üretici.

Veri tamamen sentetiktir; gerçek ilanlara, fiyatlara veya piyasa verisine dayanmaz.
Model adları yalnızca gerçekçi etiket olarak kullanılmıştır, fiyatlar kurgusaldır.

Fiyat, bilinen "gerçek" etkilerle üretilir (TRUE_EFFECTS). Bu sayede analizde regresyonun
bu etkileri geri bulup bulamadığı kontrol edilebilir (bkz. run_analysis.py).

Çıktı: data/listings.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 21
N = 8000
CURRENT_YEAR = 2026
OUT = Path(__file__).parent / "data"

# model: (baz fiyat TL (sıfır araç, kurgusal), yıllık değer kaybı oranı, segment, hibrit seçeneği var mı)
MODELS = {
    "Renault Clio":      (1_150_000, 0.075, "ekonomi", False),
    "Renault Megane":    (1_600_000, 0.080, "orta", False),
    "Fiat Egea":         (1_000_000, 0.070, "ekonomi", False),
    "Toyota Corolla":    (1_700_000, 0.065, "orta", True),
    "Toyota Yaris":      (1_300_000, 0.065, "ekonomi", True),
    "Volkswagen Golf":   (1_900_000, 0.085, "orta", False),
    "Volkswagen Passat": (2_600_000, 0.095, "premium", False),
    "Hyundai i20":       (1_200_000, 0.075, "ekonomi", True),
    "Hyundai Tucson":    (2_400_000, 0.080, "orta", True),
    "Ford Focus":        (1_500_000, 0.085, "orta", False),
    "Honda Civic":       (1_900_000, 0.070, "orta", True),
    "Dacia Duster":      (1_500_000, 0.070, "orta", False),
    "BMW 3 Serisi":      (3_200_000, 0.100, "premium", False),
    "Mercedes C Serisi": (3_600_000, 0.105, "premium", False),
    "Skoda Octavia":     (1_900_000, 0.080, "orta", False),
}

# log fiyata eklenen sabit etkiler (referans: Benzin, Manuel, hasar yok, Sahibinden, Diğer şehir)
TRUE_EFFECTS = {
    "fuel_Dizel": 0.030,
    "fuel_Hibrit": 0.080,
    "fuel_LPG": -0.040,
    "gearbox_Otomatik": 0.070,
    "damage": -0.120,
    "seller_Galeri": 0.040,
    "city_İstanbul": 0.020,
    "city_Ankara": 0.010,
    "city_İzmir": 0.010,
    "km_per_10k": -0.004,
    "per_month": 0.012,
}
AGE_QUAD = 0.0025  # yaşın doğrusal olmayan (kuadratik) etkisi: değer kaybı yaşlandıkça yavaşlar
NOISE_SD = 0.09
CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Diğer"]
CITY_P = [0.28, 0.14, 0.12, 0.08, 0.07, 0.06, 0.05, 0.20]
MONTHS = pd.period_range("2025-11", "2026-10", freq="M").astype(str).tolist()


def main() -> None:
    rng = np.random.default_rng(SEED)
    names = list(MODELS)
    weights = np.array([1.3, 0.8, 1.6, 1.2, 0.8, 1.1, 0.7, 1.0, 0.7, 0.9, 0.8, 0.9, 0.6, 0.6, 0.9])  # ilan payı (göreli)
    model = rng.choice(names, N, p=weights / weights.sum())
    age = np.clip(rng.gamma(2.4, 2.4, N).round(), 0, 15).astype(int)
    year = CURRENT_YEAR - age
    km_per_year = np.minimum(rng.lognormal(np.log(14_000), 0.35, N), 35_000)  # yıllık km, gerçekçi üst sınır
    km = np.maximum(500, age * km_per_year + rng.normal(0, 1500, N)).round(-2).astype(int)

    fuel = np.empty(N, dtype=object)
    for i, m in enumerate(model):
        hybrid_ok = MODELS[m][3]
        probs = {"Benzin": 0.50, "Dizel": 0.32, "LPG": 0.06, "Hibrit": 0.12 if hybrid_ok else 0.0}
        p = np.array(list(probs.values()))
        fuel[i] = rng.choice(list(probs), p=p / p.sum())
    gearbox = np.where(rng.random(N) < np.where(np.isin(model, [n for n in names if MODELS[n][2] == "premium"]), 0.85, 0.45), "Otomatik", "Manuel")
    damage = (rng.random(N) < 0.22).astype(int)
    seller = np.where(rng.random(N) < 0.42, "Galeri", "Sahibinden")
    city = rng.choice(CITIES, N, p=CITY_P)
    month_idx = rng.integers(0, len(MONTHS), N)

    base = np.array([MODELS[m][0] for m in model], dtype=float)
    dep = np.array([MODELS[m][1] for m in model])
    log_price = (
        np.log(base)
        - dep * age
        + AGE_QUAD * age**2
        + TRUE_EFFECTS["km_per_10k"] * km / 10_000
        + TRUE_EFFECTS["per_month"] * month_idx
        + np.vectorize(lambda f: TRUE_EFFECTS.get(f"fuel_{f}", 0.0))(fuel)
        + np.where(gearbox == "Otomatik", TRUE_EFFECTS["gearbox_Otomatik"], 0.0)
        + damage * TRUE_EFFECTS["damage"]
        + np.where(seller == "Galeri", TRUE_EFFECTS["seller_Galeri"], 0.0)
        + np.vectorize(lambda c: TRUE_EFFECTS.get(f"city_{c}", 0.0))(city)
        + rng.normal(0, NOISE_SD, N)
    )
    price = np.maximum(150_000, np.exp(log_price)).round(-3).astype(int)

    df = pd.DataFrame(
        {
            "listing_id": np.arange(1, N + 1),
            "model": model,
            "year": year,
            "km": km,
            "fuel": fuel,
            "gearbox": gearbox,
            "damage": damage,
            "seller_type": seller,
            "city": city,
            "listing_month": [MONTHS[i] for i in month_idx],
            "price": price,
        }
    )
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "listings.csv", index=False)
    print(f"{len(df):,} ilan yazıldı -> data/listings.csv  (medyan fiyat: {int(df.price.median()):,} TL)")


if __name__ == "__main__":
    main()
