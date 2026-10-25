"""İkinci el fiyat analizi için yardımcı fonksiyonlar (özellik üretimi, OLS, metrikler, önem analizi).

Beklenen veri şeması (gerçek bir CSV kullanılacaksa sütun adları bunlar olmalı):
    model, year, km, fuel, gearbox, damage (0/1), seller_type, city, listing_month (YYYY-MM), price
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["model", "year", "km", "fuel", "gearbox", "damage", "seller_type", "city", "listing_month", "price"]
# kategorik sütun -> (özellik öneki, referans düzey). Referans düzey modelden çıkarılır, etkiler ona göre okunur.
CATEGORICAL = {
    "fuel": ("fuel", "Benzin"),
    "gearbox": ("gearbox", "Manuel"),
    "seller_type": ("seller", "Sahibinden"),
    "city": ("city", "Diğer"),
}


def load_listings(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütun(lar): {missing}. Beklenen şema: {REQUIRED}")
    df = df.dropna(subset=REQUIRED).copy()
    if (df["price"] <= 0).any():
        raise ValueError("price sütunu pozitif olmalı (log dönüşümü için).")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, ref_year: int | None = None, month_order: list | None = None) -> pd.DataFrame:
    """Sayısal + one-hot özellik matrisi. Aynı ref_year ve month_order ile tekrar üretilebilir."""
    ref_year = int(df["year"].max()) if ref_year is None else ref_year
    months = month_order or sorted(df["listing_month"].unique())
    X = pd.DataFrame(index=df.index)
    X["age"] = (ref_year - df["year"]).astype(float)
    X["age_sq"] = X["age"] ** 2
    X["km_10k"] = df["km"] / 10_000
    X["month_idx"] = df["listing_month"].map({m: i for i, m in enumerate(months)}).astype(float)
    X["damage"] = df["damage"].astype(float)
    for col, (prefix, ref) in CATEGORICAL.items():
        levels = sorted(df[col].unique())
        ref = ref if ref in levels else levels[0]  # referans düzey veride yoksa alfabetik ilk düzey (çoklu doğrusal bağıntıyı önler)
        for level in levels:
            if level != ref:
                X[f"{prefix}_{level}"] = (df[col] == level).astype(float)
    model_levels = sorted(df["model"].unique())
    for level in model_levels[1:]:  # ilk model (alfabetik) referans
        X[f"model_{level}"] = (df["model"] == level).astype(float)
    return X


# ----------------------------------------------------------------------------- OLS
def fit_ols(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Sabit terimli klasik OLS; katsayı, standart hata ve %95 güven aralığı döndürür."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    out = pd.DataFrame({"coef": beta, "se": se}, index=["intercept", *X.columns])
    out["ci_low"] = out.coef - 1.96 * out.se
    out["ci_high"] = out.coef + 1.96 * out.se
    return out


def ols_predict(table: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    return table.loc["intercept", "coef"] + X.to_numpy(float) @ table.loc[X.columns, "coef"].to_numpy()


def pct_effect(coef: float) -> float:
    """Log-fiyat katsayısını yüzde etkiye çevirir: exp(b) - 1."""
    return float(np.expm1(coef))


# ----------------------------------------------------------------------------- metrikler (TL ölçeğinde)
def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(actual - pred) / actual))


def median_ape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.median(np.abs(actual - pred) / actual))


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))))


def r2(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())


# ----------------------------------------------------------------------------- önem analizi
def feature_groups(columns) -> dict:
    """One-hot sütunlarını orijinal değişkene bağlar (ör. fuel_Dizel, fuel_LPG -> 'fuel')."""
    groups: dict = {}
    for c in columns:
        if c in ("age", "age_sq"):
            key = "age (yaş)"
        elif c == "km_10k":
            key = "km"
        elif c == "month_idx":
            key = "ilan ayı"
        elif c == "damage":
            key = "hasar kaydı"
        else:
            key = c.split("_", 1)[0]
            key = {"seller": "satıcı tipi", "fuel": "yakıt", "gearbox": "vites", "city": "şehir", "model": "model"}.get(key, key)
        groups.setdefault(key, []).append(c)
    return groups


def grouped_permutation_importance(predict_log, X: pd.DataFrame, y_log: np.ndarray, n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Bir değişkene ait tüm sütunları aynı permütasyonla karıştırıp log-RMSE artışını ölçer."""
    rng = np.random.default_rng(seed)
    base = np.sqrt(np.mean((y_log - predict_log(X)) ** 2))
    rows = []
    for name, cols in feature_groups(X.columns).items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            deltas.append(np.sqrt(np.mean((y_log - predict_log(Xp)) ** 2)) - base)
        rows.append((name, float(np.mean(deltas)), float(np.std(deltas))))
    out = pd.DataFrame(rows, columns=["variable", "rmse_increase", "std"]).sort_values("rmse_increase", ascending=False)
    return out.reset_index(drop=True)


def partial_dependence_age(predict_log, X: pd.DataFrame, ages) -> pd.Series:
    """Yaşı tüm gözlemlerde sabitleyip ortalama tahmin edilen log fiyatı hesaplar (geometrik ortalama fiyat)."""
    out = {}
    for a in ages:
        Xa = X.copy()
        Xa["age"] = float(a)
        Xa["age_sq"] = float(a) ** 2
        out[a] = float(np.exp(np.mean(predict_log(Xa))))
    return pd.Series(out)
