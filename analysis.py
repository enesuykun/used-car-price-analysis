"""İkinci el fiyat analizi için yardımcı fonksiyonlar (özellik üretimi, OLS, metrikler, önem analizi, kaporta etkisi).

Beklenen veri şeması
    Zorunlu (REQUIRED): model, year, km, fuel, gearbox, damage (0/1), seller_type, city, listing_month (YYYY-MM), price
    İsteğe bağlı (varsa modele girer):
        color, engine_cc, trim_level (Temel/Orta/Üst),
        kaporta parçaları (BODY_PARTS): her biri "Orijinal" / "Lokal boyalı" / "Boyalı" / "Değişen"
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
OPTIONAL_CATEGORICAL = {
    "color": ("color", "Beyaz"),
    "trim_level": ("trim", "Temel"),
}

# ----------------------------------------------------------------------------- kaporta
# Ekspertiz raporlarındaki 13 parça ve analizde birleştirildikleri bölge.
PART_REGION = {
    "kaput": "kaput",
    "tavan": "tavan",
    "bagaj": "bagaj",
    "sol_on_camurluk": "on_camurluk",
    "sag_on_camurluk": "on_camurluk",
    "sol_on_kapi": "kapi",
    "sag_on_kapi": "kapi",
    "sol_arka_kapi": "kapi",
    "sag_arka_kapi": "kapi",
    "sol_arka_camurluk": "arka_camurluk",
    "sag_arka_camurluk": "arka_camurluk",
    "on_tampon": "tampon",
    "arka_tampon": "tampon",
}
BODY_PARTS = list(PART_REGION)
REGIONS = list(dict.fromkeys(PART_REGION.values()))
REGION_LABELS = {
    "kaput": "Kaput", "tavan": "Tavan", "bagaj": "Bagaj kapağı", "on_camurluk": "Ön çamurluk",
    "kapi": "Kapı", "arka_camurluk": "Arka çamurluk", "tampon": "Tampon",
}
# veri değeri -> özellik adı eki
STATE_KEYS = {"Lokal boyalı": "lokal", "Boyalı": "boyali", "Değişen": "degisen"}
BODY_STATES = ["Orijinal", *STATE_KEYS]


def has_body_parts(df: pd.DataFrame) -> bool:
    return all(p in df.columns for p in BODY_PARTS)


def load_listings(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütun(lar): {missing}. Beklenen şema: {REQUIRED}")
    df = df.dropna(subset=REQUIRED).copy()
    if (df["price"] <= 0).any():
        raise ValueError("price sütunu pozitif olmalı (log dönüşümü için).")
    if not df["damage"].isin([0, 1]).all():
        raise ValueError("damage sütunu 0/1 olmalı.")
    if has_body_parts(df):
        bad = ~df[BODY_PARTS].isin(BODY_STATES)
        if bad.any().any():
            col = bad.any()[lambda s: s].index[0]
            raise ValueError(f"Kaporta sütunu '{col}' geçersiz değer içeriyor; izin verilenler: {BODY_STATES}")
    return df.reset_index(drop=True)


def body_region_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Her bölge ve işlem türü için parça sayısı: ör. kapi_boyali = boyalı kapı adedi (0-4)."""
    out = pd.DataFrame(index=df.index)
    for region in REGIONS:
        parts = [p for p in BODY_PARTS if PART_REGION[p] == region]
        for state, key in STATE_KEYS.items():
            out[f"body_{region}_{key}"] = (df[parts] == state).sum(axis=1).astype(float)
    return out


def listing_age(df: pd.DataFrame) -> pd.Series:
    """Araç yaşı = ilan yılı - model yılı (ilanlar farklı yıllara yayıldığı için sabit bir referans yıl kullanılmaz)."""
    return (df["listing_month"].astype(str).str[:4].astype(int) - df["year"]).clip(lower=0).astype(float)


def build_features(df: pd.DataFrame, month_order: list | None = None, time_mode: str = "dummies",
                   age_by_model: bool = False) -> pd.DataFrame:
    """Sayısal + one-hot özellik matrisi.

    time_mode
        "dummies": her ilan ayı için ayrı kukla değişken (ilk ay referans). Fiyat seviyesi 10 yılda ~20 kat
                   değiştiği ve doğrusal olmadığı için OLS'te zamanı bu şekilde modelleriz; katsayılar
                   doğrudan bir piyasa endeksidir.
        "numeric": tek bir ay sırası sütunu (ağaç modelleri için yeterli, daha az sütun).
    age_by_model
        True ise her model için ayrı yaş eğimi eklenir (agex_<model> = yaş x model kuklası, referans model hariç).
        Markalar farklı hızda değer kaybettiği için OLS'te bu etkileşim olmadan diğer katsayılar yanlı olabilir.
    """
    if time_mode not in ("dummies", "numeric"):
        raise ValueError("time_mode 'dummies' veya 'numeric' olmalı")
    months = month_order or sorted(df["listing_month"].unique())
    cols: dict = {}
    cols["age"] = listing_age(df)
    cols["age_sq"] = cols["age"] ** 2
    cols["km_10k"] = df["km"] / 10_000
    cols["damage"] = df["damage"].astype(float)
    if "engine_cc" in df.columns:
        cols["engine_100cc"] = df["engine_cc"] / 100
    if time_mode == "numeric":
        cols["month_idx"] = df["listing_month"].map({m: i for i, m in enumerate(months)}).astype(float)
    else:
        for m in months[1:]:
            cols[f"period_{m}"] = (df["listing_month"] == m).astype(float)
    cats = dict(CATEGORICAL)
    cats.update({k: v for k, v in OPTIONAL_CATEGORICAL.items() if k in df.columns})
    for col, (prefix, ref) in cats.items():
        levels = sorted(df[col].unique())
        ref = ref if ref in levels else levels[0]  # referans düzey veride yoksa alfabetik ilk düzey (çoklu doğrusal bağıntıyı önler)
        for level in levels:
            if level != ref:
                cols[f"{prefix}_{level}"] = (df[col] == level).astype(float)
    if has_body_parts(df):
        cols.update(body_region_counts(df).to_dict("series"))
    model_levels = sorted(df["model"].unique())
    for level in model_levels[1:]:  # ilk model (alfabetik) referans
        cols[f"model_{level}"] = (df["model"] == level).astype(float)
    if age_by_model:
        for level in model_levels[1:]:
            cols[f"agex_{level}"] = cols["age"] * cols[f"model_{level}"]
    X = pd.concat(cols, axis=1)
    return X


# ----------------------------------------------------------------------------- OLS
def fit_ols(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Sabit terimli klasik OLS; katsayı, standart hata ve %95 güven aralığı döndürür.

    Hiç değişmeyen (sabit) sütunlar sabit terimle doğrusal bağıntılı olacağı için kabul edilmez.
    """
    const = [c for c in X.columns if X[c].nunique() <= 1]
    if const:
        raise ValueError(f"Sabit sütun(lar) OLS'e verilemez: {const[:5]}")
    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    out = pd.DataFrame({"coef": beta, "se": se}, index=["intercept", *X.columns])
    out["ci_low"] = out.coef - 1.96 * out.se
    out["ci_high"] = out.coef + 1.96 * out.se
    return out


def drop_constant(X: pd.DataFrame) -> pd.DataFrame:
    """Hiç değişmeyen sütunları atar (ör. küçük bir eğitim kümesinde hiç görülmeyen nadir kaporta durumu)."""
    return X.loc[:, X.nunique() > 1]


def ols_predict(table: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Tablodaki katsayılarla tahmin; X'te tabloda olmayan (eğitimde atılmış sabit) sütunlar yok sayılır."""
    cols = [c for c in table.index if c != "intercept"]
    return table.loc["intercept", "coef"] + X[cols].to_numpy(float) @ table.loc[cols, "coef"].to_numpy()


def pct_effect(coef: float) -> float:
    """Log-fiyat katsayısını yüzde etkiye çevirir: exp(b) - 1."""
    return float(np.expm1(coef))


def period_index(table: pd.DataFrame, months: list) -> pd.Series:
    """OLS'teki ay kuklalarından piyasa endeksini çıkarır (ilk ay = 1.0, kompozisyon etkisinden arındırılmış)."""
    vals = [1.0] + [float(np.exp(table.loc[f"period_{m}", "coef"])) for m in months[1:]]
    return pd.Series(vals, index=months, name="ols_index")


def model_depreciation(table: pd.DataFrame, models: list) -> pd.DataFrame:
    """Model bazlı yıllık değer kaybı (log eğim; ilk model referans). Yalnızca age_by_model=True ile kurulan OLS'te.

    Döndürür: model, yaş eğimi (age katsayısı + model etkileşimi) ve buna karşılık gelen yıllık % kayıp
    (kuadratik terim hariç, ilk yıl için yaklaşık).
    """
    base = float(table.loc["age", "coef"])
    rows = []
    for m in sorted(models):
        slope = base + (float(table.loc[f"agex_{m}", "coef"]) if f"agex_{m}" in table.index else 0.0)
        rows.append(dict(model=m, age_slope=slope, annual_loss_pct=-np.expm1(slope)))
    return pd.DataFrame(rows).sort_values("annual_loss_pct").reset_index(drop=True)


def body_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Bölge x işlem türü başına OLS etkisi (tek parça için fiyattaki % değişim ve %95 GA)."""
    rows = []
    for region in REGIONS:
        for state, key in STATE_KEYS.items():
            col = f"body_{region}_{key}"
            if col not in table.index:
                continue
            r = table.loc[col]
            rows.append(dict(region=region, region_label=REGION_LABELS[region], state=state, variable=col, coef=r.coef,
                             pct_effect=pct_effect(r.coef), pct_low=pct_effect(r.ci_low), pct_high=pct_effect(r.ci_high)))
    return pd.DataFrame(rows)


def body_loss_breakdown(parts: dict, effects: pd.DataFrame) -> pd.DataFrame:
    """Bir aracın kaporta durumunu (parça -> durum) parça bazlı değer kaybına çevirir.

    effects: body_effects() çıktısı (ya da aynı sütunlara sahip planlanan etkiler).
    Döndürür: parça, bölge, durum, log etkisi ve toplamın yüzdesi; son satır toplam kayıp.
    """
    coef = effects.set_index("variable")["coef"]
    rows = []
    for part, state in parts.items():
        if part not in PART_REGION:
            raise KeyError(f"Bilinmeyen parça: {part}")
        if state == "Orijinal":
            continue
        if state not in STATE_KEYS:
            raise ValueError(f"Bilinmeyen durum: {state}")
        col = f"body_{PART_REGION[part]}_{STATE_KEYS[state]}"
        rows.append(dict(part=part, region=PART_REGION[part], state=state, log_effect=float(coef.get(col, 0.0))))
    out = pd.DataFrame(rows, columns=["part", "region", "state", "log_effect"])
    out["pct_effect"] = np.expm1(out["log_effect"])
    total = float(out["log_effect"].sum())
    out.loc[len(out)] = dict(part="TOPLAM", region="", state="", log_effect=total, pct_effect=np.expm1(total))
    return out


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
GROUP_NAMES = {
    "seller": "satıcı tipi", "fuel": "yakıt", "gearbox": "vites", "city": "şehir", "model": "model",
    "color": "renk", "trim": "donanım", "period": "ilan dönemi",
}


def feature_groups(columns) -> dict:
    """One-hot sütunlarını orijinal değişkene bağlar (ör. fuel_Dizel, fuel_LPG -> 'yakıt')."""
    groups: dict = {}
    for c in columns:
        if c in ("age", "age_sq"):
            key = "yaş"
        elif c == "km_10k":
            key = "km"
        elif c in ("month_idx",) or c.startswith("period_"):
            key = "ilan dönemi"
        elif c == "damage":
            key = "tramer kaydı"
        elif c == "engine_100cc":
            key = "motor hacmi"
        elif c.startswith("agex_"):
            key = "yaş"
        elif c.startswith("body_"):
            key = "kaporta (boya/değişen)"
        else:
            key = GROUP_NAMES.get(c.split("_", 1)[0], c.split("_", 1)[0])
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
    """Yaşı tüm gözlemlerde sabitleyip ortalama tahmin edilen log fiyatı hesaplar (geometrik ortalama fiyat).

    Model bazlı yaş eğimleri (agex_<model>) varsa onlar da aynı yaşa göre güncellenir.
    """
    out = {}
    agex = [c for c in X.columns if c.startswith("agex_")]
    for a in ages:
        Xa = X.copy()
        Xa["age"] = float(a)
        Xa["age_sq"] = float(a) ** 2
        for c in agex:
            Xa[c] = float(a) * X[f"model_{c[5:]}"]
        out[a] = float(np.exp(np.mean(predict_log(Xa))))
    return pd.Series(out)
