"""İkinci el araç fiyat analizi: EDA, yorumlanabilir log-lineer regresyon, ağaç tabanlı modeller,
özellik önemi ve yaşa göre değer kaybı eğrisi.

Kullanım
    python generate_data.py                       # sentetik veriyi üretir
    python run_analysis.py                        # varsayılan: data/listings.csv
    python run_analysis.py --csv kendi_verim.csv  # gerçek veriyle (şema: analysis.REQUIRED)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split

from analysis import (
    build_features,
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

ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "data" / "listings.csv"
RESULTS, IMAGES = ROOT / "results", ROOT / "images"
BLUE, RED, GREY = "#1f4e79", "#c0392b", "#95a5a6"
SEED = 0


# ----------------------------------------------------------------------------- modeller
def fit_ols_model(Xtr, ytr_log):
    table = fit_ols(Xtr, ytr_log)
    return lambda X: ols_predict(table, X), table


def fit_rf(Xtr, ytr_log):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED).fit(Xtr, ytr_log)
    return lambda X: m.predict(X), m


def fit_gb(Xtr, ytr_log):
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED).fit(Xtr, ytr_log)
    return lambda X: m.predict(X), m


MODELS = {
    "OLS (log-lineer)": fit_ols_model,
    "Random Forest": fit_rf,
    "Gradient Boosting": fit_gb,
}


def baseline_predict(df_train, price_train, df_eval) -> np.ndarray:
    """Referans: her modelin eğitimdeki medyan fiyatı (yaş, km vb. hiç kullanılmaz)."""
    med = price_train.groupby(df_train["model"].to_numpy()).median()
    return df_eval["model"].map(med).fillna(price_train.median()).to_numpy()


def score(actual, pred) -> dict:
    return {"MAPE": mape(actual, pred), "MedAPE": median_ape(actual, pred), "MAE_TL": mae(actual, pred), "R2": r2(actual, pred)}


def cross_validate(X, y_log, df, n_splits=5) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits, shuffle=True, random_state=SEED)
    price = np.exp(y_log)
    for name in ["Referans (model medyanı)", *MODELS]:
        mapes = []
        for tr, va in kf.split(X):
            if name.startswith("Referans"):
                pred = baseline_predict(df.iloc[tr], pd.Series(price[tr], index=df.index[tr]), df.iloc[va])
            else:
                predict, _ = MODELS[name](X.iloc[tr], y_log[tr])
                pred = np.exp(predict(X.iloc[va]))
            mapes.append(mape(price[va], pred))
        rows.append((name, np.mean(mapes), np.std(mapes)))
    return pd.DataFrame(rows, columns=["model", "cv_mape_mean", "cv_mape_std"])


# ----------------------------------------------------------------------------- grafikler
def money(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    age = (df.year.max() - df.year)
    g = df.groupby(age).price
    ax = axes[0]
    ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=BLUE, alpha=0.18, label="Çeyrekler arası")
    ax.plot(g.median().index, g.median().values, color=BLUE, lw=2.2, label="Medyan")
    ax.set_xlabel("Araç yaşı")
    ax.set_ylabel("Fiyat (TL)")
    money(ax)
    ax.legend(frameon=False)
    style(ax, "Yaşa göre fiyat")
    ax = axes[1]
    order = df.groupby("fuel").price.median().sort_values(ascending=False).index
    ax.boxplot([df[df.fuel == f].price for f in order], tick_labels=order, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#d6e4f0", color=BLUE), medianprops=dict(color=RED, lw=2))
    money(ax)
    style(ax, "Yakıt tipine göre fiyat (ham)")
    fig.tight_layout()
    fig.savefig(IMAGES / "eda.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(test_scores, y_true, y_pred_best, best_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = list(test_scores.index)
    vals = test_scores["MAPE"].to_numpy() * 100
    bars = ax.bar(names, vals, color=[GREY, BLUE, BLUE, RED][: len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"%{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Test MAPE (%)")
    ax.tick_params(axis="x", rotation=15)
    style(ax, "Model karşılaştırması (düşük = iyi)")
    ax = axes[1]
    ax.scatter(y_true, y_pred_best, s=5, alpha=0.3, color=BLUE)
    lim = [min(y_true.min(), y_pred_best.min()), max(y_true.max(), y_pred_best.max())]
    ax.plot(lim, lim, color=RED, lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gerçek fiyat (TL)")
    ax.set_ylabel("Tahmin (TL)")
    style(ax, f"{best_name}: test seti")
    fig.tight_layout()
    fig.savefig(IMAGES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_effects(eff: pd.DataFrame, synthetic: bool):
    d = eff.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.errorbar(d.pct_effect * 100, y, xerr=[(d.pct_effect - d.pct_low) * 100, (d.pct_high - d.pct_effect) * 100],
                fmt="o", color=BLUE, capsize=3, label="OLS tahmini (%95 GA)")
    if synthetic and d.planted_pct.notna().any():
        ax.scatter(d.planted_pct * 100, y, marker="D", s=70, facecolors="none", edgecolors=RED, linewidths=1.6, zorder=5, label="Veride kurulan gerçek etki")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label)
    ax.set_xlabel("Fiyata etkisi (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "Diğer her şey sabitken fiyat etkileri")
    fig.tight_layout()
    fig.savefig(IMAGES / "ols_effects.png", dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, name):
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(d) + 1.3))
    ax.barh(d.variable, d.rmse_increase, xerr=d["std"], color=BLUE, capsize=2)
    ax.set_xlabel("Karıştırınca log-RMSE artışı")
    style(ax, f"Özellik önemi (permütasyon, {name})")
    fig.tight_layout()
    fig.savefig(IMAGES / "permutation_importance.png", dpi=150)
    plt.close(fig)


def plot_age_curve(curves: dict, truth: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (name, s), color in zip(curves.items(), [GREY, RED]):
        ax.plot(s.index, s.values / s.iloc[0] * 100, marker="o", ms=4, lw=2, label=name, color=color)
    if truth is not None:
        ax.plot(truth.index, truth.values * 100, color="black", lw=1.6, ls="--", label="Veride kurulan gerçek eğri")
    ax.set_xlabel("Araç yaşı")
    ax.set_ylabel("Sıfır yaşa göre fiyat (%)")
    ax.legend(frameon=False)
    style(ax, "Yaşa göre değer kaybı (diğer her şey sabit)")
    fig.tight_layout()
    fig.savefig(IMAGES / "age_depreciation.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- ana akış
LABELS = {
    "damage": "Hasar kaydı var", "gearbox_Otomatik": "Otomatik vites", "fuel_Dizel": "Dizel (vs Benzin)",
    "fuel_Hibrit": "Hibrit (vs Benzin)", "fuel_LPG": "LPG (vs Benzin)", "seller_Galeri": "Galeriden (vs sahibinden)",
    "city_İstanbul": "İstanbul (vs Diğer)", "city_Ankara": "Ankara (vs Diğer)", "city_İzmir": "İzmir (vs Diğer)",
    "km_10k": "+10.000 km", "month_idx": "+1 ay (enflasyon/trend)",
}
PLANTED_MAP = {"km_per_10k": "km_10k", "per_month": "month_idx", "seller_Galeri": "seller_Galeri"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    synthetic = Path(args.csv).resolve() == DEFAULT_CSV.resolve()
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    df = load_listings(args.csv)
    X, y_log = build_features(df), np.log(df["price"].to_numpy(float))
    print(f"{len(df):,} ilan, {X.shape[1]} özellik ({'sentetik' if synthetic else 'kullanıcı verisi'})")
    plot_eda(df)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y_log[idx_tr], y_log[idx_te]
    price_te = np.exp(yte)

    # --- test seti performansı
    preds, fitted = {}, {}
    preds["Referans (model medyanı)"] = baseline_predict(df.iloc[idx_tr], pd.Series(np.exp(ytr), index=df.index[idx_tr]), df.iloc[idx_te])
    for name, fit in MODELS.items():
        predict, obj = fit(Xtr, ytr)
        fitted[name] = (predict, obj)
        preds[name] = np.exp(predict(Xte))
    test_scores = pd.DataFrame({k: score(price_te, v) for k, v in preds.items()}).T
    cv = cross_validate(X.iloc[idx_tr].reset_index(drop=True), ytr, df.iloc[idx_tr].reset_index(drop=True))
    out = test_scores.join(cv.set_index("model"))
    out.to_csv(RESULTS / "model_comparison.csv")
    print((out.assign(MAPE=out.MAPE * 100, MedAPE=out.MedAPE * 100, cv_mape_mean=out.cv_mape_mean * 100, cv_mape_std=out.cv_mape_std * 100).round(2)).to_string())

    best = test_scores.drop(index="Referans (model medyanı)")["MAPE"].idxmin()
    plot_model_comparison(test_scores, price_te, preds[best], best)

    # --- OLS: yorumlanabilir etkiler (tüm veriyle)
    table = fit_ols(X, y_log)
    rows = []
    planted = {}
    if synthetic:
        from generate_data import TRUE_EFFECTS

        planted = {PLANTED_MAP.get(k, k): v for k, v in TRUE_EFFECTS.items()}
    for col, label in LABELS.items():
        if col not in table.index:
            continue
        r = table.loc[col]
        p = planted.get(col)
        rows.append(dict(variable=col, label=label, coef=r.coef, pct_effect=pct_effect(r.coef),
                         pct_low=pct_effect(r.ci_low), pct_high=pct_effect(r.ci_high),
                         planted_pct=pct_effect(p) if p is not None else np.nan,
                         ci_covers_planted=(r.ci_low <= p <= r.ci_high) if p is not None else None))
    eff = pd.DataFrame(rows)
    eff.to_csv(RESULTS / "ols_effects.csv", index=False)
    plot_effects(eff, synthetic)
    if synthetic:
        print(f"\nOLS %95 güven aralığı kurulan gerçek etkiyi kapsıyor: {int(eff.ci_covers_planted.sum())}/{len(eff)} katsayı")

    # --- özellik önemi ve yaş eğrisi (en iyi model)
    predict_best, _ = fitted[best]
    imp = grouped_permutation_importance(predict_best, Xte, yte, n_repeats=10, seed=SEED)
    imp.to_csv(RESULTS / "permutation_importance.csv", index=False)
    plot_importance(imp, best)
    print("\nÖzellik önemi (ilk 5):")
    print(imp.head(5).round(4).to_string(index=False))

    ages = list(range(0, int(X["age"].max()) + 1, 1))
    curves = {
        "OLS (kuadratik yaş)": partial_dependence_age(fitted["OLS (log-lineer)"][0], Xte, ages),
        best: partial_dependence_age(predict_best, Xte, ages),
    }
    truth = None
    if synthetic:
        from generate_data import AGE_QUAD, MODELS as GEN_MODELS

        dep = df.iloc[idx_te]["model"].map({k: v[1] for k, v in GEN_MODELS.items()}).to_numpy()
        truth = pd.Series({a: float(np.exp(np.mean(-dep * a) + AGE_QUAD * a**2)) for a in ages})
        truth = truth / truth.iloc[0]
    out_curves = pd.DataFrame(curves)
    if truth is not None:
        out_curves["Veride kurulan gerçek eğri (referans 1.0)"] = truth
    out_curves.to_csv(RESULTS / "age_depreciation_curve.csv", index_label="age")
    plot_age_curve(curves, truth)

    pd.DataFrame({"actual": price_te, **{k: v for k, v in preds.items()}}).round(0).to_csv(RESULTS / "test_predictions.csv", index=False)
    print("\nTamamlandı: results/ ve images/ güncellendi.")


if __name__ == "__main__":
    main()
