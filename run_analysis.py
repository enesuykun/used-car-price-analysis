"""İkinci el araç fiyat analizi (Türkiye, 2016-2026): EDA, yorumlanabilir log-lineer regresyon, ağaç tabanlı
modeller, piyasa endeksi, kaporta (boya/değişen) bölge etkileri, özellik önemi ve yaşa göre değer kaybı.

Kullanım
    python generate_data.py                       # sentetik veriyi üretir
    python run_analysis.py                        # varsayılan: data/listings.csv
    python run_analysis.py --csv kendi_verim.csv  # gerçek veriyle (şema: analysis.REQUIRED)
    python run_analysis.py --fast                 # hızlı mod (az ağaç, çapraz doğrulama yok); testlerde kullanılır
"""
from __future__ import annotations

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
    BODY_PARTS,
    STATE_KEYS,
    body_effects,
    build_features,
    drop_constant,
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

ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "data" / "listings.csv"
BLUE, RED, GREY, ORANGE = "#1f4e79", "#c0392b", "#95a5a6", "#e67e22"
SEED = 0
BASELINE = "Referans (model medyanı)"


# ----------------------------------------------------------------------------- modeller
def fit_ols_model(Xtr, ytr_log, fast=False):
    table = fit_ols(drop_constant(Xtr), ytr_log)
    return lambda X: ols_predict(table, X), table


def fit_rf(Xtr, ytr_log, fast=False):
    m = RandomForestRegressor(n_estimators=40 if fast else 200, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                              random_state=SEED).fit(Xtr, ytr_log)
    return lambda X: m.predict(X), m


def fit_gb(Xtr, ytr_log, fast=False):
    m = HistGradientBoostingRegressor(max_iter=150 if fast else 800, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=SEED).fit(Xtr, ytr_log)
    return lambda X: m.predict(X), m


# model adı -> (eğitim fonksiyonu, kullandığı özellik seti). OLS ay kuklaları, ağaçlar sayısal ay sırası kullanır.
MODELS = {
    "OLS (log-lineer)": (fit_ols_model, "dummies"),
    "Random Forest": (fit_rf, "numeric"),
    "Gradient Boosting": (fit_gb, "numeric"),
}


def baseline_predict(df_train, price_train, df_eval) -> np.ndarray:
    """Referans: her modelin eğitimdeki ve aynı ilan yılındaki medyan fiyatı (yaş, km vb. hiç kullanılmaz)."""
    key_tr = df_train["model"].to_numpy() + "|" + df_train["listing_month"].str[:4].to_numpy()
    key_ev = df_eval["model"].to_numpy() + "|" + df_eval["listing_month"].str[:4].to_numpy()
    med = price_train.groupby(key_tr).median()
    by_model = price_train.groupby(df_train["model"].to_numpy()).median()
    out = pd.Series(key_ev).map(med)
    return out.fillna(pd.Series(df_eval["model"].to_numpy()).map(by_model)).fillna(price_train.median()).to_numpy()


def score(actual, pred) -> dict:
    return {"MAPE": mape(actual, pred), "MedAPE": median_ape(actual, pred), "MAE_TL": mae(actual, pred), "R2": r2(actual, pred)}


def cross_validate(feats: dict, y_log, df, n_splits=5) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits, shuffle=True, random_state=SEED)
    price = np.exp(y_log)
    for name in [BASELINE, *MODELS]:
        mapes = []
        for tr, va in kf.split(df):
            if name == BASELINE:
                pred = baseline_predict(df.iloc[tr], pd.Series(price[tr], index=df.index[tr]), df.iloc[va])
            else:
                fit, fs = MODELS[name]
                predict, _ = fit(feats[fs].iloc[tr], y_log[tr])
                pred = np.exp(predict(feats[fs].iloc[va]))
            mapes.append(mape(price[va], pred))
        rows.append((name, np.mean(mapes), np.std(mapes)))
    return pd.DataFrame(rows, columns=["model", "cv_mape_mean", "cv_mape_std"])


# ----------------------------------------------------------------------------- grafikler
def money(ax, axis="y"):
    def fmt(v, _):
        return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(plt.FuncFormatter(fmt))


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_eda(df, images: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 1) yıllara göre ortalama ve medyan fiyat (nominal, log ölçek)
    ax = axes[0, 0]
    g = df.groupby(df.listing_month.str[:4]).price
    ax.plot(g.mean().index, g.mean().values, marker="o", color=BLUE, lw=2, label="Ortalama")
    ax.plot(g.median().index, g.median().values, marker="s", color=GREY, lw=1.5, label="Medyan")
    ax.set_yscale("log")
    money(ax)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    style(ax, "İlan yılına göre fiyat (nominal TL, log)")
    # 2) yaşa göre fiyat, ilan yılının medyanına oranla (enflasyondan arındırılmış görünüm)
    ax = axes[0, 1]
    rel = df.price / df.groupby(df.listing_month.str[:4]).price.transform("median")
    age = listing_age(df)
    gr = rel.groupby(age.clip(upper=18))
    ax.fill_between(gr.median().index, gr.quantile(0.25), gr.quantile(0.75), color=BLUE, alpha=0.18, label="Çeyrekler arası")
    ax.plot(gr.median().index, gr.median().values, color=BLUE, lw=2.2, label="Medyan")
    ax.set_xlabel("Araç yaşı")
    ax.set_ylabel("Fiyat / o yılın medyanı")
    ax.legend(frameon=False)
    style(ax, "Yaşa göre göreli fiyat")
    # 3) renk payları (model yılına göre)
    ax = axes[1, 0]
    if "color" in df.columns:
        yr = df.year.clip(lower=2008)
        counts = yr.value_counts()
        keep = yr.isin(counts[counts >= 150].index)  # az ilanlı yıllar (en yeni model yılları) gürültülü
        sh = pd.crosstab(yr[keep], df.color[keep], normalize="index")
        order = sh.mean().sort_values(ascending=False).index
        colors = {"Beyaz": "#d9d9d9", "Gri": "#7f8c8d", "Siyah": "#2c3e50", "Kırmızı": RED, "Mavi": "#2e86c1", "Diğer": ORANGE}
        ax.stackplot(sh.index, *[sh[c] for c in order], labels=order, colors=[colors.get(c, GREY) for c in order], alpha=0.9)
        ax.set_xlabel("Model yılı")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("Pay")
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
        style(ax, "Renk payı: beyazdan griye kayış")
    # 4) kaporta: parça başına boyalı/değişen sıklığı
    ax = axes[1, 1]
    if has_body_parts(df):
        freq = pd.DataFrame({s: (df[BODY_PARTS] == s).mean() for s in STATE_KEYS}).sort_values("Boyalı")
        left = np.zeros(len(freq))
        for s, c in zip(STATE_KEYS, ["#f5cba7", ORANGE, RED]):
            ax.barh(freq.index, freq[s] * 100, left=left, color=c, label=s)
            left += freq[s].to_numpy() * 100
        ax.set_xlabel("İlanların yüzdesi")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax, "Parça bazında boya / değişen sıklığı")
    fig.tight_layout()
    fig.savefig(images / "eda.png", dpi=130)
    plt.close(fig)


def plot_model_comparison(test_scores, y_true, y_pred_best, best_name, images: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = list(test_scores.index)
    vals = test_scores["MAPE"].to_numpy() * 100
    bars = ax.bar(names, vals, color=[RED if n == best_name else GREY if n == BASELINE else BLUE for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"%{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Test MAPE (%)")
    ax.tick_params(axis="x", rotation=15)
    style(ax, "Model karşılaştırması (düşük = iyi)")
    ax = axes[1]
    ax.scatter(y_true, y_pred_best, s=4, alpha=0.25, color=BLUE)
    lim = [min(y_true.min(), y_pred_best.min()), max(y_true.max(), y_pred_best.max())]
    ax.plot(lim, lim, color=RED, lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gerçek fiyat (TL)")
    ax.set_ylabel("Tahmin (TL)")
    style(ax, f"{best_name}: test seti")
    fig.tight_layout()
    fig.savefig(images / "model_comparison.png", dpi=130)
    plt.close(fig)


def _effect_plot(ax, d: pd.DataFrame, synthetic: bool, labels):
    y = np.arange(len(d))
    ax.errorbar(d.pct_effect * 100, y, xerr=[(d.pct_effect - d.pct_low) * 100, (d.pct_high - d.pct_effect) * 100],
                fmt="o", color=BLUE, capsize=3, label="OLS tahmini (%95 GA)")
    if synthetic and d.planted_pct.notna().any():
        ax.scatter(d.planted_pct * 100, y, marker="D", s=60, facecolors="none", edgecolors=RED, linewidths=1.5, zorder=5,
                   label="Veride kurulan gerçek etki")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fiyata etkisi (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_effects(eff: pd.DataFrame, synthetic: bool, images: Path):
    d = eff.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(d) + 1.4))
    _effect_plot(ax, d, synthetic, d.label)
    style(ax, "Diğer her şey sabitken fiyat etkileri")
    fig.tight_layout()
    fig.savefig(images / "ols_effects.png", dpi=130)
    plt.close(fig)


def plot_body_effects(body: pd.DataFrame, synthetic: bool, images: Path):
    d = body.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(d) + 1.5))
    _effect_plot(ax, d, synthetic, d.region_label + " — " + d.state)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "Kaporta: tek parçanın fiyata etkisi (bölge × işlem)")
    fig.tight_layout()
    fig.savefig(images / "body_part_effects.png", dpi=130)
    plt.close(fig)


def plot_market_index(idx: pd.DataFrame, images: Path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = pd.PeriodIndex(idx.index, freq="M").to_timestamp()
    ax.plot(x, idx["raw_mean_index"], color=GREY, lw=1.2, label="Ham aylık ortalama fiyat (kompozisyon etkili)")
    ax.plot(x, idx["ols_index"], color=BLUE, lw=2, label="OLS ay etkileri (aynı araç, sabit özellikler)")
    if "planted_index" in idx:
        ax.plot(x, idx["planted_index"], color=RED, lw=1.3, ls="--", label="Veride kurulan gerçek endeks")
    ax.set_yscale("log")
    ax.set_ylabel(f"Endeks ({idx.index[0]} = 1, log)")
    for when, text in [("2018-08", "2018 kur şoku"), ("2020-06", "Pandemi"), ("2021-12", "Ara 2021 kur krizi"), ("2022-06", "Haz 2022 düzenleme")]:
        t = pd.Period(when, "M").to_timestamp()
        ax.axvline(t, color="black", lw=0.6, ls=":")
        ax.text(t, ax.get_ylim()[0] * 1.15, " " + text, fontsize=8, rotation=90, va="bottom", ha="right")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "İkinci el piyasa endeksi (nominal)")
    fig.tight_layout()
    fig.savefig(images / "market_index.png", dpi=130)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, name, images: Path):
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(d) + 1.3))
    ax.barh(d.variable, d.rmse_increase, xerr=d["std"], color=BLUE, capsize=2)
    ax.set_xlabel("Karıştırınca log-RMSE artışı")
    style(ax, f"Özellik önemi (permütasyon, {name})")
    fig.tight_layout()
    fig.savefig(images / "permutation_importance.png", dpi=130)
    plt.close(fig)


def plot_age_curve(curves: dict, truth: pd.Series | None, images: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (name, s), color in zip(curves.items(), [BLUE, RED]):
        ax.plot(s.index, s.values / s.iloc[0] * 100, marker="o", ms=4, lw=2, label=name, color=color)
    if truth is not None:
        ax.plot(truth.index, truth.values * 100, color="black", lw=1.6, ls="--", label="Veride kurulan gerçek eğri")
    ax.set_xlabel("Araç yaşı")
    ax.set_ylabel("Sıfır yaşa göre fiyat (%)")
    ax.legend(frameon=False)
    style(ax, "Yaşa göre değer kaybı (diğer her şey sabit)")
    fig.tight_layout()
    fig.savefig(images / "age_depreciation.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- ana akış
LABELS = {
    "damage": "Tramer kaydı var", "gearbox_Otomatik": "Otomatik vites", "fuel_Dizel": "Dizel (vs Benzin)",
    "fuel_Hibrit": "Hibrit (vs Benzin)", "fuel_LPG": "LPG (vs Benzin)", "seller_Galeri": "Galeriden (vs sahibinden)",
    "city_İstanbul": "İstanbul (vs Diğer)", "city_Ankara": "Ankara (vs Diğer)", "city_İzmir": "İzmir (vs Diğer)",
    "km_10k": "+10.000 km", "engine_100cc": "+100 cm³ motor hacmi", "trim_Orta": "Orta donanım (vs Temel)",
    "trim_Üst": "Üst donanım (vs Temel)", "color_Gri": "Gri (vs Beyaz)", "color_Siyah": "Siyah (vs Beyaz)",
    "color_Kırmızı": "Kırmızı (vs Beyaz)", "color_Mavi": "Mavi (vs Beyaz)", "color_Diğer": "Diğer renk (vs Beyaz)",
}


def planted_effects() -> tuple[dict, dict]:
    """Sentetik veride kurulan gerçek etkiler: (genel etkiler, kaporta özelliği -> log etkisi)."""
    from generate_data import BODY_PENALTY, LOCAL_SHARE, TRUE_EFFECTS

    body = {}
    for region, pen in BODY_PENALTY.items():
        body[f"body_{region}_boyali"] = -pen["Boyalı"]
        body[f"body_{region}_degisen"] = -pen["Değişen"]
        body[f"body_{region}_lokal"] = -LOCAL_SHARE * pen["Boyalı"]
    return dict(TRUE_EFFECTS), body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--images-dir", default=str(ROOT / "images"))
    ap.add_argument("--fast", action="store_true", help="az ağaç, çapraz doğrulama yok")
    ap.add_argument("--synthetic", choices=["auto", "yes", "no"], default="auto",
                    help="kurulan gerçek etkilerle karşılaştır (auto: yalnızca varsayılan sentetik CSV'de)")
    args = ap.parse_args(argv)
    results, images = Path(args.results_dir), Path(args.images_dir)
    synthetic = (Path(args.csv).resolve() == DEFAULT_CSV.resolve()) if args.synthetic == "auto" else args.synthetic == "yes"
    results.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    df = load_listings(args.csv)
    months = sorted(df["listing_month"].unique())
    feats = {"dummies": build_features(df, month_order=months, time_mode="dummies", age_by_model=True),
             "numeric": build_features(df, month_order=months, time_mode="numeric")}
    y_log = np.log(df["price"].to_numpy(float))
    print(f"{len(df):,} ilan, {months[0]} - {months[-1]}, OLS {feats['dummies'].shape[1]} / ağaç {feats['numeric'].shape[1]} özellik "
          f"({'sentetik' if synthetic else 'kullanıcı verisi'})")
    plot_eda(df, images)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    ytr, yte = y_log[idx_tr], y_log[idx_te]
    price_te = np.exp(yte)

    # --- test seti performansı
    preds, fitted = {}, {}
    preds[BASELINE] = baseline_predict(df.iloc[idx_tr], pd.Series(np.exp(ytr), index=df.index[idx_tr]), df.iloc[idx_te])
    for name, (fit, fs) in MODELS.items():
        predict, obj = fit(feats[fs].iloc[idx_tr], ytr, fast=args.fast)
        fitted[name] = (predict, obj, fs)
        preds[name] = np.exp(predict(feats[fs].iloc[idx_te]))
    test_scores = pd.DataFrame({k: score(price_te, v) for k, v in preds.items()}).T
    out = test_scores
    if not args.fast:
        cv = cross_validate({k: v.iloc[idx_tr].reset_index(drop=True) for k, v in feats.items()}, ytr,
                            df.iloc[idx_tr].reset_index(drop=True))
        out = test_scores.join(cv.set_index("model"))
    out.to_csv(results / "model_comparison.csv")
    pretty = out.copy()
    for c in ("MAPE", "MedAPE", "cv_mape_mean", "cv_mape_std"):
        if c in pretty:
            pretty[c] = pretty[c] * 100
    print(pretty.round(2).to_string())

    best = test_scores.drop(index=BASELINE)["MAPE"].idxmin()
    plot_model_comparison(test_scores, price_te, preds[best], best, images)

    # --- OLS: yorumlanabilir etkiler (tüm veriyle)
    table = fit_ols(drop_constant(feats["dummies"]), y_log)
    planted, planted_body = planted_effects() if synthetic else ({}, {})

    def with_planted(r, col):
        p = planted.get(col, planted_body.get(col))
        return dict(planted_pct=pct_effect(p) if p is not None else np.nan,
                    ci_covers_planted=bool(r.ci_low <= p <= r.ci_high) if p is not None else None)

    rows = []
    for col, label in LABELS.items():
        if col in table.index:
            r = table.loc[col]
            rows.append(dict(variable=col, label=label, coef=r.coef, pct_effect=pct_effect(r.coef), pct_low=pct_effect(r.ci_low),
                             pct_high=pct_effect(r.ci_high), **with_planted(r, col)))
    eff = pd.DataFrame(rows)
    eff.to_csv(results / "ols_effects.csv", index=False)
    plot_effects(eff, synthetic, images)

    covered = eff.ci_covers_planted.dropna()
    body = pd.DataFrame()
    if has_body_parts(df):
        body = body_effects(table)
        body = body.join(pd.DataFrame([with_planted(table.loc[v], v) for v in body.variable], index=body.index))
        body.to_csv(results / "body_part_effects.csv", index=False)
        plot_body_effects(body, synthetic, images)
        covered = pd.concat([covered, body.ci_covers_planted.dropna()])
    if synthetic:
        print(f"\nOLS %95 güven aralığı kurulan gerçek etkiyi kapsıyor: {int(covered.sum())}/{len(covered)} katsayı")
    if len(body):
        top = body.sort_values("pct_effect").head(5)
        print("\nEn pahalı kaporta işlemleri (tek parça, OLS):")
        print(top.assign(pct=lambda d: (d.pct_effect * 100).round(1))[["region_label", "state", "pct"]].to_string(index=False))

    # --- model bazlı yıllık değer kaybı
    dep = model_depreciation(table, sorted(df["model"].unique()))
    if synthetic:
        from generate_data import MODELS as GEN_MODELS

        dep["planted_annual_loss_pct"] = dep.model.map(lambda m: -np.expm1(-GEN_MODELS[m]["dep"]))
    dep.to_csv(results / "model_depreciation.csv", index=False)
    print("\nYıllık değer kaybı en düşük / en yüksek modeller (OLS, %):")
    print(pd.concat([dep.head(3), dep.tail(3)]).assign(pct=lambda d: (d.annual_loss_pct * 100).round(1))[["model", "pct"]].to_string(index=False))

    # --- piyasa endeksi: OLS ay etkileri vs ham ortalama (vs kurulan gerçek)
    raw = df.groupby("listing_month").price.mean().reindex(months)
    idx = pd.DataFrame({"ols_index": period_index(table, months), "raw_mean_index": raw / raw.iloc[0]})
    if synthetic:
        from generate_data import market_index

        planted_idx = market_index().reindex(months)
        idx["planted_index"] = planted_idx / planted_idx.iloc[0]
        err = (np.log(idx.ols_index) - np.log(idx.planted_index)).abs()
        print(f"\nOLS piyasa endeksi, kurulan endeksten ortalama {err.mean() * 100:.1f} log-puan sapıyor (en çok {err.max() * 100:.1f}).")
    idx.to_csv(results / "market_index.csv", index_label="month")
    plot_market_index(idx, images)

    # --- özellik önemi ve yaş eğrisi (en iyi model)
    predict_best, _, fs_best = fitted[best]
    Xte_best = feats[fs_best].iloc[idx_te]
    imp = grouped_permutation_importance(predict_best, Xte_best, yte, n_repeats=3 if args.fast else 8, seed=SEED)
    imp.to_csv(results / "permutation_importance.csv", index=False)
    plot_importance(imp, best, images)
    print("\nÖzellik önemi (ilk 6):")
    print(imp.head(6).round(4).to_string(index=False))

    ages = list(range(0, int(min(feats["numeric"]["age"].max(), 18)) + 1))
    curves = {"OLS (kuadratik yaş)": partial_dependence_age(fitted["OLS (log-lineer)"][0], feats["dummies"].iloc[idx_te], ages)}
    best_tree = test_scores.drop(index=[BASELINE, "OLS (log-lineer)"])["MAPE"].idxmin()
    predict_tree, _, fs_tree = fitted[best_tree]
    curves[best_tree] = partial_dependence_age(predict_tree, feats[fs_tree].iloc[idx_te], ages)
    truth = None
    if synthetic:
        from generate_data import AGE_QUAD, MODELS as GEN_MODELS

        dep_te = df.iloc[idx_te]["model"].map({k: v["dep"] for k, v in GEN_MODELS.items()}).to_numpy()
        truth = pd.Series({a: float(np.exp(np.mean(-dep_te * a) + AGE_QUAD * a**2)) for a in ages})
        truth = truth / truth.iloc[0]
    out_curves = pd.DataFrame(curves)
    if truth is not None:
        out_curves["Veride kurulan gerçek eğri (referans 1.0)"] = truth
    out_curves.to_csv(results / "age_depreciation_curve.csv", index_label="age")
    plot_age_curve(curves, truth, images)

    pd.DataFrame({"actual": price_te, **{k: v for k, v in preds.items()}}).round(0).to_csv(results / "test_predictions.csv", index=False)
    print(f"\nTamamlandı: {results} ve {images} güncellendi.")
    return dict(test_scores=test_scores, effects=eff, body=body, index=idx, importance=imp, best=best, depreciation=dep)


if __name__ == "__main__":
    main()
