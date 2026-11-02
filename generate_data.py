"""Türkiye ikinci el otomobil piyasasına kalibre edilmiş sentetik ilan verisi üretici (Ocak 2016 - Ekim 2026).

Veri SENTETİKTİR: tek tek ilanlar gerçek değildir. Ancak genel seviyeler ve dağılımlar kamuya açık
kaynaklara göre ayarlanmıştır (ayrıntı ve kaynaklar: README.md, "Kalibrasyon" bölümü):

* Ortalama ilan fiyatının seyri (PRICE_ANCHORS): 2023 sonrası BETAM/sahibindex raporlarındaki ortalamalar;
  2016-2022 arası, bilinen olaylara (2018 kur şoku, 2020 pandemi talebi, Aralık 2021 kur krizi,
  2022 ortası düzenleme sonrası yatay seyir) göre yaklaşık değerlerdir.
* Renk payları zamanla beyazdan griye kayar (TÜİK ve Cardata verilerindeki eğilim).
* Yakıt payında dizel düşer, hibrit artar; LPG sonradan dönüşüm olarak benzinli araçlara takılır.
* Motor hacimleri model başına gerçek seçeneklerden seçilir (çoğunluk 1.6 L altı, ÖTV dilimleri nedeniyle).

Fiyat, bilinen "gerçek" etkilerle üretilir (TRUE_EFFECTS, BODY_PENALTY). Böylece analizde
regresyonun bu etkileri geri bulup bulamadığı kontrol edilebilir (bkz. run_analysis.py).

Çıktı: data/listings.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import BODY_PARTS, PART_REGION, STATE_KEYS

SEED = 21
N = 16_000
OUT = Path(__file__).parent / "data"
MONTHS = pd.period_range("2016-01", "2026-10", freq="M").astype(str).tolist()

# ----------------------------------------------------------------------------- piyasa seviyesi
# Ortalama ikinci el ilan fiyatı (TL). Aradaki aylar log-doğrusal enterpolasyonla doldurulur.
# 2023-11 ve sonrası: BETAM/sahibindex aylık raporları. 2022-11: 2023-11 değerinin yıllık %76,2 artışından geri hesap.
# 2016-2022 diğer noktalar: kesin kaynak bulunamadı, olaylara göre yaklaşık değer (README'de belirtildi).
PRICE_ANCHORS = {
    "2016-01": 62_000,
    "2017-01": 68_000,
    "2018-01": 78_000,
    "2018-10": 92_000,     # Ağustos 2018 kur şoku
    "2019-06": 98_000,
    "2020-01": 110_000,
    "2020-08": 165_000,    # pandemi: toplu taşımadan kaçış + ucuz kredi
    "2021-01": 175_000,
    "2021-11": 270_000,    # çip krizi, sıfır araç arzı daraldı
    "2022-01": 360_000,    # Aralık 2021 kur krizi
    "2022-06": 520_000,
    "2022-11": 499_000,    # Haziran 2022 düzenlemesi (6 ay / 6.000 km kuralı) sonrası yatay seyir
    "2023-05": 650_000,
    "2023-11": 879_000,    # BETAM: Kasım 2023 ortalaması 879.146 TL
    "2023-12": 860_000,    # BETAM: Aralık 2023 ortalaması 860.480 TL
    "2024-06": 969_000,    # sahibinden raporu: Haziran 2024 ortalaması 968.926 TL
    "2025-09": 1_058_000,  # sahibinden: Eylül 2025 ortalaması 1.058.000 TL
    "2025-11": 1_086_000,  # BETAM: Kasım 2025 ~1,086 milyon TL
    "2026-03": 1_160_000,  # BETAM: Mart 2026 ~1,16 milyon TL
    "2026-06": 1_175_000,  # BETAM: Haziran 2026 ~1,175 milyon TL
    "2026-10": 1_205_000,  # yıllık ~%8 nominal artış eğilimiyle uzatıldı
}


def market_index(months=MONTHS, anchors=PRICE_ANCHORS) -> pd.Series:
    """Aylık piyasa endeksi (ilk ay = 1.0). Çapa noktaları arasında log-doğrusal enterpolasyon."""
    pos = {m: i for i, m in enumerate(months)}
    xs = np.array([pos[m] for m in anchors])
    ys = np.log(np.array(list(anchors.values()), dtype=float))
    log_level = np.interp(np.arange(len(months)), xs, ys)
    return pd.Series(np.exp(log_level - log_level[0]), index=months, name="index")


# ----------------------------------------------------------------------------- modeller
# model: segment, ilan payı (göreli), Ocak 2016 sıfır fiyatı (TL, yaklaşık), yıllık değer kaybı (log),
#        launch: bu model adının Türkiye'de satıldığı ilk model yılı (önceki kasalar dahil), motor seçenekleri [(cc, yakıt, ilk model yılı)], donanım paketleri (temel -> üst)
MODELS = {
    "Fiat Egea":          dict(seg="ekonomi", w=1.7, base=52_000, dep=0.050, launch=2015,
                               engines=[(1368, "Benzin", 2015), (1598, "Benzin", 2015), (1248, "Dizel", 2015), (1598, "Dizel", 2015), (1469, "Hibrit", 2024)],
                               trims=["Easy", "Urban", "Lounge"]),
    "Renault Clio":       dict(seg="ekonomi", w=1.5, base=55_000, dep=0.055, launch=2004,
                               engines=[(1149, "Benzin", 2013), (898, "Benzin", 2013), (999, "Benzin", 2020), (1461, "Dizel", 2013), (1598, "Hibrit", 2021)],
                               trims=["Joy", "Touch", "Icon"]),
    "Renault Megane":     dict(seg="orta", w=1.0, base=72_000, dep=0.060, launch=2004,
                               engines=[(1598, "Benzin", 2013), (1332, "Benzin", 2019), (1461, "Dizel", 2013)],
                               trims=["Joy", "Touch", "Icon"]),
    "Toyota Corolla":     dict(seg="orta", w=1.1, base=78_000, dep=0.045, launch=2004,
                               engines=[(1329, "Benzin", 2013), (1598, "Benzin", 2013), (1798, "Hibrit", 2019), (1364, "Dizel", 2013)],
                               trims=["Life", "Vision", "Dream"]),
    "Toyota C-HR":        dict(seg="suv", w=0.4, base=110_000, dep=0.050, launch=2016,
                               engines=[(1798, "Hibrit", 2017), (1197, "Benzin", 2017)],
                               trims=["Flame", "Passion", "Passion X"]),
    "Volkswagen Polo":    dict(seg="ekonomi", w=0.8, base=62_000, dep=0.055, launch=2004,
                               engines=[(999, "Benzin", 2018), (1198, "Benzin", 2013), (1422, "Dizel", 2013), (1598, "Dizel", 2018)],
                               trims=["Trendline", "Comfortline", "Highline"]),
    "Volkswagen Golf":    dict(seg="orta", w=0.9, base=88_000, dep=0.060, launch=2004,
                               engines=[(1197, "Benzin", 2013), (1395, "Benzin", 2013), (1498, "Benzin", 2020), (1598, "Dizel", 2013)],
                               trims=["Trendline", "Comfortline", "Highline"]),
    "Volkswagen Passat":  dict(seg="premium", w=0.8, base=135_000, dep=0.070, launch=2004,
                               engines=[(1395, "Benzin", 2013), (1498, "Benzin", 2019), (1598, "Dizel", 2013), (1968, "Dizel", 2013)],
                               trims=["Trendline", "Comfortline", "Highline"]),
    "Hyundai i20":        dict(seg="ekonomi", w=1.0, base=50_000, dep=0.055, launch=2009,
                               engines=[(1248, "Benzin", 2014), (998, "Benzin", 2020), (1396, "Dizel", 2014)],
                               trims=["Jump", "Style", "Elite"]),
    "Hyundai Tucson":     dict(seg="suv", w=0.5, base=115_000, dep=0.060, launch=2005,
                               engines=[(1591, "Benzin", 2015), (1598, "Hibrit", 2021), (1598, "Dizel", 2015), (1995, "Dizel", 2015)],
                               trims=["Comfort", "Elite", "Elite Plus"]),
    "Ford Focus":         dict(seg="orta", w=0.9, base=72_000, dep=0.065, launch=2004,
                               engines=[(1596, "Benzin", 2013), (999, "Benzin", 2015), (1499, "Dizel", 2015), (1560, "Dizel", 2013)],
                               trims=["Trend X", "Titanium", "ST-Line"]),
    "Opel Astra":         dict(seg="orta", w=0.6, base=70_000, dep=0.065, launch=2004,
                               engines=[(1399, "Benzin", 2013), (1199, "Benzin", 2020), (1598, "Dizel", 2013), (1496, "Dizel", 2020)],
                               trims=["Edition", "Dynamic", "Excellence"]),
    "Honda Civic":        dict(seg="orta", w=0.9, base=80_000, dep=0.045, launch=2004,
                               engines=[(1597, "Benzin", 2013), (1498, "Benzin", 2017), (1597, "Dizel", 2017), (1993, "Hibrit", 2022)],
                               trims=["Elegance", "Executive", "Executive+"]),
    "Dacia Duster":       dict(seg="suv", w=0.8, base=62_000, dep=0.050, launch=2010,
                               engines=[(1598, "Benzin", 2013), (1332, "Benzin", 2019), (1461, "Dizel", 2013)],
                               trims=["Ambiance", "Comfort", "Prestige"]),
    "Peugeot 3008":       dict(seg="suv", w=0.5, base=125_000, dep=0.060, launch=2010,
                               engines=[(1199, "Benzin", 2017), (1499, "Dizel", 2018), (1560, "Dizel", 2017)],
                               trims=["Active", "Allure", "GT"]),
    "Skoda Octavia":      dict(seg="orta", w=0.7, base=85_000, dep=0.060, launch=2004,
                               engines=[(1197, "Benzin", 2013), (999, "Benzin", 2018), (1498, "Benzin", 2019), (1598, "Dizel", 2013)],
                               trims=["Ambition", "Style", "Prestige"]),
    "BMW 3 Serisi":       dict(seg="premium", w=0.5, base=215_000, dep=0.075, launch=2004,
                               engines=[(1499, "Benzin", 2016), (1598, "Benzin", 2013), (1995, "Dizel", 2013), (1998, "Benzin", 2013)],
                               trims=["Standart", "Sport Line", "M Sport"]),
    "Mercedes C Serisi":  dict(seg="premium", w=0.5, base=240_000, dep=0.075, launch=2004,
                               engines=[(1497, "Benzin", 2018), (1595, "Benzin", 2014), (1496, "Benzin", 2021), (1598, "Dizel", 2014), (1950, "Dizel", 2018)],
                               trims=["Standart", "Avantgarde", "AMG"]),
}
HYBRID_MODELS = sorted({m for m, s in MODELS.items() for _, f, _ in s["engines"] if f == "Hibrit"})

# ----------------------------------------------------------------------------- planlanan ("gerçek") etkiler, log fiyat
# Referans: Benzin, Manuel, tramer kaydı yok, Sahibinden, Diğer şehir, Beyaz, temel donanım, orijinal kaporta
TRUE_EFFECTS = {
    "fuel_Dizel": 0.040,
    "fuel_Hibrit": 0.070,
    "fuel_LPG": -0.060,
    "gearbox_Otomatik": 0.080,
    "damage": -0.050,            # tramer (sigorta hasar) kaydı; boya/değişen parçalara EK olarak
    "seller_Galeri": 0.035,
    "city_İstanbul": 0.020,
    "city_Ankara": 0.010,
    "city_İzmir": 0.010,
    "km_10k": -0.006,
    "engine_100cc": 0.025,       # aynı model içinde +100 cm³
    "trim_Orta": 0.040,
    "trim_Üst": 0.090,
    "color_Gri": 0.005,
    "color_Siyah": -0.010,
    "color_Kırmızı": -0.030,
    "color_Mavi": -0.020,
    "color_Diğer": -0.040,
}
AGE_QUAD = 0.0015    # değer kaybı yaşlandıkça yavaşlar
NOISE_SD = 0.08
# 2021-06 .. 2022-12 çip krizi: 0-1 yaşındaki ikinci el araçlar sıfıra yakın, hatta üstünde fiyatlandı.
# Bu bir etkileşimdir ve OLS'te bilerek modellenmez (ağaç modellerinin yakalayabileceği yapı).
CHIP_PREMIUM = 0.06
CHIP_WINDOW = ("2021-06", "2022-12")

# Kaporta: bölge başına tek parça boyalı/değişen olmasının log fiyat cezası (pozitif = kayıp).
# Lokal boya, aynı bölgedeki "Boyalı" cezasının LOCAL_SHARE katı kabul edilir.
# Sıralama, ekspertiz pratiği ve hukuk sitelerindeki değer kaybı tablolarındaki aralıklarla (ör. kapı değişimi %5-12)
# uyumlu olacak şekilde seçildi; parça bazlı resmi bir yüzde tablosu yoktur.
BODY_PENALTY = {
    "tavan":         {"Boyalı": 0.060, "Değişen": 0.120},  # yapısal/devrilme şüphesi: en ağır
    "arka_camurluk": {"Boyalı": 0.030, "Değişen": 0.070},  # kaynaklı, gövdeye bağlı
    "kaput":         {"Boyalı": 0.025, "Değişen": 0.045},  # önden darbe şüphesi
    "kapi":          {"Boyalı": 0.015, "Değişen": 0.025},
    "bagaj":         {"Boyalı": 0.015, "Değişen": 0.030},
    "on_camurluk":   {"Boyalı": 0.012, "Değişen": 0.020},  # cıvatalı, değişimi kolay
    "tampon":        {"Boyalı": 0.003, "Değişen": 0.005},  # plastik, piyasada önemsenmez
}
LOCAL_SHARE = 0.4
# Bir "olay" (çizik, sürtme, kaza) hangi parçaya isabet eder: göreli ağırlıklar
PART_HIT_WEIGHT = {
    "on_tampon": 2.2, "arka_tampon": 2.0, "sol_on_kapi": 1.0, "sag_on_kapi": 1.0, "sol_arka_kapi": 0.9, "sag_arka_kapi": 0.9,
    "sol_on_camurluk": 1.1, "sag_on_camurluk": 1.1, "sol_arka_camurluk": 0.6, "sag_arka_camurluk": 0.6,
    "kaput": 0.8, "bagaj": 0.6, "tavan": 0.15,
}
# Parça türüne göre işlem olasılıkları: (lokal boya, boya, değişim)
STATE_PROBS = {
    "tampon": (0.25, 0.50, 0.25), "tavan": (0.35, 0.55, 0.10), "arka_camurluk": (0.40, 0.50, 0.10),
    "default": (0.30, 0.50, 0.20),
}

# Ortalama ilan fiyatını çapa seviyesine oturtan sabit ölçek (_calibrate() ile hesaplandı).
PRICE_SCALE = 0.913

CITIES = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Kocaeli", "Diğer"]
CITY_P = [0.27, 0.12, 0.09, 0.06, 0.05, 0.04, 0.04, 0.04, 0.29]
COLORS = ["Beyaz", "Gri", "Siyah", "Kırmızı", "Mavi", "Diğer"]
# renk payı, model yılına göre: 2010 ve 2026 arasında doğrusal geçiş (beyazdan griye kayış)
COLOR_P_OLD = np.array([0.56, 0.16, 0.13, 0.05, 0.05, 0.05])
COLOR_P_NEW = np.array([0.28, 0.40, 0.14, 0.06, 0.06, 0.06])


def _color_probs(model_year: np.ndarray) -> np.ndarray:
    t = np.clip((model_year - 2010) / 16, 0, 1)[:, None]
    return (1 - t) * COLOR_P_OLD + t * COLOR_P_NEW


def _choose_rows(rng, probs: np.ndarray) -> np.ndarray:
    """Her satır için kendi olasılık vektöründen bir indeks seçer (vektörize)."""
    cum = probs.cumsum(axis=1)
    cum /= cum[:, -1:]
    return (rng.random((len(probs), 1)) > cum).sum(axis=1)


def body_penalty_log(states: pd.DataFrame) -> np.ndarray:
    """Parça durumlarından (BODY_PARTS sütunları) toplam log fiyat cezasını hesaplar (pozitif sayı)."""
    total = np.zeros(len(states))
    for part in BODY_PARTS:
        pen = BODY_PENALTY[PART_REGION[part]]
        s = states[part].to_numpy()
        total += np.where(s == "Boyalı", pen["Boyalı"], 0.0)
        total += np.where(s == "Değişen", pen["Değişen"], 0.0)
        total += np.where(s == "Lokal boyalı", LOCAL_SHARE * pen["Boyalı"], 0.0)
    return total


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(MODELS)
    w = np.array([MODELS[m]["w"] for m in names])
    model = rng.choice(names, n, p=w / w.sum())

    month_idx = rng.integers(0, len(MONTHS), n)
    listing_year = np.array([int(MONTHS[i][:4]) for i in month_idx])

    # yaş: ilan sitelerinde ortalama ~7-8 yaş; model yılı, model adının satışa girdiği yıldan önce olamaz
    launch = np.array([MODELS[m]["launch"] for m in model])
    age = np.clip(rng.gamma(2.3, 3.4, n).round(), 0, 20).astype(int)
    for _ in range(20):  # uygunsuz yaşları yeniden çek
        bad = listing_year - age < launch
        if not bad.any():
            break
        age[bad] = np.clip(rng.gamma(2.3, 3.4, bad.sum()).round(), 0, 20).astype(int)
    age = np.minimum(age, listing_year - launch)
    year = listing_year - age

    # motor + yakıt: o model yılında satılan seçeneklerden; dizel payı zamanla düşer, hibrit payı artar
    engine_cc = np.zeros(n, dtype=int)
    fuel = np.empty(n, dtype=object)
    for i in range(n):
        engines = MODELS[model[i]]["engines"]
        first = min(y0 for _, _, y0 in engines)
        # listede ilk motor yılından eski araçlar (önceki kasa) ilk dönem motorlarıyla temsil edilir
        if year[i] >= first:
            opts = [(cc, f) for cc, f, y0 in engines if y0 <= year[i]]
        else:
            opts = [(cc, f) for cc, f, y0 in engines if y0 == first and f != "Hibrit"]
        diesel_w = np.interp(year[i], [2008, 2016, 2026], [1.6, 1.3, 0.55])
        hybrid_w = 1.6 if year[i] >= 2023 else 1.0
        ww = np.array([diesel_w if f == "Dizel" else hybrid_w if f == "Hibrit" else 1.0 for _, f in opts])
        cc, f = opts[rng.choice(len(opts), p=ww / ww.sum())]
        engine_cc[i], fuel[i] = cc, f
    # LPG: benzinli araçlara sonradan dönüşüm; eski ve ekonomi segmentinde daha yaygın, premium/hibritte yok
    seg = np.array([MODELS[m]["seg"] for m in model])
    p_lpg = np.where(seg == "premium", 0.0, 0.03 + 0.012 * age) * (fuel == "Benzin")
    fuel = np.where(rng.random(n) < p_lpg, "LPG", fuel).astype(object)

    p_auto = np.select([seg == "premium", seg == "suv"], [0.92, 0.70], 0.30) + 0.025 * np.clip(year - 2016, -6, 10)
    gearbox = np.where((fuel == "Hibrit") | (rng.random(n) < np.clip(p_auto, 0.05, 0.98)), "Otomatik", "Manuel")

    trim_level = _choose_rows(rng, np.where((seg == "premium")[:, None], [0.25, 0.40, 0.35], [0.35, 0.45, 0.20]))
    trim = np.array([MODELS[m]["trims"][t] for m, t in zip(model, trim_level)])
    trim_class = np.array(["Temel", "Orta", "Üst"])[trim_level]

    color = np.array(COLORS)[_choose_rows(rng, _color_probs(year))]

    # km: Türkiye'de yıllık ~15 bin km; dizel araçlar daha çok yol yapar
    km_per_year = np.minimum(rng.lognormal(np.log(15_000), 0.38, n) * np.where(fuel == "Dizel", 1.35, 1.0), 45_000)
    km = np.maximum(50, (age + 0.5) * km_per_year + rng.normal(0, 1500, n)).round(-2).astype(int)

    # kaporta: yaşla ve km ile artan olay sayısı; her olay bir parçaya isabet eder
    lam = 0.15 + 0.17 * age + 0.012 * km / 10_000
    n_events = np.minimum(rng.poisson(lam), 8)
    part_w = np.array([PART_HIT_WEIGHT[p] for p in BODY_PARTS])
    states = {p: np.full(n, "Orijinal", dtype=object) for p in BODY_PARTS}
    severity = {"Orijinal": 0, "Lokal boyalı": 1, "Boyalı": 2, "Değişen": 3}
    for i in np.nonzero(n_events)[0]:
        for p_idx in rng.choice(len(BODY_PARTS), n_events[i], p=part_w / part_w.sum()):
            part = BODY_PARTS[p_idx]
            region = PART_REGION[part]
            probs = STATE_PROBS.get(region, STATE_PROBS["default"])
            new = ["Lokal boyalı", "Boyalı", "Değişen"][rng.choice(3, p=probs)]
            if severity[new] > severity[states[part][i]]:
                states[part][i] = new
    states = pd.DataFrame(states)
    n_painted = states.isin(["Boyalı", "Lokal boyalı"]).sum(axis=1).to_numpy()
    n_replaced = (states == "Değişen").sum(axis=1).to_numpy()
    serious = (states.drop(columns=["on_tampon", "arka_tampon"]).isin(["Boyalı", "Değişen"])).sum(axis=1).to_numpy()
    # tramer kaydı: ciddi kaporta işlemi varsa çoğunlukla sigortadan geçmiştir; temiz araçta da (cam, mekanik) olabilir
    damage = (rng.random(n) < np.where(serious > 0, 0.55 + 0.08 * np.minimum(serious, 4), 0.08)).astype(int)

    seller = np.where(rng.random(n) < 0.45, "Galeri", "Sahibinden")
    city = rng.choice(CITIES, n, p=CITY_P)

    # ---- fiyat
    idx = market_index().to_numpy()[month_idx]
    base = np.array([MODELS[m]["base"] for m in model], dtype=float)
    dep = np.array([MODELS[m]["dep"] for m in model])
    min_cc = np.array([min(cc for cc, _, _ in MODELS[m]["engines"]) for m in model])
    in_chip = (np.array([MONTHS[i] for i in month_idx]) >= CHIP_WINDOW[0]) & (np.array([MONTHS[i] for i in month_idx]) <= CHIP_WINDOW[1])
    eff = TRUE_EFFECTS
    body_pen = body_penalty_log(states)
    log_price = (
        np.log(base * idx)
        + np.log(PRICE_SCALE)
        - dep * age
        + AGE_QUAD * age**2
        + eff["km_10k"] * km / 10_000
        + eff["engine_100cc"] * (engine_cc - min_cc) / 100
        + np.array([eff.get(f"fuel_{f}", 0.0) for f in fuel])
        + np.where(gearbox == "Otomatik", eff["gearbox_Otomatik"], 0.0)
        + np.array([eff.get(f"trim_{t}", 0.0) for t in trim_class])
        + np.array([eff.get(f"color_{c}", 0.0) for c in color])
        + damage * eff["damage"]
        - body_pen
        + np.where(seller == "Galeri", eff["seller_Galeri"], 0.0)
        + np.array([eff.get(f"city_{c}", 0.0) for c in city])
        + CHIP_PREMIUM * (in_chip & (age <= 1))
        + rng.normal(0, NOISE_SD, n)
    )
    price = np.maximum(15_000, np.exp(log_price)).round(-3).astype(int)

    df = pd.DataFrame(
        {
            "listing_id": np.arange(1, n + 1),
            "listing_month": [MONTHS[i] for i in month_idx],
            "model": model,
            "segment": seg,
            "year": year,
            "km": km,
            "fuel": fuel,
            "engine_cc": engine_cc,
            "gearbox": gearbox,
            "trim": trim,
            "trim_level": trim_class,
            "color": color,
            "damage": damage,
            "seller_type": seller,
            "city": city,
        }
    )
    df = pd.concat([df, states], axis=1)
    df["n_painted"] = n_painted
    df["n_replaced"] = n_replaced
    # Bilgi amaçlı "gerçek" kaporta kaybı (%): yalnızca sentetik veride bilinir, modele GİRMEZ.
    df["body_loss_pct_true"] = np.round(-np.expm1(-body_pen) * 100, 2)
    df["price"] = price
    return df


def _calibrate(n: int = N, seed: int = SEED) -> float:
    """PRICE_SCALE'i, 2023-11 sonrası (kaynaklı çapaların olduğu dönem) üretilen ortalama fiyat çapa seviyesine
    eşit olacak şekilde hesaplar. Model/dağılım parametreleri değişirse çalıştırıp PRICE_SCALE'i güncelleyin:
        python -c "import generate_data as g; print(g._calibrate())"
    """
    global PRICE_SCALE
    old, PRICE_SCALE = PRICE_SCALE, 1.0
    try:
        df = generate(n, seed)
    finally:
        PRICE_SCALE = old
    recent = df["listing_month"] >= "2023-11"
    target = (market_index()[df.loc[recent, "listing_month"]].to_numpy() * PRICE_ANCHORS["2016-01"]).mean()
    return float(target / df.loc[recent, "price"].mean())


def main() -> None:
    df = generate()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "listings.csv", index=False)
    by_year = df.groupby(df.listing_month.str[:4]).price.mean().round(-3).astype(int)
    print(f"{len(df):,} ilan yazıldı -> data/listings.csv")
    print("Yıllara göre ortalama ilan fiyatı (TL):")
    print(by_year.to_string())


if __name__ == "__main__":
    main()
