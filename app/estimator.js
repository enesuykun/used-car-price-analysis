/* Araç değer tahmini hesabı: saf fonksiyonlar (tarayıcıda ve Node'da çalışır).
 * Model: log(fiyat) = sabit + dönem düzeyi + Σ katsayı x özellik (analysis.build_features ile aynı özellikler).
 * app/export_model.py içindeki predict_log bunun Python karşılığıdır; tests/test_app.py ikisini karşılaştırır. */
(function (root) {
  "use strict";

  const STATES = ["Orijinal", "Lokal boyalı", "Boyalı", "Değişen"];
  const PARTS = ["kaput", "tavan", "bagaj", "sol_on_camurluk", "sag_on_camurluk", "sol_on_kapi", "sag_on_kapi",
    "sol_arka_kapi", "sag_arka_kapi", "sol_arka_camurluk", "sag_arka_camurluk", "on_tampon", "arka_tampon"];

  function cleanParts() {
    const p = {};
    PARTS.forEach((k) => { p[k] = "Orijinal"; });
    return p;
  }

  function predictLog(model, car, opts) {
    opts = opts || {};
    const c = model.coef, meta = model.meta;
    const age = Math.max(0, meta.ref_year - car.year);
    let z = c.intercept + meta.period_level;
    z += (c.age || 0) * age + (c.age_sq || 0) * age * age + (c.km_10k || 0) * car.km / 10000;
    z += (c.engine_100cc || 0) * car.engine_cc / 100;
    if (!opts.skipDamage) z += (c.damage || 0) * car.damage;
    [["fuel", car.fuel], ["gearbox", car.gearbox], ["seller", car.seller_type], ["city", car.city],
      ["color", car.color], ["trim", car.trim_level]].forEach(function (pair) {
      z += c[pair[0] + "_" + pair[1]] || 0;
    });
    if (meta.models_with_dummy.indexOf(car.model) >= 0) {
      z += c["model_" + car.model] + (c["agex_" + car.model] || 0) * age;
    }
    if (!opts.skipBody) {
      Object.keys(car.parts).forEach(function (part) {
        const state = car.parts[part];
        if (state !== "Orijinal") z += c["body_" + model.part_region[part] + "_" + model.state_keys[state]] || 0;
      });
    }
    return z;
  }

  /* Hesap dökümü: hatasız fiyat -> boya/değişen kaybı -> tramer kaybı -> tahmin, ve %80'lik aralık. */
  function estimate(model, car) {
    const zFull = predictLog(model, car);
    const zClean = predictLog(model, car, { skipBody: true, skipDamage: true });
    const zBody = predictLog(model, car, { skipDamage: true });
    const clean = Math.exp(zClean), afterBody = Math.exp(zBody), final = Math.exp(zFull);
    const parts = [];
    Object.keys(car.parts).forEach(function (part) {
      const state = car.parts[part];
      if (state === "Orijinal") return;
      const key = "body_" + model.part_region[part] + "_" + model.state_keys[state];
      const logEffect = model.coef[key] || 0;
      parts.push({ part: part, region: model.part_region[part], state: state, pct: Math.expm1(logEffect), tl: clean * Math.expm1(logEffect) });
    });
    parts.sort(function (a, b) { return a.pct - b.pct; });
    return {
      clean: clean, afterBody: afterBody, final: final,
      bodyLoss: afterBody - clean, damageLoss: final - afterBody,
      low: Math.exp(zFull + model.meta.resid_p10), high: Math.exp(zFull + model.meta.resid_p90),
      parts: parts,
    };
  }

  /* O model yılında satılan motorlar (generate_data.py ile aynı kural): ilk motor yılından eski araçlar
   * ilk dönem motorlarıyla temsil edilir ve hibrit içermez. */
  function enginesFor(model, name, year) {
    const engines = model.catalog[name].engines;
    const first = Math.min.apply(null, engines.map(function (e) { return e[2]; }));
    const opts = year >= first
      ? engines.filter(function (e) { return e[2] <= year; })
      : engines.filter(function (e) { return e[2] === first && e[1] !== "Hibrit"; });
    const seen = {};
    return opts.filter(function (e) {
      const k = e[0] + "|" + e[1];
      if (seen[k]) return false;
      seen[k] = true;
      return true;
    }).map(function (e) { return { cc: e[0], fuel: e[1] }; });
  }

  function yearRange(model, name) {
    const hi = model.meta.ref_year;
    return { min: Math.max(model.catalog[name].launch, hi - model.max_age), max: hi };
  }

  /* Kilometre yaşa göre alışılmadıksa uyarı: "low" | "high" | null */
  function kmCheck(model, car) {
    const age = Math.max(0, model.meta.ref_year - car.year);
    if (age < 1) return null;
    const perYear = car.km / (age + 0.5);
    if (perYear < model.meta.km_per_year_p05) return "low";
    if (perYear > model.meta.km_per_year_p95) return "high";
    return null;
  }

  const api = { STATES: STATES, PARTS: PARTS, cleanParts: cleanParts, predictLog: predictLog, estimate: estimate,
    enginesFor: enginesFor, yearRange: yearRange, kmCheck: kmCheck };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Estimator = api;
})(typeof window !== "undefined" ? window : globalThis);
