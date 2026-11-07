/* JavaScript tahmin formülünü Python'un altın örnekleriyle karşılaştırır: node app/check_golden.js */
const fs = require("fs");
const path = require("path");
const E = require("./estimator.js");

const model = JSON.parse(fs.readFileSync(path.join(__dirname, "model.json"), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "golden_cases.json"), "utf8"));

let worst = 0;
for (const c of cases) {
  const z = E.predictLog(model, c.car);
  worst = Math.max(worst, Math.abs(z - c.expected_log));
  const est = E.estimate(model, c.car);
  if (Math.abs(est.final - c.expected_price) > 1e-6 * c.expected_price) {
    console.error("estimate().final uyuşmuyor", est.final, c.expected_price);
    process.exit(1);
  }
  if (!(est.low < est.final && est.final < est.high)) {
    console.error("aralık tutarsız", est);
    process.exit(1);
  }
  // hesap dökümü toplamı: hatasız + boya kaybı + tramer kaybı = tahmin
  const sum = est.clean + est.bodyLoss + est.damageLoss;
  if (Math.abs(sum - est.final) > 1e-6 * est.final) {
    console.error("döküm toplamı tutmuyor", sum, est.final);
    process.exit(1);
  }
}
if (worst > 1e-9) {
  console.error("Python ile fark büyük:", worst);
  process.exit(1);
}
console.log(`OK: ${cases.length} örnek, en büyük log farkı ${worst.toExponential(2)}`);
