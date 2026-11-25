// Checks the browser model against Python (LightGBM predict_proba and pred_contrib).
// Run: node --test tests_js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { encode, predict, rawScore, shap, groupedShap, whatIf, featureNames } from "../site/js/model.js";
import { bestOffer, planWithBudget, realisedProfit, DEFAULT_OFFERS } from "../site/js/business.js";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const model = load("../site/data/model.json");
const parity = load("../tests/fixtures/parity.json");

test("feature order matches the exported model", () => {
  assert.deepEqual(featureNames(model.schema), model.features);
});

test("probabilities match LightGBM", () => {
  parity.rows.forEach((row, i) => {
    const p = predict(model, encode(model.schema, row));
    assert.ok(Math.abs(p - parity.prob[i]) < 1e-9, `row ${i}: ${p} vs ${parity.prob[i]}`);
  });
});

test("TreeSHAP values match LightGBM pred_contrib", () => {
  parity.rows.forEach((row, i) => {
    const x = encode(model.schema, row);
    const { base, phi } = shap(model, x);
    assert.ok(Math.abs(base - parity.expected_value[i]) < 1e-6, `base ${base} vs ${parity.expected_value[i]}`);
    phi.forEach((v, j) => assert.ok(Math.abs(v - parity.contrib[i][j]) < 1e-6, `row ${i} feature ${j}: ${v} vs ${parity.contrib[i][j]}`));
  });
});

test("SHAP values add up to the model score", () => {
  const x = encode(model.schema, parity.rows[0]);
  const { base, items } = groupedShap(model, x);
  const sum = items.reduce((s, it) => s + it.value, base);
  assert.ok(Math.abs(sum - rawScore(model, x)) < 1e-6);
});

test("encode rejects bad input", () => {
  const row = { ...parity.rows[0], "Contract Length": "Weekly" };
  assert.throws(() => encode(model.schema, row), /Contract Length/);
  assert.throws(() => encode(model.schema, { ...parity.rows[0], Age: "abc" }), /Age/);
});

test("what-if suggestions never include a no-op", () => {
  const row = { ...parity.rows[0], "Contract Length": "Annual" };
  const { actions } = whatIf(model, row);
  assert.ok(actions.every((a) => !a.label.includes("Annual")));
});

test("business maths matches the Python worked example", () => {
  // p=0.8, value=1000: email 0.8*0.05*1000-2=38, discount 200-30=170, call 320-60=260
  const best = bestOffer(0.8, 1000, DEFAULT_OFFERS);
  assert.equal(best.offer, 2);
  assert.ok(Math.abs(best.ev - 260) < 1e-9);
  assert.equal(bestOffer(0.01, 100, DEFAULT_OFFERS).offer, -1);
  const customers = [
    { prob: 0.9, value: 1000, churn: 1 },
    { prob: 0.9, value: 1000, churn: 0 },
    { prob: 0.0, value: 1000, churn: 0 },
  ];
  const { chosen, spent } = planWithBudget(customers, DEFAULT_OFFERS, { budget: 60 });
  assert.deepEqual(chosen, [2, -1, -1]);
  assert.equal(spent, 60);
  assert.equal(realisedProfit(customers, chosen, DEFAULT_OFFERS).profit, 400 - 60);
});
