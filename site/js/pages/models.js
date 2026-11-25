import { $, table, pct, num, chart, palette, cssVar } from "../ui.js";

const f = (x, d = 4) => Number(x).toFixed(d);

export function init({ insights }) {
  const sel = $("#cm-model");
  sel.innerHTML = insights.models.map((m) => `<option>${m.model}</option>`).join("");
  sel.value = insights.deployed_model;
  sel.addEventListener("change", () => confusion(insights));
}

export function render({ insights }) {
  const ms = insights.models;
  const best = ms.reduce((a, b) => (b.errors < a.errors ? b : a));
  const deployed = ms.find((m) => m.model === insights.deployed_model);
  $("#models-lead").textContent =
    `${ms.length} approaches on ${num(insights.dataset.test_rows)} held-out customers. ` +
    `${best.model} makes the fewest mistakes (${best.errors}). The site runs ${deployed.model} (${deployed.errors} mistakes) ` +
    `because its trees export to compact JSON and its in-browser SHAP values are verified against Python. ` +
    `The difference is ${pct((deployed.errors - best.errors) / insights.dataset.test_rows, 3)} of customers.`;

  table($("#models-table"), [
    { label: "Model", render: (r) => r.model + (r.model === insights.deployed_model ? ' <span class="pill pill-accent">deployed</span>' : "") },
    { label: "Accuracy", num: true, render: (r) => pct(r.accuracy, 2) },
    { label: "Precision", num: true, render: (r) => f(r.precision) },
    { label: "Recall", num: true, render: (r) => f(r.recall) },
    { label: "F1", num: true, render: (r) => f(r.f1) },
    { label: "ROC-AUC", num: true, render: (r) => f(r.roc_auc, 5) },
    { label: "Brier", num: true, render: (r) => f(r.brier, 5) },
    { label: "Errors", num: true, render: (r) => num(r.errors) },
    { label: "Train time", num: true, render: (r) => (r.train_seconds ? `${r.train_seconds}s` : "–") },
  ], ms, { highlight: (r) => r.model === insights.deployed_model });

  const cv = insights.cv;
  $("#cv-note").textContent = cv
    ? `5-fold cross-validated ROC-AUC for LightGBM on the training set: ${f(cv.mean, 6)} ± ${f(cv.std, 6)}. The result is stable, not a lucky split.`
    : "";

  const colors = palette();
  chart("models-roc", {
    type: "line",
    data: {
      datasets: Object.entries(insights.curves).map(([name, c], i) => ({
        label: name,
        data: c.roc.map(([x, y]) => ({ x, y })),
        borderColor: colors[i % colors.length],
        borderWidth: name === insights.deployed_model ? 3 : 1.5,
        pointRadius: 0,
        fill: false,
      })),
    },
    options: {
      parsing: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" } },
        y: { min: 0, max: 1, title: { display: true, text: "True positive rate" } },
      },
    },
  });

  confusion(insights);
  smote(insights);
  ablation(insights);

  const cal = insights.calibration;
  const total = cal.reduce((s, b) => s + b.count, 0);
  const extremes = cal.filter((b) => b.predicted < 0.1 || b.predicted > 0.9).reduce((s, b) => s + b.count, 0);
  const middle = total - extremes;
  $("#calibration-note").textContent =
    `${pct(extremes / total, 2)} of predictions are below 10% or above 90%, and they land on the diagonal. ` +
    `The middle bins hold only ${num(middle)} customers between them, so their zig-zag is noise from tiny samples, not miscalibration.`;
  chart("models-calibration", {
    type: "line",
    data: {
      datasets: [
        { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: cssVar("--chart-5"), borderDash: [5, 5], pointRadius: 0, borderWidth: 1.5 },
        { label: insights.deployed_model, data: cal.map((b) => ({ x: b.predicted, y: b.observed, n: b.count })), borderColor: colors[0], backgroundColor: colors[0], pointRadius: 4, borderWidth: 2 },
        { label: "With SMOTE", data: insights.smote.with_smote.calibration.map((b) => ({ x: b.predicted, y: b.observed })), borderColor: colors[2], backgroundColor: colors[2], pointRadius: 3, borderWidth: 1.5 },
      ],
    },
    options: {
      parsing: false,
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: predicted ${pct(c.raw.x)}, observed ${pct(c.raw.y)}${c.raw.n ? ` (${num(c.raw.n)} customers)` : ""}` } } },
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Predicted churn probability" } },
        y: { min: 0, max: 1, title: { display: true, text: "Observed churn rate" } },
      },
    },
  });
}

function confusion(insights) {
  const name = $("#cm-model").value;
  const m = insights.models.find((x) => x.model === name);
  const { tn, fp, fn, tp } = m.confusion;
  $("#cm-title").textContent = name;
  $("#models-cm").innerHTML = `
    <div></div><div class="cm-head">Predicted stay</div><div class="cm-head">Predicted churn</div>
    <div class="cm-side">Stayed</div><div class="cell good">${num(tn)}<small>correct</small></div><div class="cell bad">${num(fp)}<small>wasted offers</small></div>
    <div class="cm-side">Churned</div><div class="cell bad">${num(fn)}<small>missed churners</small></div><div class="cell good">${num(tp)}<small>caught</small></div>`;
}

function smote({ smote: s }) {
  $("#smote-text").textContent =
    `The training data is ${pct(s.train_churn_rate)} churners vs ${pct(1 - s.train_churn_rate)} stayers. That is mildly unbalanced, not rare-event imbalance. ` +
    `SMOTE (used in both original notebooks) invents synthetic stayers to force 50/50. Ranking quality barely changes, but probabilities ` +
    `get slightly worse (higher Brier score and log-loss). The profit maths multiplies by those probabilities, so SMOTE is dropped.`;
  const rows = [
    { name: "Without SMOTE (deployed)", ...s.without_smote },
    { name: "With SMOTE", ...s.with_smote },
  ];
  table($("#smote-table"), [
    { label: "Setup", key: "name" },
    { label: "Accuracy", num: true, render: (r) => pct(r.accuracy, 3) },
    { label: "ROC-AUC", num: true, render: (r) => f(r.roc_auc, 6) },
    { label: "Brier", num: true, render: (r) => f(r.brier, 6) },
    { label: "Log-loss", num: true, render: (r) => f(r.log_loss, 5) },
    { label: "Errors", num: true, render: (r) => num(r.errors) },
  ], rows, { highlight: (_, i) => i === 0 });
}

function ablation({ ablation: rows }) {
  table($("#ablation-table"), [
    { label: "Model", key: "model" },
    { label: "Features", key: "features" },
    { label: "Accuracy", num: true, render: (r) => pct(r.accuracy, 2) },
    { label: "ROC-AUC", num: true, render: (r) => f(r.roc_auc, 5) },
    { label: "Errors", num: true, render: (r) => num(r.errors) },
  ], rows);
  const lr = rows.filter((r) => r.model === "Logistic Regression");
  const gb = rows.filter((r) => r.model === "LightGBM");
  $("#ablation-note").textContent =
    `Engineered features cut Logistic Regression errors by ${num(lr[0].errors - lr[1].errors)} (linear models can't build interactions themselves). ` +
    `For LightGBM the change is ${num(gb[0].errors - gb[1].errors)}, because trees already learn those interactions. ` +
    `The deployed model therefore uses raw features only, which keeps explanations readable.`;
}
