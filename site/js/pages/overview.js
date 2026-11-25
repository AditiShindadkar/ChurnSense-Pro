import { $, kpis, table, pct, num, chart, cssVar } from "../ui.js";

export function init() {}

export function render({ insights }) {
  const ms = insights.models;
  const rule = ms.find((m) => m.model.startsWith("5-rule"));
  const deployed = ms.find((m) => m.model === insights.deployed_model);
  const d = insights.dataset;

  kpis($("#overview-kpis"), [
    { label: "Customers analysed", value: num(d.rows), sub: `${num(d.test_rows)} held out for testing` },
    { label: "Churn rate", value: pct(d.churn_rate), sub: "share of customers who left" },
    { label: "Model accuracy", value: pct(deployed.accuracy, 2), sub: `${deployed.model}, held-out test set` },
    { label: "Errors vs rules", value: `${num(deployed.errors)} vs ${num(rule.errors)}`, sub: `${pct(1 - deployed.errors / rule.errors, 1)} fewer mistakes than no-ML rules` },
  ]);

  $("#rule-acc").textContent = pct(rule.accuracy, 1);
  $("#rule-err").textContent = num(rule.errors);
  $("#model-err").textContent = num(deployed.errors);

  table($("#rules-table"), [
    { label: "Rule", key: "rule" },
    { label: "Customers flagged", num: true, render: (r) => pct(r.share_of_customers) },
    { label: "Churn if true", num: true, render: (r) => `<span class="pill pill-risk">${pct(r.churn_rate_when_true)}</span>` },
    { label: "Churn if false", num: true, render: (r) => pct(r.churn_rate_when_false) },
  ], insights.rules, { highlight: (r) => r.rule === "Any rule" });

  const shap = insights.shap.by_feature;
  chart("overview-shap", {
    type: "bar",
    data: {
      labels: shap.map((s) => s.feature),
      datasets: [{ data: shap.map((s) => s.value), backgroundColor: cssVar("--chart-1"), borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "mean |SHAP| (log-odds)" } }, y: { grid: { display: false } } },
    },
  });
}
