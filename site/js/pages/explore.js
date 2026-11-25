import { $, chart, cssVar, pct } from "../ui.js";

const NOTES = {
  "Contract Length": "Every Monthly-contract customer churned. This is one of the dataset’s hard-coded rules.",
  "Support Calls": "Churn jumps to 100% above 5 support calls, another generated cut-off.",
  "Payment Delay": "Flat until about 20 days, then everyone churns.",
  "Total Spend": "Everyone spending under about $500 churned.",
  Age: "Over-50s all churned. Below that, churn falls slowly with age.",
  "Last Interaction": "Customers silent for more than about 2 weeks churn noticeably more.",
  Gender: "Women churn more often in this dataset. A real deployment should check this kind of attribute for fairness before acting on it.",
};

export function init({ insights }) {
  const select = $("#explore-feature");
  const features = [...Object.keys(insights.eda)];
  select.innerHTML = features.map((f) => `<option>${f}</option>`).join("");
  select.value = "Support Calls";
  select.addEventListener("change", () => render({ insights }));
}

export function render({ insights }) {
  const f = $("#explore-feature").value;
  const eda = insights.eda[f];
  const pdp = insights.pdp[f];
  const accent = cssVar("--chart-1");
  const risk = cssVar("--risk");

  chart("explore-eda", {
    type: "bar",
    data: {
      labels: eda.map((b) => b.label),
      datasets: [{
        label: "Churn rate",
        data: eda.map((b) => b.churn_rate),
        backgroundColor: eda.map((b) => (b.churn_rate > 0.99 ? risk : accent)),
        borderRadius: 4,
      }],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `${pct(c.raw)} churn · ${eda[c.dataIndex].count.toLocaleString()} customers` } },
      },
      scales: { y: { min: 0, max: 1, ticks: { callback: (v) => pct(v, 0) } }, x: { grid: { display: false } } },
    },
  });

  const categorical = typeof pdp.x[0] === "string";
  chart("explore-pdp", {
    type: categorical ? "bar" : "line",
    data: {
      labels: pdp.x,
      datasets: [{
        label: "Average predicted risk",
        data: pdp.y,
        borderColor: cssVar("--chart-2"),
        backgroundColor: categorical ? cssVar("--chart-2") : "transparent",
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.2,
        borderRadius: 4,
      }],
    },
    options: {
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${pct(c.raw)} average risk` } } },
      scales: {
        y: { min: 0, max: 1, ticks: { callback: (v) => pct(v, 0) } },
        x: { title: { display: !categorical, text: f }, grid: { display: false } },
      },
    },
  });

  $("#explore-note").textContent = NOTES[f] || "";
}
