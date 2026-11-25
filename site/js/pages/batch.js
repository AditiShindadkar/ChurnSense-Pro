import { $, kpis, table, pct, num, money, escapeHtml, download, toCsv, parseCsv, riskBand } from "../ui.js";
import { encode, predict, groupedShap } from "../model.js";
import { bestOffer, lifetimeValue } from "../business.js";

const MAX_ROWS = 50000;
let scored = null;

const columnsOf = (schema) => [...schema.numeric, ...Object.keys(schema.categorical)];

export function init(ctx) {
  const cols = columnsOf(ctx.model.schema);
  $("#batch-columns").textContent = cols.join(", ");
  const input = $("#batch-file");
  input.addEventListener("change", () => input.files[0] && readFile(ctx, input.files[0]));

  const zone = $("#dropzone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) readFile(ctx, f);
  });

  $("#batch-template").addEventListener("click", () => {
    const ex = ctx.sample.slice(0, 3).map((c) => cols.map((k) => c.raw[k]));
    download("churnsense-template.csv", toCsv(["CustomerID", ...cols], ex.map((r, i) => [`C${i + 1}`, ...r])));
  });
  $("#batch-demo").addEventListener("click", () => {
    const rows = ctx.sample.slice(0, 200).map((c, i) => [`DEMO-${i + 1}`, ...cols.map((k) => c.raw[k])]);
    run(ctx, [["CustomerID", ...cols], ...rows.map((r) => r.map(String))], "200 sample customers");
  });
  $("#batch-download").addEventListener("click", () => scored && exportScored(ctx));
}

export function render() {}

function status(msg, error = false) {
  const el = $("#batch-status");
  el.textContent = msg;
  el.classList.toggle("error", error);
}

function readFile(ctx, file) {
  if (file.size > 25 * 1024 * 1024) return status("File is larger than 25 MB. Please split it.", true);
  const reader = new FileReader();
  reader.onload = () => run(ctx, parseCsv(String(reader.result)), file.name);
  reader.onerror = () => status("Could not read that file.", true);
  reader.readAsText(file);
}

function run(ctx, rows, label) {
  const { model, economics } = ctx;
  if (rows.length < 2) return status("The file needs a header row and at least one customer.", true);
  const header = rows[0].map((h) => h.trim());
  const need = columnsOf(model.schema);
  const missing = need.filter((c) => !header.includes(c));
  if (missing.length) return status(`Missing columns: ${missing.join(", ")}. Download the template to see the expected format.`, true);
  const idCol = header.findIndex((h) => /^(customer ?id|id)$/i.test(h));

  const body = rows.slice(1, MAX_ROWS + 1);
  const results = [];
  const errors = [];
  body.forEach((r, i) => {
    const raw = {};
    header.forEach((h, j) => (raw[h] = r[j]?.trim()));
    try {
      const x = encode(model.schema, raw);
      const p = predict(model, x);
      const driver = groupedShap(model, x).items.find((it) => it.value > 0);
      const value = lifetimeValue(Number(raw["Total Spend"]), economics);
      const best = bestOffer(p, value, economics.offers, economics.contactCost);
      results.push({ id: idCol >= 0 ? r[idCol] : String(i + 1), raw, p, driver: driver ? `${driver.feature} = ${raw[driver.feature]}` : "–", value, best });
    } catch (err) {
      errors.push(`Row ${i + 2}: ${err.message}`);
    }
  });

  scored = results;
  const note = rows.length - 1 > MAX_ROWS ? ` Only the first ${num(MAX_ROWS)} rows were scored.` : "";
  const errText = errors.length ? ` ${errors.length} row(s) skipped, e.g. ${errors[0]}.` : "";
  status(`Scored ${num(results.length)} customers from ${label}.${note}${errText}`, results.length === 0);
  if (!results.length) return;

  const high = results.filter((r) => r.p >= 0.7).length;
  const targeted = results.filter((r) => r.best.offer >= 0);
  kpis($("#batch-kpis"), [
    { label: "Customers", value: num(results.length) },
    { label: "High risk (≥70%)", value: num(high), sub: pct(high / results.length) },
    { label: "Recommended to contact", value: num(targeted.length) },
    { label: "Total expected value", value: money(targeted.reduce((s, r) => s + r.best.ev, 0)) },
  ]);

  const view = [...results].sort((a, b) => b.best.ev - a.best.ev || b.p - a.p).slice(0, 50);
  table($("#batch-table"), [
    { label: "Customer", render: (r) => escapeHtml(r.id) },
    { label: "Risk", num: true, render: (r) => `<span class="pill ${riskBand(r.p).cls}">${pct(r.p, 1)}</span>` },
    { label: "Main driver", render: (r) => escapeHtml(r.driver) },
    { label: "Best offer", render: (r) => escapeHtml(r.best.offer >= 0 ? economics.offers[r.best.offer].name : "No offer") },
    { label: "Expected value", num: true, render: (r) => money(r.best.ev) },
  ], view);
  $("#batch-results").hidden = false;
}

function exportScored(ctx) {
  const cols = columnsOf(ctx.model.schema);
  const offers = ctx.economics.offers;
  download(
    "churnsense-scored.csv",
    toCsv(
      ["CustomerID", ...cols, "churn_risk", "risk_band", "main_driver", "customer_value", "recommended_offer", "expected_value"],
      scored.map((r) => [r.id, ...cols.map((c) => r.raw[c]), r.p.toFixed(4), riskBand(r.p).label, r.driver, r.value.toFixed(2),
        r.best.offer >= 0 ? offers[r.best.offer].name : "No offer", r.best.ev.toFixed(2)]),
    ),
  );
}
