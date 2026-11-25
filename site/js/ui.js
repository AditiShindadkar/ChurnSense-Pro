// Small shared helpers: formatting, DOM, tables, charts, CSV.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
export const num = (x, d = 0) => Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
export const money = (x) => {
  const sign = x < 0 ? "−" : "";
  const a = Math.abs(x);
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${sign}$${(a / 1e3).toFixed(1)}k`;
  return `${sign}$${num(a, 0)}`;
};

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export function kpis(container, items) {
  container.innerHTML = items
    .map(
      (k) => `<div class="kpi"><div class="kpi-label">${escapeHtml(k.label)}</div>
        <div class="kpi-value">${escapeHtml(k.value)}</div>${k.sub ? `<div class="kpi-sub">${escapeHtml(k.sub)}</div>` : ""}</div>`,
    )
    .join("");
}

/** columns: [{label, key|render, num?}] */
export function table(el, columns, rows, { highlight } = {}) {
  const head = `<thead><tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}">${escapeHtml(c.label)}</th>`).join("")}</tr></thead>`;
  const body = rows
    .map((r, i) => {
      const cls = highlight && highlight(r, i) ? ' class="highlight"' : "";
      const cells = columns
        .map((c) => {
          const v = c.render ? c.render(r, i) : escapeHtml(r[c.key]);
          return `<td class="${c.num ? "num" : ""}">${v}</td>`;
        })
        .join("");
      return `<tr${cls}>${cells}</tr>`;
    })
    .join("");
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const palette = () => ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5"].map(cssVar);

const charts = new Map();

/** Create or replace a Chart.js chart on a canvas id, themed from CSS variables. */
export function chart(id, config) {
  const Chart = window.Chart;
  if (!Chart) return null;
  const text = cssVar("--muted");
  const grid = cssVar("--grid");
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = text;
  Chart.defaults.borderColor = grid;
  const existing = charts.get(id);
  if (existing) existing.destroy();
  const base = {
    responsive: true,
    maintainAspectRatio: false,
    animation: matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 300 },
    plugins: { legend: { labels: { boxWidth: 12, boxHeight: 12, usePointStyle: true } }, tooltip: { padding: 10 } },
  };
  config.options = deepMerge(base, config.options || {});
  const c = new Chart(document.getElementById(id), config);
  charts.set(id, c);
  return c;
}

function deepMerge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) && a[k] && typeof a[k] === "object" ? deepMerge(a[k], v) : v;
  }
  return out;
}

export function download(filename, text, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const csvCell = (v) => {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export const toCsv = (headers, rows) => [headers, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n");

/** RFC 4180-ish parser: quoted fields, escaped quotes, CRLF. Returns array of string arrays. */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((c) => c !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((c) => c !== "")) rows.push(row);
  return rows;
}

export function riskBand(p) {
  if (p >= 0.7) return { label: "High risk", cls: "pill-risk", color: cssVar("--risk") };
  if (p >= 0.3) return { label: "Medium risk", cls: "pill-warn", color: cssVar("--warn") };
  return { label: "Low risk", cls: "pill-safe", color: cssVar("--safe") };
}
