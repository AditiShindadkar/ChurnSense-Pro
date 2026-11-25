import { $, $$ } from "./ui.js";
import { DEFAULT_OFFERS } from "./business.js";
import * as overview from "./pages/overview.js";
import * as explore from "./pages/explore.js";
import * as models from "./pages/models.js";
import * as predict from "./pages/predict.js";
import * as planner from "./pages/planner.js";
import * as segments from "./pages/segments.js";
import * as batch from "./pages/batch.js";

const PAGES = { overview, explore, models, predict, planner, segments, batch };
const STORE_KEY = "churnsense.economics.v1";

function loadEconomics() {
  const defaults = {
    base: 100000, valueMode: "spend", margin: 0.4, horizon: 2, flatValue: 500, contactCost: 1, budget: null,
    offers: DEFAULT_OFFERS.map((o) => ({ ...o })),
  };
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    return saved ? { ...defaults, ...saved } : defaults;
  } catch {
    return defaults;
  }
}

function saveEconomics(e) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(e));
  } catch {
    /* storage unavailable: settings just won't persist */
  }
}

async function getJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function setupTheme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("churnsense.theme");
    if (saved) root.dataset.theme = saved;
  } catch { /* ignore */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("churnsense.theme", root.dataset.theme); } catch { /* ignore */ }
    ctx.themeVersion++;
    show(currentPage(), true);
  });
}

const ctx = { themeVersion: 0, economics: loadEconomics(), saveEconomics: () => saveEconomics(ctx.economics) };
const initialised = new Map();

const currentPage = () => {
  const id = location.hash.replace("#", "");
  return PAGES[id] ? id : "overview";
};

function show(id, rerender = false) {
  $$(".page").forEach((p) => (p.hidden = p.id !== `page-${id}`));
  $$(".tabs a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${id}`));
  const page = PAGES[id];
  // Charts are (re)built when a page is first shown, after a theme change, or when economics changed elsewhere.
  if (initialised.get(id) !== ctx.themeVersion || rerender) {
    if (!initialised.has(id)) page.init(ctx);
    page.render?.(ctx);
    initialised.set(id, ctx.themeVersion);
  } else {
    page.onShow?.(ctx);
  }
  document.title = id === "overview" ? "ChurnSense Pro" : `${$(`.tabs a[href="#${id}"]`).textContent} · ChurnSense Pro`;
}

async function main() {
  setupTheme();
  try {
    const [model, insights, sample] = await Promise.all([
      getJson("data/model.json"),
      getJson("data/insights.json"),
      getJson("data/sample.json"),
    ]);
    Object.assign(ctx, { model, insights, sample: toCustomers(sample), sampleMeta: sample });
  } catch (err) {
    $("#loading").textContent = `Could not load the model (${err.message}). If you opened index.html directly, run a local server instead: python -m http.server -d site`;
    return;
  }
  if (!window.Chart) await new Promise((r) => window.addEventListener("load", r, { once: true }));
  // Chart.js measures label widths once, so wait for the web font to avoid clipped axis labels.
  await Promise.race([document.fonts?.ready, new Promise((r) => setTimeout(r, 1500))]);
  $("#loading").remove();
  window.addEventListener("hashchange", () => {
    show(currentPage());
    window.scrollTo({ top: 0 });
  });
  show(currentPage());
}

function toCustomers(sample) {
  const cols = sample.columns;
  return sample.rows.map((r) => {
    const raw = {};
    cols.forEach((c, i) => (raw[c] = r[i]));
    return { raw, churn: raw.Churn, prob: raw.risk, segment: raw.segment };
  });
}

main();
