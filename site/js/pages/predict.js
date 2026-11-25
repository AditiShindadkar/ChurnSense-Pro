import { $, $$, pct, money, escapeHtml, riskBand } from "../ui.js";
import { encode, predict, groupedShap, whatIf } from "../model.js";
import { bestOffer, lifetimeValue } from "../business.js";

const PRESETS = {
  safe: { Age: 34, Gender: "Male", Tenure: 40, "Usage Frequency": 22, "Support Calls": 1, "Payment Delay": 4, "Subscription Type": "Premium", "Contract Length": "Annual", "Total Spend": 850, "Last Interaction": 4 },
  risky: { Age: 28, Gender: "Female", Tenure: 6, "Usage Frequency": 5, "Support Calls": 7, "Payment Delay": 24, "Subscription Type": "Basic", "Contract Length": "Monthly", "Total Spend": 320, "Last Interaction": 25 },
};

let state = null;

export function init(ctx) {
  const { schema } = ctx.model;
  const fields = [];
  for (const col of schema.numeric) {
    const r = schema.ranges[col];
    const step = col === "Total Spend" ? 10 : 1;
    fields.push(`<label class="field"><span>${col}</span>
      <div class="slider-row">
        <input type="range" data-col="${col}" min="${r.min}" max="${r.max}" step="${step}" aria-label="${col}">
        <input type="number" data-col="${col}" min="${r.min}" max="${r.max}" step="${step}" aria-label="${col} value">
      </div></label>`);
  }
  for (const [col, levels] of Object.entries(schema.categorical)) {
    fields.push(`<label class="field"><span>${col}</span>
      <select data-col="${col}">${levels.map((l) => `<option>${l}</option>`).join("")}</select></label>`);
  }
  $("#predict-fields").innerHTML = fields.join("");

  $$("#predict-fields [data-col]").forEach((el) =>
    el.addEventListener("input", () => {
      const col = el.dataset.col;
      const v = el.type === "range" || el.type === "number" ? Number(el.value) : el.value;
      if (el.type === "number" && (el.value === "" || !Number.isFinite(v))) return;
      state[col] = v;
      // keep slider and number box in sync
      $$(`#predict-fields [data-col="${col}"]`).forEach((other) => other !== el && (other.value = v));
      update(ctx);
    }),
  );

  $$("[data-preset]").forEach((b) => b.addEventListener("click", () => setCustomer(ctx, preset(ctx, b.dataset.preset))));
  setCustomer(ctx, preset(ctx, "borderline"), false);
}

function preset(ctx, name) {
  if (PRESETS[name]) return { ...PRESETS[name] };
  const pool = ctx.sample;
  if (name === "borderline") {
    const c = pool.reduce((a, b) => (Math.abs(b.prob - 0.5) < Math.abs(a.prob - 0.5) ? b : a));
    return pick(ctx, c.raw);
  }
  return pick(ctx, pool[Math.floor(Math.random() * pool.length)].raw);
}

function pick({ model }, raw) {
  const out = {};
  for (const c of [...model.schema.numeric, ...Object.keys(model.schema.categorical)]) out[c] = raw[c];
  return out;
}

function setCustomer(ctx, raw, render = true) {
  state = raw;
  for (const [col, v] of Object.entries(raw)) $$(`#predict-fields [data-col="${col}"]`).forEach((el) => (el.value = v));
  if (render) update(ctx);
}

export function render(ctx) {
  update(ctx);
}

export const onShow = render;

function update(ctx) {
  const { model, economics } = ctx;
  let x;
  try {
    x = encode(model.schema, state);
  } catch (err) {
    $("#risk-band").textContent = err.message;
    return;
  }
  const p = predict(model, x);
  const band = riskBand(p);
  $("#risk-value").textContent = pct(p, 1);
  $("#risk-value").style.color = band.color;
  $("#risk-meter").style.width = `${Math.max(2, p * 100)}%`;
  $("#risk-meter").style.background = band.color;
  $("#risk-band").innerHTML = `<span class="pill ${band.cls}">${band.label}</span>`;

  const value = lifetimeValue(state["Total Spend"], economics);
  const best = bestOffer(p, value, economics.offers, economics.contactCost);
  $("#risk-offer").innerHTML = best.offer >= 0
    ? `Best action: <strong>${escapeHtml(economics.offers[best.offer].name)}</strong>, expected value ${money(best.ev)} (customer value ${money(value)})`
    : `Best action: <strong>no offer</strong>. No offer has a positive expected value at this risk (customer value ${money(value)}).`;

  const { items } = groupedShap(model, x);
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  $("#shap-bars").innerHTML = items
    .map((it) => {
      const w = (Math.abs(it.value) / max) * 50;
      const shown = state[it.feature];
      return `<div class="shap-row">
        <div class="name" title="${escapeHtml(it.feature)} = ${escapeHtml(shown)}">${escapeHtml(it.feature)} <small>${escapeHtml(shown)}</small></div>
        <div class="shap-track"><div class="shap-bar ${it.value >= 0 ? "pos" : "neg"}" style="width:${w}%"></div></div>
        <div class="val">${it.value >= 0 ? "+" : "−"}${Math.abs(it.value).toFixed(2)}</div></div>`;
    })
    .join("");

  const wi = whatIf(model, state);
  const helpful = wi.actions.filter((a) => a.delta < -0.005);
  $("#whatif").innerHTML = helpful.length
    ? helpful
        .map((a, i) => `<div class="whatif-row"><span>${escapeHtml(a.label)}</span>
          <span><span class="pill pill-safe">${pct(a.prob, 1)} (${(a.delta * 100).toFixed(1)} pts)</span>
          <button type="button" class="btn btn-small" data-apply="${i}">Apply</button></span></div>`)
        .join("")
    : `<p class="muted small">No single change lowers this customer’s risk by more than half a point${p < 0.05 ? ". They are already low risk." : "."}</p>`;
  $$("#whatif [data-apply]").forEach((b) =>
    b.addEventListener("click", () => setCustomer(ctx, { ...state, ...helpful[Number(b.dataset.apply)].change })),
  );
}
