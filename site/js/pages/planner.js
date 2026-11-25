import { $, $$, kpis, table, pct, num, money, chart, cssVar, palette, escapeHtml, download, toCsv } from "../ui.js";
import { lifetimeValue, planWithBudget, realisedProfit, bestOffer } from "../business.js";
import { encode, groupedShap } from "../model.js";

let lastPlan = null;

export function init(ctx) {
  const e = ctx.economics;
  $("#planner-n").textContent = num(ctx.sample.length);
  $("#p-base").value = e.base;
  $("#p-margin").value = e.margin;
  $("#p-horizon").value = e.horizon;
  $("#p-flat").value = e.flatValue;
  $("#p-contact").value = e.contactCost;
  $("#p-budget").value = e.budget ?? "";
  $(`input[name="p-valuemode"][value="${e.valueMode}"]`).checked = true;

  const read = () => {
    const n = (id, fallback) => {
      const v = Number($(id).value);
      return $(id).value !== "" && Number.isFinite(v) && v >= 0 ? v : fallback;
    };
    e.base = Math.max(1, n("#p-base", 100000));
    e.margin = n("#p-margin", 0.4);
    e.horizon = n("#p-horizon", 2);
    e.flatValue = n("#p-flat", 500);
    e.contactCost = n("#p-contact", 0);
    e.budget = $("#p-budget").value === "" ? null : n("#p-budget", null);
    e.valueMode = $('input[name="p-valuemode"]:checked').value;
    ctx.saveEconomics();
    render(ctx);
  };
  $$(".planner-controls input").forEach((el) => el.addEventListener("input", read));

  $("#p-add-offer").addEventListener("click", () => {
    e.offers.push({ name: `Offer ${e.offers.length + 1}`, cost: 20, success: 0.2 });
    ctx.saveEconomics();
    renderOffers(ctx);
    render(ctx);
  });
  $("#planner-download").addEventListener("click", () => exportPlan(ctx));
  renderOffers(ctx);
}

function renderOffers(ctx) {
  const offers = ctx.economics.offers;
  $("#p-offers").innerHTML = offers
    .map((o, i) => `<div class="offer" data-i="${i}">
      <div class="offer-top"><input type="text" data-k="name" value="${escapeHtml(o.name)}" aria-label="Offer name">
        ${offers.length > 1 ? `<button type="button" class="offer-remove" data-remove="${i}" aria-label="Remove offer">×</button>` : ""}</div>
      <div class="offer-nums">
        <label>Cost ($)<input type="number" data-k="cost" min="0" step="1" value="${o.cost}"></label>
        <label>Success rate (%)<input type="number" data-k="success" min="0" max="100" step="1" value="${Math.round(o.success * 100)}"></label>
      </div></div>`)
    .join("");
  $$("#p-offers .offer").forEach((box) => {
    const i = Number(box.dataset.i);
    $$("input", box).forEach((input) =>
      input.addEventListener("input", (ev) => {
        ev.stopPropagation();
        const k = input.dataset.k;
        if (k === "name") offers[i].name = input.value || `Offer ${i + 1}`;
        else {
          const v = Number(input.value);
          if (input.value === "" || !Number.isFinite(v) || v < 0) return;
          offers[i][k] = k === "success" ? Math.min(v, 100) / 100 : v;
        }
        ctx.saveEconomics();
        render(ctx);
      }),
    );
  });
  $$("#p-offers [data-remove]").forEach((b) =>
    b.addEventListener("click", () => {
      offers.splice(Number(b.dataset.remove), 1);
      ctx.saveEconomics();
      renderOffers(ctx);
      render(ctx);
    }),
  );
}

/** Threshold strategy with one offer; with a budget, highest-risk customers are funded first. */
function thresholdChosen(customers, threshold, k, offers, contactCost, budget) {
  const chosen = new Array(customers.length).fill(-1);
  const cost = offers[k].cost + contactCost;
  const order = customers.map((c, i) => i).filter((i) => customers[i].prob >= threshold).sort((a, b) => customers[b].prob - customers[a].prob);
  let spent = 0;
  for (const i of order) {
    if (budget != null && spent + cost > budget) break;
    chosen[i] = k;
    spent += cost;
  }
  return chosen;
}

export function render(ctx) {
  const e = ctx.economics;
  const scale = e.base / ctx.sample.length;
  const sampleBudget = e.budget == null ? null : e.budget / scale;
  const customers = ctx.sample.map((c) => ({ ...c, value: lifetimeValue(c.raw["Total Spend"], e) }));
  const offers = e.offers;
  $("#p-spend-fields").hidden = e.valueMode !== "spend";
  $("#p-flat-field").hidden = e.valueMode !== "flat";
  $("#o-margin").textContent = pct(e.margin, 0);
  $("#o-horizon").textContent = `${e.horizon} yr`;

  // Smart, personalised plan
  const smart = planWithBudget(customers, offers, { budget: sampleBudget ?? Infinity, contactCost: e.contactCost });
  const smartRes = realisedProfit(customers, smart.chosen, offers, e.contactCost);

  // Reference single offer for the threshold strategies: whichever does best at its own optimal threshold
  const thresholds = Array.from({ length: 21 }, (_, i) => i / 20);
  let ref = { k: 0, curve: [], bestT: 0, best: -Infinity };
  offers.forEach((_, k) => {
    const curve = thresholds.map((t) =>
      realisedProfit(customers, thresholdChosen(customers, t, k, offers, e.contactCost, sampleBudget), offers, e.contactCost).profit);
    const bi = curve.indexOf(Math.max(...curve));
    if (curve[bi] > ref.best) ref = { k, curve, bestT: thresholds[bi], best: curve[bi] };
  });
  $("#planner-ref-offer").textContent = `“${offers[ref.k].name}”`;

  const everyone = realisedProfit(customers, thresholdChosen(customers, 0, ref.k, offers, e.contactCost, sampleBudget), offers, e.contactCost).profit;
  const half = realisedProfit(customers, thresholdChosen(customers, 0.5, ref.k, offers, e.contactCost, sampleBudget), offers, e.contactCost).profit;

  const contacted = smart.chosen.filter((k) => k >= 0).length;
  const spend = smart.spent;
  kpis($("#planner-kpis"), [
    { label: "Expected profit", value: money(smartRes.profit * scale), sub: "smart targeting, net of all costs" },
    { label: "Customers to contact", value: num(contacted * scale), sub: `${pct(contacted / customers.length)} of base` },
    { label: "Campaign spend", value: money(spend * scale), sub: e.budget != null ? `budget ${money(e.budget)}` : "no budget cap" },
    { label: "Churners kept (expected)", value: num(smartRes.saved * scale), sub: spend ? `ROI ${pct(smartRes.profit / spend, 0)}` : "" },
  ]);

  const colors = palette();
  const strategies = [
    { name: "Do nothing", v: 0 },
    { name: `Contact everyone`, v: everyone },
    { name: "Risk ≥ 50%", v: half },
    { name: `Best threshold (${pct(ref.bestT, 0)})`, v: ref.best },
    { name: "Smart targeting", v: smartRes.profit },
  ];
  chart("planner-strategies", {
    type: "bar",
    data: {
      labels: strategies.map((s) => s.name),
      datasets: [{
        data: strategies.map((s) => s.v * scale),
        backgroundColor: strategies.map((s, i) => (i === 4 ? colors[0] : s.v < 0 ? cssVar("--risk") : colors[4])),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => money(c.raw) } } },
      scales: { y: { ticks: { callback: (v) => money(v) } }, x: { grid: { display: false } } },
    },
  });

  chart("planner-curve", {
    type: "line",
    data: {
      labels: thresholds.map((t) => pct(t, 0)),
      datasets: [
        { label: `Threshold + “${offers[ref.k].name}”`, data: ref.curve.map((v) => v * scale), borderColor: colors[1], backgroundColor: colors[1], borderWidth: 2.5, pointRadius: 2, tension: 0.2 },
        { label: "Smart targeting", data: thresholds.map(() => smartRes.profit * scale), borderColor: colors[0], borderDash: [6, 5], borderWidth: 2, pointRadius: 0 },
      ],
    },
    options: {
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${money(c.raw)}` } } },
      scales: { y: { ticks: { callback: (v) => money(v) } }, x: { title: { display: true, text: "Contact everyone with risk ≥" } } },
    },
  });

  // Target list
  const rows = customers
    .map((c, i) => ({ c, k: smart.chosen[i], ev: bestOffer(c.prob, c.value, offers, e.contactCost).ev }))
    .filter((r) => r.k >= 0)
    .sort((a, b) => b.ev - a.ev);
  lastPlan = { rows, offers };
  const top = rows.slice(0, 25).map((r) => {
    const { items } = groupedShap(ctx.model, encode(ctx.model.schema, r.c.raw));
    const driver = items.find((it) => it.value > 0);
    return { ...r, driver: driver ? `${driver.feature} = ${r.c.raw[driver.feature]}` : "–" };
  });
  const segNames = ctx.insights.segments.profiles.map((p) => p.name);
  table($("#planner-table"), [
    { label: "Risk", num: true, render: (r) => pct(r.c.prob, 1) },
    { label: "Value", num: true, render: (r) => money(r.c.value) },
    { label: "Main driver", render: (r) => escapeHtml(r.driver) },
    { label: "Segment", render: (r) => escapeHtml(segNames[r.c.segment] ?? "") },
    { label: "Offer", render: (r) => `<span class="pill pill-accent">${escapeHtml(offers[r.k].name)}</span>` },
    { label: "Expected value", num: true, render: (r) => money(r.ev) },
  ], top);
}

function exportPlan(ctx) {
  if (!lastPlan) return;
  const cols = [...ctx.model.schema.numeric, ...Object.keys(ctx.model.schema.categorical)];
  const csv = toCsv(
    [...cols, "churn_risk", "customer_value", "recommended_offer", "expected_value"],
    lastPlan.rows.map((r) => [...cols.map((c) => r.c.raw[c]), r.c.prob.toFixed(4), r.c.value.toFixed(2), lastPlan.offers[r.k].name, r.ev.toFixed(2)]),
  );
  download("churnsense-target-list.csv", csv);
}
