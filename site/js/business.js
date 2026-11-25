// Retention economics. Mirrors churnsense/business.py — see that file for the maths.
//   EV(customer, offer) = p * successRate * value - (offerCost + contactCost)

export const DEFAULT_OFFERS = [
  { name: "Email nudge", cost: 2, success: 0.05 },
  { name: "15% loyalty discount", cost: 30, success: 0.25 },
  { name: "Retention call + VIP offer", cost: 60, success: 0.4 },
];

export function lifetimeValue(totalSpend, p) {
  return p.valueMode === "flat" ? p.flatValue : totalSpend * p.margin * p.horizon;
}

/** Best offer for one customer: {offer: index or -1, ev}. */
export function bestOffer(prob, value, offers, contactCost = 0) {
  let best = -1;
  let bestEv = 0;
  offers.forEach((o, k) => {
    const ev = prob * o.success * value - (o.cost + contactCost);
    if (ev > bestEv) {
      bestEv = ev;
      best = k;
    }
  });
  return { offer: best, ev: bestEv };
}

/** Greedy budgeted plan over many customers. Returns chosen offer index per customer (-1 = skip). */
export function planWithBudget(customers, offers, { budget = Infinity, contactCost = 0 } = {}) {
  const picks = customers.map((c) => bestOffer(c.prob, c.value, offers, contactCost));
  const order = picks
    .map((p, i) => ({ i, ratio: p.offer >= 0 ? p.ev / (offers[p.offer].cost + contactCost) : -1 }))
    .filter((o) => o.ratio > 0)
    .sort((a, b) => b.ratio - a.ratio);
  const chosen = new Array(customers.length).fill(-1);
  let spent = 0;
  for (const { i } of order) {
    const cost = offers[picks[i].offer].cost + contactCost;
    if (spent + cost > budget) continue;
    chosen[i] = picks[i].offer;
    spent += cost;
  }
  return { chosen, spent };
}

/** Expected profit of a plan, given who actually churned. */
export function realisedProfit(customers, chosen, offers, contactCost = 0) {
  let profit = 0;
  let saved = 0;
  customers.forEach((c, i) => {
    const k = chosen[i];
    if (k < 0) return;
    const keep = c.churn * offers[k].success;
    saved += keep;
    profit += keep * c.value - (offers[k].cost + contactCost);
  });
  return { profit, saved };
}

export function thresholdPlan(customers, threshold, offerIndex) {
  return customers.map((c) => (c.prob >= threshold ? offerIndex : -1));
}
