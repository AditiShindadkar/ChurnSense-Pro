import { $, pct, num, money, chart, palette, escapeHtml } from "../ui.js";

// Playbook lines keyed by "feature:high|low" — chosen from the segment's most distinctive traits.
const PLAYBOOK = {
  "Support Calls:high": "Route to a senior support queue and fix the underlying issue before offering discounts",
  "Payment Delay:high": "Offer autopay with a small credit, and send reminders before the due date",
  "Total Spend:low": "Upsell a bundle or annual plan, since low spend is a churn cut-off in this data",
  "Total Spend:high": "Protect revenue with VIP perks and an account manager",
  "Last Interaction:high": "Run a re-engagement campaign: what’s new, usage tips, a check-in call",
  "Last Interaction:low": "Recently engaged, so ask for feedback or a referral while they’re active",
  "Usage Frequency:low": "Onboarding nudges and feature discovery to build the habit",
  "Usage Frequency:high": "Power users: early access and loyalty rewards",
  "Tenure:low": "Strengthen the first 90 days with a welcome series and success check-ins",
  "Tenure:high": "Long-time customers: thank-you perks and a contract renewal incentive",
  "Support Calls:low": "Low-touch: keep to light automated communication",
  "Payment Delay:low": "Reliable payers: annual-plan upgrade offer",
};

export function init() {}

export function render({ insights }) {
  const segs = insights.segments.profiles;
  const colors = palette();
  $("#segments").innerHTML = segs
    .map((s, i) => {
      const traits = Object.entries(s.z).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
      const plays = traits.map(([f, z]) => PLAYBOOK[`${f}:${z > 0 ? "high" : "low"}`]).filter(Boolean).slice(0, 2);
      const risk = s.churn_rate >= 0.7 ? "pill-risk" : s.churn_rate >= 0.45 ? "pill-warn" : "pill-safe";
      return `<article class="card segment">
        <div class="segment-top">
          <h2>${escapeHtml(s.name)}</h2>
          <span class="segment-dot" style="background:${colors[i % colors.length]}"></span>
        </div>
        <div><span class="pill ${risk}">${pct(s.churn_rate, 0)} churn</span></div>
        <div class="stats">
          <span>Share of customers</span><b>${pct(s.share, 0)}</b>
          <span>Avg tenure</span><b>${num(s.means.Tenure, 0)} mo</b>
          <span>Avg spend</span><b>${money(s.means["Total Spend"])}</b>
          <span>Support calls</span><b>${num(s.means["Support Calls"], 1)}</b>
          <span>Payment delay</span><b>${num(s.means["Payment Delay"], 0)} days</b>
          <span>On monthly contract</span><b>${pct(s.monthly_contract_share, 0)}</b>
        </div>
        <div class="playbook"><strong>Playbook</strong><ul>${plays.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul></div>
      </article>`;
    })
    .join("");

  chart("segments-chart", {
    type: "bubble",
    data: {
      datasets: segs.map((s, i) => ({
        label: s.name,
        data: [{ x: s.means["Total Spend"], y: s.churn_rate, r: 8 + Math.sqrt(s.share) * 40 }],
        backgroundColor: colors[i % colors.length] + "b3",
        borderColor: colors[i % colors.length],
      })),
    },
    options: {
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw.y, 0)} churn, ${money(c.raw.x)} avg spend` } } },
      scales: {
        x: { title: { display: true, text: "Average total spend" }, ticks: { callback: (v) => money(v) } },
        y: { min: 0, max: 1.1, title: { display: true, text: "Churn rate" }, ticks: { callback: (v) => (v <= 1 ? pct(v, 0) : "") } },
      },
    },
  });
}
