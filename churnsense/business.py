"""Retention economics: who to contact, with which offer, and what it is worth.

This is the reference implementation. site/js/business.js mirrors it for the
interactive planner; tests in both languages use the same worked examples.

Model of a retention offer
--------------------------
For customer i with churn probability p_i and lifetime value V_i, an offer
with cost c and success rate s (share of would-be churners it keeps) has

    expected value  EV_i = p_i * s * V_i - c

Contact the customer with the best offer only if that EV is positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Offer:
    name: str
    cost: float
    success_rate: float


DEFAULT_OFFERS = (
    Offer("Email nudge", 2.0, 0.05),
    Offer("15% loyalty discount", 30.0, 0.25),
    Offer("Retention call + VIP offer", 60.0, 0.40),
)


def lifetime_value(total_spend, margin: float, horizon_years: float, flat_value: float | None = None):
    """Per-customer value. Total Spend is treated as yearly revenue."""
    total_spend = np.asarray(total_spend, dtype=float)
    if flat_value is not None:
        return np.full_like(total_spend, float(flat_value))
    return total_spend * margin * horizon_years


def expected_values(prob, value, offers=DEFAULT_OFFERS, contact_cost: float = 0.0) -> np.ndarray:
    """Matrix (customers x offers) of expected net value."""
    prob = np.asarray(prob, dtype=float)[:, None]
    value = np.asarray(value, dtype=float)[:, None]
    s = np.array([o.success_rate for o in offers])[None, :]
    c = np.array([o.cost for o in offers])[None, :] + contact_cost
    return prob * s * value - c


def best_offers(prob, value, offers=DEFAULT_OFFERS, contact_cost: float = 0.0):
    """Index of best offer per customer (-1 = do not contact) and its EV (0 if none)."""
    ev = expected_values(prob, value, offers, contact_cost)
    idx = ev.argmax(axis=1)
    best = ev[np.arange(len(ev)), idx]
    idx = np.where(best > 0, idx, -1)
    return idx, np.where(best > 0, best, 0.0)


def plan_with_budget(prob, value, offers=DEFAULT_OFFERS, budget: float | None = None, contact_cost: float = 0.0):
    """Greedy plan: best offer per customer, highest EV-per-dollar first, until budget runs out."""
    idx, ev = best_offers(prob, value, offers, contact_cost)
    costs = np.array([o.cost + contact_cost for o in offers])
    chosen = np.full(len(idx), -1)
    spent = 0.0
    candidates = np.where(idx >= 0)[0]
    ratio = ev[candidates] / costs[idx[candidates]]
    for i in candidates[np.argsort(-ratio, kind="stable")]:
        cost = costs[idx[i]]
        if budget is not None and spent + cost > budget:
            continue
        chosen[i] = idx[i]
        spent += cost
    return chosen, spent


def realised_profit(y_true, chosen, value, offers=DEFAULT_OFFERS, contact_cost: float = 0.0) -> float:
    """Expected profit of a plan given the true outcome (offer success is still a rate)."""
    y_true = np.asarray(y_true)
    value = np.asarray(value, dtype=float)
    total = 0.0
    for k, offer in enumerate(offers):
        mask = np.asarray(chosen) == k
        total += float((y_true[mask] * offer.success_rate * value[mask]).sum())
        total -= mask.sum() * (offer.cost + contact_cost)
    return total


def threshold_profit(y_true, prob, threshold: float, value, offer: Offer, contact_cost: float = 0.0) -> float:
    """Classic approach: everyone above a probability threshold gets the same offer."""
    chosen = np.where(np.asarray(prob) >= threshold, 0, -1)
    return realised_profit(y_true, chosen, value, (offer,), contact_cost)
