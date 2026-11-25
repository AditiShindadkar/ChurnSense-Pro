# ChurnSense Pro

**Predict which customers will churn, explain why, and decide who is worth saving, all in the browser.**

ChurnSense Pro trains and compares five approaches on 440,832 customers. It ships a LightGBM model that runs entirely
client-side, with exact SHAP explanations, a profit-driven retention planner and batch CSV scoring.
No server, no sign-up, and no customer data ever leaves the visitor's device.

**Live demo:** _link added after the first Vercel deploy_

| Page | What it does |
|---|---|
| **Overview** | Headline results, and why 99.9% accuracy needs context on this dataset |
| **Explore** | Churn rate per feature next to the model's partial dependence |
| **Models** | Rule baseline vs Logistic Regression, Random Forest, XGBoost and LightGBM; ROC, confusion matrix, calibration, SMOTE and feature-engineering experiments |
| **Predict** | Score one customer, see exact SHAP contributions, try what-if changes, get the best offer |
| **Retention planner** | Margin, horizon, contact cost, budget and multiple offers → expected profit, strategy comparison, downloadable target list |
| **Segments** | KMeans personas with auto-generated names and playbooks |
| **Batch scoring** | Upload a CSV, get risk, main driver and recommended offer for every row |

## Results

Held-out test set of 88,167 customers (full table in [reports/metrics.md](reports/metrics.md)):

| Model | Accuracy | ROC-AUC | Errors |
|---|---|---|---|
| 5-rule baseline (no ML) | 96.83% | 0.972 | 2,791 |
| Logistic Regression | 89.33% | 0.959 | 9,410 |
| Random Forest | 99.90% | 1.000 | 90 |
| XGBoost | 99.99% | 1.000 | 7 |
| **LightGBM (deployed)** | **99.98%** | **1.000** | **18** |

5-fold cross-validated ROC-AUC for LightGBM: 0.999999 ± 0.000001.

## What makes this different

1. **It is honest about the 99.9%.** The [Kaggle customer churn dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
   is synthetic. Churn was generated from hard cut-offs (Monthly contract, more than 5 support calls, payment delay over 20 days, spend under $500,
   age over 50), and every customer matching any of them churned. A five-rule baseline with no ML is the real bar to beat. ML cuts errors
   from 2,791 to 18 by learning the interactions behind the remaining churners.
2. **No SMOTE.** The classes are 57/43, so they are not imbalanced. SMOTE leaves ranking unchanged but makes probabilities slightly worse,
   and the profit maths multiplies by those probabilities. The experiment is on the Models page.
3. **Feature engineering is tested, not assumed.** The four behavioural features help Logistic Regression (285 fewer errors) but not
   LightGBM (0), so the deployed model uses raw features and stays easy to explain.
4. **Decisions are based on expected value, not a threshold.** For every customer and every offer:
   `EV = churn probability × offer success rate × customer value − (offer cost + contact cost)`. The planner picks the best offer per
   customer and funds the highest EV-per-dollar first, within your budget.
5. **Exact SHAP in JavaScript.** TreeSHAP is re-implemented in [site/js/model.js](site/js/model.js) and tested against LightGBM's own
   `pred_contrib` to within 1e-6 on 300 customers.

## How it works

```
data/*.csv ──► scripts/train.py ──► site/data/model.json     (LightGBM trees, 326 KB)
                    │               site/data/insights.json  (metrics, curves, EDA, segments)
                    │               site/data/sample.json    (5,000 scored test customers)
                    ▼
             churnsense/ (Python package)          site/ (static HTML + JS, deployed to Vercel)
             data · rules · models · business       model.js: tree inference + TreeSHAP
             segments · export                     business.js: expected-value planner
```

## Run it locally

```bash
# Website only (uses the committed model)
python -m http.server 8000 -d site
# open http://localhost:8000

# Retrain everything (about 2 minutes)
pip install -r requirements.txt
python scripts/train.py          # add --fast to skip Random Forest, XGBoost and CV

# Tests
python -m pytest -q tests        # Python: encoding, rules, business maths, segments
node --test tests_js/            # JS model matches Python predictions and SHAP values
```

## Deploy

The site is static, so Vercel serves the `site/` folder with no build step (see [vercel.json](vercel.json)).

```bash
npx vercel --prod
```

## Project structure

```
churnsense/      Python package: data prep, rule baseline, models, business maths, segments, export
scripts/train.py Train, evaluate and export everything the site needs
site/            Static website (index.html, styles.css, js/, data/)
tests/           pytest suite + fixtures used by the JS parity test
tests_js/        node:test suite for the browser model
data/            Training data (Kaggle customer churn dataset)
reports/         Generated model comparison
```

## Limitations and next steps

- The dataset is synthetic. On real data, expect ROC-AUC of about 0.75–0.90, and calibration will matter even more.
- `Total Spend` is treated as yearly revenue for customer value. Switch to a flat CLV in the planner if that doesn't fit.
- Offer success rates are assumptions. In production they should come from an A/B test or an uplift model.
- Ideas: uplift modelling (who is *persuadable*, not just at risk), survival analysis for *when* customers churn, drift monitoring.

---

Built by Aditi Shindadkar.
