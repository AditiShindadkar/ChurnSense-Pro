// In-browser LightGBM inference and exact TreeSHAP.
// Mirrors churnsense/data.py (encoding) and churnsense/export.py (tree format).
// Verified against Python by tests_js/model.test.mjs.

export function featureNames(schema) {
  const names = [...schema.numeric];
  for (const [col, levels] of Object.entries(schema.categorical)) {
    for (const level of levels.slice(1)) names.push(`${col} = ${level}`);
  }
  return names;
}

export const sourceFeature = (name) => name.split(" = ")[0];

/** Raw customer object -> encoded feature vector. Throws on bad input. */
export function encode(schema, raw) {
  const x = [];
  for (const col of schema.numeric) {
    const v = Number(raw[col]);
    if (raw[col] === "" || raw[col] == null || !Number.isFinite(v)) throw new Error(`"${col}" must be a number`);
    x.push(v);
  }
  for (const [col, levels] of Object.entries(schema.categorical)) {
    const value = String(raw[col] ?? "").trim();
    if (!levels.includes(value)) throw new Error(`"${col}" must be one of ${levels.join(", ")}`);
    for (const level of levels.slice(1)) x.push(value === level ? 1 : 0);
  }
  return x;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function leafValue(tree, x) {
  let i = 0;
  while (tree.f[i] !== -1) i = x[tree.f[i]] <= tree.t[i] ? tree.l[i] : tree.r[i];
  return tree.v[i];
}

export function rawScore(model, x) {
  let s = 0;
  for (const tree of model.trees) s += leafValue(tree, x);
  return s;
}

export const predict = (model, x) => sigmoid(rawScore(model, x));

// ---------------------------------------------------------------------------
// TreeSHAP (Lundberg et al. 2018, Algorithm 2), exact for tree ensembles.
// Path entries: d = feature index, z = zero fraction, o = one fraction, w = weight.

function extendPath(path, depth, z, o, d) {
  path[depth] = { d, z, o, w: depth === 0 ? 1 : 0 };
  for (let i = depth - 1; i >= 0; i--) {
    path[i + 1].w += (o * path[i].w * (i + 1)) / (depth + 1);
    path[i].w = (z * path[i].w * (depth - i)) / (depth + 1);
  }
}

function unwindPath(path, depth, k) {
  const { o, z } = path[k];
  let next = path[depth].w;
  for (let i = depth - 1; i >= 0; i--) {
    if (o !== 0) {
      const tmp = path[i].w;
      path[i].w = (next * (depth + 1)) / ((i + 1) * o);
      next = tmp - (path[i].w * z * (depth - i)) / (depth + 1);
    } else {
      path[i].w = (path[i].w * (depth + 1)) / (z * (depth - i));
    }
  }
  for (let i = k; i < depth; i++) {
    path[i].d = path[i + 1].d;
    path[i].z = path[i + 1].z;
    path[i].o = path[i + 1].o;
  }
}

function unwoundPathSum(path, depth, k) {
  const { o, z } = path[k];
  let next = path[depth].w;
  let total = 0;
  for (let i = depth - 1; i >= 0; i--) {
    if (o !== 0) {
      const tmp = (next * (depth + 1)) / ((i + 1) * o);
      total += tmp;
      next = path[i].w - (tmp * z * (depth - i)) / (depth + 1);
    } else {
      total += path[i].w / z / ((depth - i) / (depth + 1));
    }
  }
  return total;
}

function recurse(tree, x, phi, node, parentPath, depth, z, o, d) {
  const path = parentPath.slice(0, depth).map((p) => ({ ...p }));
  extendPath(path, depth, z, o, d);

  if (tree.f[node] === -1) {
    for (let i = 1; i <= depth; i++) {
      const w = unwoundPathSum(path, depth, i);
      phi[path[i].d] += w * (path[i].o - path[i].z) * tree.v[node];
    }
    return;
  }

  const f = tree.f[node];
  const goLeft = x[f] <= tree.t[node];
  const hot = goLeft ? tree.l[node] : tree.r[node];
  const cold = goLeft ? tree.r[node] : tree.l[node];
  const cover = tree.c[node];
  let inZ = 1;
  let inO = 1;

  let k = 0;
  while (k <= depth && path[k].d !== f) k++;
  if (k !== depth + 1) {
    inZ = path[k].z;
    inO = path[k].o;
    unwindPath(path, depth, k);
    depth -= 1;
  }

  recurse(tree, x, phi, hot, path, depth + 1, (tree.c[hot] / cover) * inZ, inO, f);
  recurse(tree, x, phi, cold, path, depth + 1, (tree.c[cold] / cover) * inZ, 0, f);
}

function treeExpectedValue(tree, node = 0) {
  if (tree.f[node] === -1) return tree.v[node];
  const c = tree.c[node];
  return (tree.c[tree.l[node]] / c) * treeExpectedValue(tree, tree.l[node]) +
    (tree.c[tree.r[node]] / c) * treeExpectedValue(tree, tree.r[node]);
}

export function expectedValue(model) {
  if (model._ev === undefined) model._ev = model.trees.reduce((s, t) => s + treeExpectedValue(t), 0);
  return model._ev;
}

/** SHAP values in log-odds, one per encoded feature. sum(phi) + base = rawScore. */
export function shap(model, x) {
  const phi = new Array(x.length).fill(0);
  for (const tree of model.trees) {
    if (tree.f[0] === -1) continue; // single-leaf tree: no feature contributes
    recurse(tree, x, phi, 0, [], 0, 1, 1, -1);
  }
  return { base: expectedValue(model), phi };
}

/** One-hot SHAP values summed back to the original column. */
export function groupedShap(model, x) {
  const { base, phi } = shap(model, x);
  const groups = new Map();
  model.features.forEach((name, i) => {
    const src = sourceFeature(name);
    groups.set(src, (groups.get(src) ?? 0) + phi[i]);
  });
  const items = [...groups].map(([feature, value]) => ({ feature, value }));
  items.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return { base, items };
}

/** Convenience: raw customer -> {prob, shap}. */
export function score(model, raw) {
  const x = encode(model.schema, raw);
  return { x, prob: predict(model, x) };
}

// Which realistic changes to *actionable* columns would lower this customer's risk most?
const ACTIONS = [
  { col: "Contract Length", label: "Move to an Annual contract", apply: (r) => (r["Contract Length"] === "Annual" ? null : { "Contract Length": "Annual" }) },
  { col: "Support Calls", label: "Resolve issues (support calls → 2)", apply: (r) => (r["Support Calls"] > 2 ? { "Support Calls": 2 } : null) },
  { col: "Payment Delay", label: "Autopay / reminder (payment delay → 5 days)", apply: (r) => (r["Payment Delay"] > 5 ? { "Payment Delay": 5 } : null) },
  { col: "Last Interaction", label: "Re-engage now (last interaction → 1 day)", apply: (r) => (r["Last Interaction"] > 3 ? { "Last Interaction": 1 } : null) },
  { col: "Usage Frequency", label: "Drive usage (usage frequency +10)", apply: (r) => (r["Usage Frequency"] < 25 ? { "Usage Frequency": Math.min(30, Number(r["Usage Frequency"]) + 10) } : null) },
  { col: "Total Spend", label: "Upsell a bundle (total spend +$200)", apply: (r) => (r["Total Spend"] < 900 ? { "Total Spend": Math.min(1000, Number(r["Total Spend"]) + 200) } : null) },
];

export function whatIf(model, raw) {
  const base = score(model, raw).prob;
  const out = [];
  for (const a of ACTIONS) {
    const change = a.apply(raw);
    if (!change) continue;
    const prob = score(model, { ...raw, ...change }).prob;
    out.push({ label: a.label, change, prob, delta: prob - base });
  }
  out.sort((a, b) => a.delta - b.delta);
  return { base, actions: out };
}
