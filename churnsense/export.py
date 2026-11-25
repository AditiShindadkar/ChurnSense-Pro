"""Export a trained LightGBM model to compact JSON for in-browser inference.

Each tree becomes parallel arrays (node i):
    f[i]  feature index, -1 for a leaf
    t[i]  split threshold (go left when x <= t)
    l[i], r[i]  child node indices
    v[i]  leaf value (0 for internal nodes)
    c[i]  number of training rows reaching the node (needed for TreeSHAP)
"""

from __future__ import annotations


def _flatten(node: dict, arrays: dict) -> int:
    i = len(arrays["f"])
    for key in arrays:
        arrays[key].append(0)
    if "leaf_value" in node:
        arrays["f"][i] = -1
        arrays["v"][i] = node["leaf_value"]
        arrays["c"][i] = node["leaf_count"]
        arrays["l"][i] = arrays["r"][i] = -1
        return i
    if node.get("decision_type", "<=") != "<=":
        raise ValueError(f"Unsupported split type {node['decision_type']!r}")
    arrays["f"][i] = node["split_feature"]
    arrays["t"][i] = node["threshold"]
    arrays["c"][i] = node["internal_count"]
    arrays["l"][i] = _flatten(node["left_child"], arrays)
    arrays["r"][i] = _flatten(node["right_child"], arrays)
    return i


def _depth(node: dict) -> int:
    if "leaf_value" in node:
        return 0
    return 1 + max(_depth(node["left_child"]), _depth(node["right_child"]))


def lightgbm_to_json(model, feature_names: list[str]) -> dict:
    dump = model.booster_.dump_model()
    if dump.get("objective", "").split()[0] != "binary":
        raise ValueError("Only binary objectives are supported")
    trees = []
    for info in dump["tree_info"]:
        arrays = {k: [] for k in "ftlrvc"}
        _flatten(info["tree_structure"], arrays)
        arrays["depth"] = _depth(info["tree_structure"])
        trees.append(arrays)
    return {"format": "churnsense-lgbm-v1", "features": list(feature_names), "trees": trees}
