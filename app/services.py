"""CART 训练、评估与对比的服务层：把算法结果转成前端可渲染的 JSON。"""
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np

from algorithms import cart, pruning
from core.datasets import LABELS_ZH, load_iris, stratified_split
from core.metrics import classification_report, confusion_matrix
from core.pca import PCA
from core.preprocessing import standardize

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
FEATURE_LABELS = {
    "sepal_length": "花萼长度",
    "sepal_width": "花萼宽度",
    "petal_length": "花瓣长度",
    "petal_width": "花瓣宽度",
}


def _int(config, key, default, low, high):
    value = config.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
    return value


def _settings(config):
    if not isinstance(config, Mapping):
        raise ValueError("配置必须为对象")
    seed = _int(config, "seed", 42, 0, 2**31 - 1)
    depth = config.get("max_depth", 4)
    if depth is None:
        pass
    elif type(depth) is not int or not 0 <= depth <= 12:
        raise ValueError("max_depth 必须为 0–12 的整数或留空")
    params = {
        "max_depth": depth,
        "min_samples_leaf": _int(config, "min_samples_leaf", 1, 1, 50),
        "min_gain": float(config.get("min_gain", 0.0)),
        "seed": seed,
    }
    if not np.isfinite(params["min_gain"]) or params["min_gain"] < 0:
        raise ValueError("min_gain 必须为非负数")
    params["ccp"] = bool(config.get("ccp", True))
    alpha = config.get("alpha")
    if alpha in (None, ""):
        params["alpha"] = None
    else:
        params["alpha"] = float(alpha)
        if not np.isfinite(params["alpha"]) or params["alpha"] < 0:
            raise ValueError("alpha 必须为非负数")
    return params


def dataset_info():
    data = load_iris(DATASETS)
    return {
        "id": "iris",
        "name": data.name,
        "description": data.description,
        "source": data.source,
        "samples": len(data.features),
        "features": list(data.feature_names),
        "feature_labels": [FEATURE_LABELS.get(n, n) for n in data.feature_names],
        "classes": list(data.target_names),
        "class_labels": list(LABELS_ZH),
        "counts": np.bincount(data.labels).tolist(),
    }


def _evaluate(tree, x, y, classes):
    pred, paths = cart.predict(tree, x)
    cm = confusion_matrix(y, pred, len(classes))
    report = classification_report(cm, classes)
    return report, pred, paths


def _node_json(tree, node, seen_candidates):
    item = {
        "id": node.id,
        "depth": node.depth,
        "n": node.n_samples,
        "counts": list(node.counts),
        "gini": round(node.gini, 6),
        "prediction": node.prediction,
        "leaf": node.is_leaf,
    }
    if node.is_leaf:
        item["reason"] = node.reason
        return item
    item.update({
        "feature": node.feature,
        "feature_name": tree.feature_names[node.feature],
        "threshold": node.threshold,
        "gain": round(node.gain, 6),
        "candidates": seen_candidates.get(node.id, []),
    })
    return item


def _tree_payload(tree, candidate_map):
    def walk(node_id):
        node = tree.nodes[node_id]
        item = _node_json(tree, node, candidate_map)
        if not node.is_leaf:
            item["children"] = [walk(node.left), walk(node.right)]
        return item
    return walk(tree.root)


def _frames(build_frames, tree):
    names = tree.feature_names
    out = []
    for frame in build_frames:
        out.append({
            **frame,
            "feature_name": names[frame["feature"]] if frame["feature"] is not None else None,
            "candidates": [
                {**c, "feature_name": names[c["feature"]]}
                for c in frame["candidates"]
            ],
        })
    return out


def run_training(config):
    params = _settings(config)
    data = load_iris(DATASETS)
    classes = list(data.target_names)
    train_idx, val_idx, test_idx = stratified_split(data.labels, seed=params["seed"])
    x, y = data.features, data.labels
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    start = perf_counter()
    tree, build_frames = cart.fit(
        x_train, y_train, feature_names=data.feature_names, class_names=classes,
        max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
        min_gain=params["min_gain"], record=True,
    )
    build_ms = (perf_counter() - start) * 1000

    prune_steps = []
    chosen_alpha = 0.0
    final_tree = tree
    chosen_step = None
    if params["ccp"]:
        prune_steps = pruning.prune_sequence(tree, x_train, y_train, x_val, y_val)
        chosen_step = (pruning.select_alpha(prune_steps, params["alpha"])
                       if params["alpha"] is not None else pruning.best_step(prune_steps))
        final_tree = chosen_step.tree
        chosen_alpha = chosen_step.alpha

    candidate_map = {}
    evaluated = 0
    for frame in build_frames:
        if frame["event"] == "split":
            candidate_map[frame["node"]] = frame["candidates"]
            evaluated += frame["evaluated"]

    start = perf_counter()
    train_report, _, _ = _evaluate(final_tree, x_train, y_train, classes)
    val_report, _, _ = _evaluate(final_tree, x_val, y_val, classes)
    test_report, test_pred, _ = _evaluate(final_tree, x_test, y_test, classes)
    predict_ms = (perf_counter() - start) * 1000

    pca = PCA.fit(standardize(x))
    points = pca.transform(standardize(x))

    explanation = []
    chosen = []
    for label in range(len(classes)):
        members = [int(i) for i in test_idx if y[i] == label]
        if members:
            chosen.append(members[0])
    mistakes = [int(i) for i in test_idx if test_pred[list(test_idx).index(i)] != y[i]]
    if mistakes:
        chosen.append(mistakes[0])
    for sample in chosen[:4]:
        steps = cart.explain(final_tree, x[sample])
        for step in steps:
            if step.get("feature_name"):
                step["feature_label"] = FEATURE_LABELS.get(step["feature_name"], step["feature_name"])
        explanation.append({
            "index": sample,
            "row": x[sample].tolist(),
            "true_label": int(y[sample]),
            "predicted": int(steps[-1]["prediction"]),
            "steps": steps,
        })

    return {
        "dataset": "iris",
        "parameters": params,
        "split": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "sizes": [len(train_idx), len(val_idx), len(test_idx)],
        },
        "stats": {
            "nodes": len(final_tree.nodes),
            "leaves": len(cart.leaves(final_tree)),
            "depth": cart.depth(final_tree),
            "evaluated_thresholds": evaluated,
            "build_ms": build_ms,
            "predict_ms": predict_ms,
            "baseline_nodes": len(tree.nodes),
            "baseline_leaves": len(cart.leaves(tree)),
            "baseline_depth": cart.depth(tree),
        },
        "tree": _tree_payload(final_tree, candidate_map),
        "nodes": [
            _node_json(final_tree, node, candidate_map)
            for node in sorted(final_tree.nodes.values(), key=lambda n: n.id)
        ],
        "frames": _frames(build_frames, tree),
        "prune": {
            "enabled": params["ccp"],
            "chosen_alpha": chosen_alpha,
            "steps": [
                {
                    "alpha": step.alpha,
                    "pruned_node": step.pruned_node,
                    "leaves": step.leaves,
                    "depth": step.depth,
                    "train_accuracy": step.train_accuracy,
                    "val_accuracy": step.val_accuracy,
                    "nodes": len(step.tree.nodes),
                    "chosen": step is chosen_step,
                    "tree": _tree_payload(step.tree, candidate_map),
                }
                for step in prune_steps
            ],
        },
        "metrics": {
            "train": train_report,
            "val": val_report,
            "test": test_report,
        },
        "pca": {
            "points": points.tolist(),
            "labels": y.tolist(),
            "test_indices": test_idx.tolist(),
            "test_predictions": test_pred.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        },
        "explanation": explanation,
    }


def compare_variants(config):
    params = _settings(config)
    data = load_iris(DATASETS)
    classes = list(data.target_names)
    train_idx, val_idx, test_idx = stratified_split(data.labels, seed=params["seed"])
    x, y = data.features, data.labels
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    rows = []
    curve = []

    variants = [
        ("基线 CART", {"max_depth": None}),
        ("预剪枝", {"max_depth": 3, "min_samples_leaf": 2}),
    ]
    for name, override in variants:
        start = perf_counter()
        model, _ = cart.fit(
            x_train, y_train, feature_names=data.feature_names, class_names=classes,
            min_samples_leaf=override.get("min_samples_leaf", 1),
            max_depth=override["max_depth"], min_gain=params["min_gain"], record=False,
        )
        ms = (perf_counter() - start) * 1000
        rows.append({
            "name": name,
            "params": override,
            "depth": cart.depth(model),
            "leaves": len(cart.leaves(model)),
            "nodes": len(model.nodes),
            "train": _evaluate(model, x_train, y_train, classes)[0]["accuracy"],
            "val": _evaluate(model, x_val, y_val, classes)[0]["accuracy"],
            "test": _evaluate(model, x_test, y_test, classes)[0]["accuracy"],
            "ms": ms,
        })

    start = perf_counter()
    base, _ = cart.fit(x_train, y_train, feature_names=data.feature_names,
                       class_names=classes, min_gain=params["min_gain"], record=False)
    steps = pruning.prune_sequence(base, x_train, y_train, x_val, y_val)
    best = pruning.best_step(steps)
    ms = (perf_counter() - start) * 1000
    curve = [
        {"alpha": s.alpha, "leaves": s.leaves, "depth": s.depth,
         "train": s.train_accuracy, "val": s.val_accuracy,
         "nodes": len(s.tree.nodes)}
        for s in steps
    ]
    rows.append({
        "name": "后剪枝（CCP）",
        "params": {"alpha": best.alpha},
        "depth": best.depth,
        "leaves": best.leaves,
        "nodes": len(best.tree.nodes),
        "train": best.train_accuracy,
        "val": best.val_accuracy,
        "test": _evaluate(best.tree, x_test, y_test, classes)[0]["accuracy"],
        "ms": ms,
    })

    return {
        "dataset": "iris",
        "split_sizes": [len(train_idx), len(val_idx), len(test_idx)],
        "variants": rows,
        "curve": curve,
        "chosen_alpha": best.alpha,
    }
