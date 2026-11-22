"""复现报告中的多随机种子对比。

用法：
    uv run --locked python scripts/run_experiments.py ./experiment_results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms import cart, pruning
from app.services import compare_variants, run_training
from core.datasets import load_iris, stratified_split
from core.metrics import classification_report, confusion_matrix

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def evaluate(tree, x, y, classes):
    pred, _ = cart.predict(tree, x)
    return classification_report(confusion_matrix(y, pred, len(classes)), classes)["accuracy"]


def sweep(seeds=20):
    data = load_iris(DATASETS)
    classes = list(data.target_names)
    rows = []
    for seed in range(seeds):
        train, val, test = stratified_split(data.labels, seed=seed)
        x, y = data.features, data.labels
        xt, yt = x[train], y[train]
        xv, yv = x[val], y[val]
        xs, ys = x[test], y[test]

        base, _ = cart.fit(xt, yt, feature_names=data.feature_names, class_names=classes, record=False)
        pre, _ = cart.fit(xt, yt, feature_names=data.feature_names, class_names=classes,
                          max_depth=3, min_samples_leaf=2, record=False)
        steps = pruning.prune_sequence(base, xt, yt, xv, yv)
        best = pruning.best_step(steps)

        rows.append({
            "seed": seed,
            "baseline": {"leaves": len(cart.leaves(base)), "depth": cart.depth(base),
                          "train": evaluate(base, xt, yt, classes),
                          "val": evaluate(base, xv, yv, classes),
                          "test": evaluate(base, xs, ys, classes)},
            "pre": {"leaves": len(cart.leaves(pre)), "depth": cart.depth(pre),
                    "train": evaluate(pre, xt, yt, classes),
                    "val": evaluate(pre, xv, yv, classes),
                    "test": evaluate(pre, xs, ys, classes)},
            "ccp": {"leaves": best.leaves, "depth": best.depth, "alpha": best.alpha,
                    "train": best.train_accuracy, "val": best.val_accuracy,
                    "test": evaluate(best.tree, xs, ys, classes)},
        })
    summary = {}
    for key in ("baseline", "pre", "ccp"):
        for metric in ("train", "val", "test"):
            values = [r[key][metric] for r in rows]
            summary[f"{key}_{metric}"] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                           "min": float(np.min(values)), "max": float(np.max(values))}
        summary[f"{key}_leaves"] = {"mean": float(np.mean([r[key]["leaves"] for r in rows]))}
    return {"runs": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sweep_result = sweep(args.seeds)
    (args.output / "seed_sweep.json").write_text(
        json.dumps(sweep_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    demo = run_training({"max_depth": 4, "ccp": True, "seed": 42})
    (args.output / "demo_run.json").write_text(
        json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compare = compare_variants({"seed": 42})
    (args.output / "compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(sweep_result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
