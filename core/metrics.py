"""基于混淆矩阵的评价指标。"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.intp)
    y_pred = np.asarray(y_pred, dtype=np.intp)
    if y_true.shape != y_pred.shape:
        raise ValueError("真实标签与预测标签长度不一致")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(cm, class_names):
    """由混淆矩阵计算每类 precision / recall / F1 与宏平均。"""
    cm = np.asarray(cm)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    per_class = []
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        support = int(cm[c].sum())
        predicted = int(cm[:, c].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    macro = {
        "precision": float(np.mean([c["precision"] for c in per_class])),
        "recall": float(np.mean([c["recall"] for c in per_class])),
        "f1": float(np.mean([c["f1"] for c in per_class])),
    }
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "classes": per_class,
        "macro": macro,
        "matrix": cm.tolist(),
    }
