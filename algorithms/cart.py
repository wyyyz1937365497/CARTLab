"""手写 CART 分类树：Gini 指数、排序扫描的阈值搜索、递归建树与路径记录。

特点：
- 每个特征只排序一次，用类别计数的前缀和增量计算左右不纯度，
  单节点复杂度 O(F·n log n + F·n·K)。
- 记录每个节点评估过的候选划分（按加权 Gini 排序取前若干），
  以及建树过程的帧序列，供界面回放。
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


def gini(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.square(p).sum())


@dataclass(frozen=True)
class Split:
    feature: int
    threshold: float
    weighted_gini: float
    gain: float
    n_left: int
    n_right: int
    left_counts: tuple[int, ...]
    right_counts: tuple[int, ...]


@dataclass(frozen=True)
class Node:
    id: int
    depth: int
    n_samples: int
    counts: tuple[int, ...]
    gini: float
    prediction: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[int] = None
    right: Optional[int] = None
    reason: Optional[str] = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


@dataclass(frozen=True)
class Tree:
    nodes: dict
    root: int
    feature_names: tuple
    class_names: tuple
    n_features: int
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.root not in self.nodes:
            raise ValueError("根节点缺失")


def _validate(x, y, params):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.intp)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("特征矩阵与标签不匹配")
    if not np.isfinite(x).all():
        raise ValueError("特征包含 NaN 或无穷值")
    max_depth = params["max_depth"]
    if max_depth is not None:
        if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError("max_depth 必须为非负整数或 None")
    for key in ("min_samples_split", "min_samples_leaf"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} 必须为正整数")
    if params["min_samples_leaf"] * 2 > x.shape[0]:
        raise ValueError("min_samples_leaf 过大，无法继续分裂")
    if not np.isfinite(params["min_gain"]) or params["min_gain"] < 0:
        raise ValueError("min_gain 必须为非负数")
    return x, y


def candidate_splits(x, y, n_classes, min_samples_leaf=1, keep=8):
    """返回 (最佳划分, 排序后的候选列表, 评估的阈值数)。"""
    n = len(y)
    counts = np.bincount(y, minlength=n_classes).astype(np.int64)
    parent_gini = gini(counts)
    candidates = []
    evaluated = 0
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        values = x[order, feature]
        labels = y[order]
        left = np.zeros(n_classes, dtype=np.int64)
        for i in range(n - 1):
            left[labels[i]] += 1
            if values[i] == values[i + 1]:
                continue
            n_left = i + 1
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            right = counts - left
            weighted = (n_left * gini(left) + n_right * gini(right)) / n
            evaluated += 1
            candidates.append(Split(
                feature=feature,
                threshold=float((values[i] + values[i + 1]) / 2.0),
                weighted_gini=float(weighted),
                gain=float(parent_gini - weighted),
                n_left=n_left,
                n_right=n_right,
                left_counts=tuple(int(v) for v in left),
                right_counts=tuple(int(v) for v in right),
            ))
    if not candidates:
        return None, [], evaluated
    candidates.sort(key=lambda s: (s.weighted_gini, s.feature, s.threshold))
    return candidates[0], candidates[:keep], evaluated


def fit(x, y, *, feature_names=None, class_names=None,
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_gain=0.0, record=True):
    """训练 CART；record=True 时返回建树帧序列。"""
    params = {
        "max_depth": max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "min_gain": float(min_gain),
    }
    x, y = _validate(x, y, params)
    names = tuple(feature_names or [f"x{i}" for i in range(x.shape[1])])
    classes = tuple(class_names or [str(i) for i in range(int(y.max()) + 1)])
    n_classes = len(classes)
    nodes: dict[int, Node] = {}
    frames = []

    def snapshot():
        def build(node_id):
            node = nodes[node_id]
            out = {
                "id": node.id,
                "depth": node.depth,
                "n": node.n_samples,
                "counts": list(node.counts),
                "gini": node.gini,
                "prediction": node.prediction,
            }
            if node.is_leaf:
                out["reason"] = node.reason
            else:
                out["feature"] = node.feature
                out["threshold"] = node.threshold
                out["gain"] = node.gain
                if node.left is None or node.right is None:
                    out["pending"] = True
                else:
                    out["children"] = [build(node.left), build(node.right)]
            return out
        return build(0)

    def add(node: Node):
        nodes[node.id] = node
        return node.id

    def frame(event, node, candidates=None, evaluated=0):
        frames.append({
            "event": event,
            "node": node.id,
            "depth": node.depth,
            "n": node.n_samples,
            "counts": list(node.counts),
            "gini": node.gini,
            "prediction": node.prediction,
            "feature": node.feature,
            "threshold": node.threshold,
            "gain": node.gain,
            "reason": node.reason,
            "evaluated": evaluated,
            "candidates": [
                {"feature": c.feature, "threshold": c.threshold, "weighted_gini": c.weighted_gini,
                 "gain": c.gain, "n_left": c.n_left, "n_right": c.n_right}
                for c in (candidates or [])
            ],
            "tree": snapshot(),
        })

    def grow(indices, depth, node_id):
        labels = y[indices]
        counts = np.bincount(labels, minlength=n_classes)
        prediction = int(np.argmax(counts))
        node = Node(node_id, depth, len(indices), tuple(int(v) for v in counts),
                    gini(counts), prediction)
        add(node)
        if np.count_nonzero(counts) == 1:
            node = replace(node, reason="pure")
        elif params["max_depth"] is not None and depth >= params["max_depth"]:
            node = replace(node, reason="max_depth")
        elif len(indices) < params["min_samples_split"]:
            node = replace(node, reason="min_samples_split")
        else:
            best, top, evaluated = candidate_splits(
                x[indices], labels, n_classes, params["min_samples_leaf"])
            if best is None:
                node = replace(node, reason="no_valid_split")
            elif best.gain <= params["min_gain"] + 1e-12:
                node = replace(node, reason="min_gain")
            else:
                split = best
                left_mask = x[indices, split.feature] <= split.threshold
                node = replace(node, feature=split.feature, threshold=split.threshold,
                               gain=split.gain)
                nodes[node_id] = node
                if record:
                    frame("split", node, top, evaluated)
                left_id = grow(indices[left_mask], depth + 1, 2 * node_id + 1)
                right_id = grow(indices[~left_mask], depth + 1, 2 * node_id + 2)
                node = replace(nodes[node_id], left=left_id, right=right_id)
                nodes[node_id] = node
                if record:
                    frame("expanded", node, top, evaluated)
                return node_id
        nodes[node_id] = node
        if record:
            frame("leaf", node)
        return node_id

    grow(np.arange(x.shape[0]), 0, 0)
    tree = Tree(nodes=dict(nodes), root=0, feature_names=names,
                class_names=classes, n_features=x.shape[1], params=params)
    return tree, frames


def predict(tree: Tree, x):
    x = np.asarray(x, dtype=np.float64)
    labels = np.empty(len(x), dtype=np.intp)
    paths = []
    for i, row in enumerate(x):
        node = tree.nodes[tree.root]
        path = [node.id]
        while not node.is_leaf:
            node = tree.nodes[node.left if row[node.feature] <= node.threshold else node.right]
            path.append(node.id)
        labels[i] = node.prediction
        paths.append(path)
    return labels, paths


def explain(tree: Tree, row) -> list:
    """返回单样本的逐节点推理步骤。"""
    row = np.asarray(row, dtype=np.float64)
    node = tree.nodes[tree.root]
    steps = []
    while True:
        step = {
            "node": node.id,
            "depth": node.depth,
            "gini": node.gini,
            "n": node.n_samples,
            "counts": list(node.counts),
            "prediction": node.prediction,
        }
        if node.is_leaf:
            step["leaf"] = True
            steps.append(step)
            return steps
        go_left = bool(row[node.feature] <= node.threshold)
        step.update({
            "feature": node.feature,
            "feature_name": tree.feature_names[node.feature],
            "threshold": node.threshold,
            "value": float(row[node.feature]),
            "go_left": go_left,
            "gain": node.gain,
        })
        steps.append(step)
        node = tree.nodes[node.left if go_left else node.right]


def depth(tree: Tree) -> int:
    def walk(node_id):
        node = tree.nodes[node_id]
        if node.is_leaf:
            return 1
        return 1 + max(walk(node.left), walk(node.right))
    return walk(tree.root)


def leaves(tree: Tree) -> list:
    return [n.id for n in tree.nodes.values() if n.is_leaf]
