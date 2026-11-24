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
    """Gini 不纯度 = 1 − Σ pₖ²；counts 是各类样本数，pₖ 为占比。越小说明节点越纯。"""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.square(p).sum())


# 一次候选划分：按哪个特征、在哪个阈值切开，以及左右两侧的样本数和类别计数
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


# 树的一个节点：样本数、类别计数、Gini、多数类预测；内部节点再记录划分方式和左右孩子
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


# 整棵树：用字典按编号存所有节点；孩子编号为 2i+1、2i+2
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
    """校验特征 / 标签形状与超参数，返回 (x, y)。"""
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
    """在一个节点内搜索最佳划分，返回 (最佳划分, 排序后的候选列表, 评估的阈值数)。

    对每个特征：
    1. 按该特征的取值排序；
    2. 从左往右扫，left 始终是「左侧已扫过样本」的类别计数（前缀和）；
    3. 只在相邻两个取值不同处切：阈值取两者中点，相同的取值不能被切开；
    4. 右侧计数 = 总计数 − 左侧计数；算加权 Gini 与增益。

    排序键为 (加权 Gini, 特征下标, 阈值)，所以平局时结果确定。
    """
    n = len(y)
    counts = np.bincount(y, minlength=n_classes).astype(np.int64)
    parent_gini = gini(counts)
    candidates = []
    evaluated = 0
    # 每个特征只排序一次
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        values = x[order, feature]
        labels = y[order]
        # 左半边的类别计数，扫描时逐个累加（前缀和）
        left = np.zeros(n_classes, dtype=np.int64)
        for i in range(n - 1):
            # 把第 i 个样本划入左侧
            left[labels[i]] += 1
            # 相同取值之间不切：切在这里左右分组不变，没有意义
            if values[i] == values[i + 1]:
                continue
            n_left = i + 1
            n_right = n - n_left
            # 任一侧样本太少则跳过这个切点
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            # 右侧计数用减法得到，不用重新统计
            right = counts - left
            # 加权 Gini = (n左·Gini左 + n右·Gini右) / n；增益 = 父节点 Gini − 加权 Gini
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
    # 加权 Gini 越小越好；平局依次比特征下标、阈值
    candidates.sort(key=lambda s: (s.weighted_gini, s.feature, s.threshold))
    return candidates[0], candidates[:keep], evaluated


def fit(x, y, *, feature_names=None, class_names=None,
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_gain=0.0, record=True):
    """训练 CART：递归建树；record=True 时同时返回建树帧序列。

    grow() 对当前节点：
    1. 统计类别计数、Gini 和多数类；
    2. 判断是否停止（纯净 / 达到 max_depth / 样本太少 / 无合法划分 / 增益不足）→ 叶节点；
    3. 否则选最佳划分，按 x ≤ 阈值 分成左右两组，先递归左子树、再递归右子树（深度优先）。
    """
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

    # 导出当前树结构的快照；只有一个孩子的节点标为 pending（构建中）
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
                children = [c for c in (node.left, node.right) if c is not None]
                if children:
                    out["children"] = [build(c) for c in children]
                if len(children) < 2:
                    out["pending"] = True
            return out
        return build(0)

    def add(node: Node):
        nodes[node.id] = node
        return node.id

    # 记一帧：当前节点信息、评估过的候选划分、以及整棵树的快照
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

    # 递归建树：grow(样本下标, 深度, 节点编号)
    def grow(indices, depth, node_id):
        labels = y[indices]
        counts = np.bincount(labels, minlength=n_classes)
        prediction = int(np.argmax(counts))
        node = Node(node_id, depth, len(indices), tuple(int(v) for v in counts),
                    gini(counts), prediction)
        add(node)
        # 停止条件 1：节点纯净（只有一类）
        if np.count_nonzero(counts) == 1:
            node = replace(node, reason="pure")
        # 停止条件 2：达到最大深度
        elif params["max_depth"] is not None and depth >= params["max_depth"]:
            node = replace(node, reason="max_depth")
        # 停止条件 3：样本数不足 min_samples_split
        elif len(indices) < params["min_samples_split"]:
            node = replace(node, reason="min_samples_split")
        else:
            # 搜索最佳划分
            best, top, evaluated = candidate_splits(
                x[indices], labels, n_classes, params["min_samples_leaf"])
            # 停止条件 4：没有合法切点
            if best is None:
                node = replace(node, reason="no_valid_split")
            # 停止条件 5：增益太小
            elif best.gain <= params["min_gain"] + 1e-12:
                node = replace(node, reason="min_gain")
            else:
                split = best
                # x ≤ 阈值 走左边，否则走右边
                left_mask = x[indices, split.feature] <= split.threshold
                node = replace(node, feature=split.feature, threshold=split.threshold,
                               gain=split.gain)
                nodes[node_id] = node
                # 「选中划分」帧
                if record:
                    frame("split", node, top, evaluated)
                # 孩子编号：左 2i+1，右 2i+2
                left_id = 2 * node_id + 1
                right_id = 2 * node_id + 2
                nodes[node_id] = replace(nodes[node_id], left=left_id)
                # 先递归左子树
                grow(indices[left_mask], depth + 1, left_id)
                # 左子树建完，挂到父节点上，记一帧
                if record:
                    frame("expanded", nodes[node_id], top, evaluated)
                nodes[node_id] = replace(nodes[node_id], left=left_id, right=right_id)
                # 再递归右子树
                grow(indices[~left_mask], depth + 1, right_id)
                node = nodes[node_id]
                if record:
                    frame("expanded", node, top, evaluated)
                return node_id
        # 叶节点：写入并记「标记叶节点」帧
        nodes[node_id] = node
        if record:
            frame("leaf", node)
        return node_id

    # 从根节点（编号 0、深度 0、全部样本）开始
    grow(np.arange(x.shape[0]), 0, 0)
    tree = Tree(nodes=dict(nodes), root=0, feature_names=names,
                class_names=classes, n_features=x.shape[1], params=params)
    return tree, frames


def predict(tree: Tree, x):
    """从根到叶逐个比较 feature ≤ threshold，返回 (预测标签, 每个样本经过的节点编号列表)。"""
    x = np.asarray(x, dtype=np.float64)
    labels = np.empty(len(x), dtype=np.intp)
    paths = []
    for i, row in enumerate(x):
        node = tree.nodes[tree.root]
        path = [node.id]
        # 沿树往下走，直到叶节点
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
    """树的层数：根为 1。"""
    def walk(node_id):
        node = tree.nodes[node_id]
        if node.is_leaf:
            return 1
        return 1 + max(walk(node.left), walk(node.right))
    return walk(tree.root)


def leaves(tree: Tree) -> list:
    """所有叶节点的编号。"""
    return [n.id for n in tree.nodes.values() if n.is_leaf]
