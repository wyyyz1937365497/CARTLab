"""代价复杂度后剪枝（CCP），手写实现。

风险定义：R(T) = Σ_leaf (n_leaf / n_root) · Gini(leaf)
弱链接：α_eff(t) = (R(t) − R(T_t)) / (|T_t| − 1)
每轮剪掉 α 最小的子树，得到一串 α 递增的树。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from algorithms import cart
from algorithms.cart import Tree


@dataclass(frozen=True)
class PruneStep:
    alpha: float
    tree: Tree
    pruned_node: Optional[int]
    leaves: int
    depth: int
    train_accuracy: float
    val_accuracy: float


def _risk(tree: Tree, node_id: int, n_total: int) -> float:
    node = tree.nodes[node_id]
    if node.is_leaf:
        return node.n_samples / n_total * node.gini
    return _risk(tree, node.left, n_total) + _risk(tree, node.right, n_total)


def _leaves(tree: Tree, node_id: int) -> int:
    node = tree.nodes[node_id]
    if node.is_leaf:
        return 1
    return _leaves(tree, node.left) + _leaves(tree, node.right)


def prune_at(tree: Tree, node_id: int) -> Tree:
    """把 node_id 的子树整体替换为叶节点。"""
    node = tree.nodes[node_id]
    nodes = dict(tree.nodes)
    descendants = set()
    stack = [node.left, node.right] if not node.is_leaf else []
    while stack:
        current = stack.pop()
        descendants.add(current)
        item = tree.nodes[current]
        if not item.is_leaf:
            stack.extend([item.left, item.right])
    for key in descendants:
        del nodes[key]
    nodes[node_id] = type(node)(
        id=node.id, depth=node.depth, n_samples=node.n_samples,
        counts=node.counts, gini=node.gini, prediction=node.prediction,
        reason="pruned",
    )
    return Tree(nodes=nodes, root=tree.root, feature_names=tree.feature_names,
                class_names=tree.class_names, n_features=tree.n_features,
                params=dict(tree.params, pruned=True))


def prune_sequence(tree: Tree, x_train, y_train, x_val, y_val):
    """返回完整的 α 序列，首项为未剪枝树。"""
    n_total = tree.nodes[tree.root].n_samples
    current = tree
    steps = []
    alpha = 0.0

    def evaluate(model):
        train = float(np.mean(cart.predict(model, x_train)[0] == y_train))
        val = float(np.mean(cart.predict(model, x_val)[0] == y_val))
        return train, val

    train_acc, val_acc = evaluate(current)
    steps.append(PruneStep(0.0, current, None, len(cart.leaves(current)),
                            cart.depth(current), train_acc, val_acc))
    while True:
        internal = [n.id for n in current.nodes.values() if not n.is_leaf]
        if not internal:
            break
        scores = []
        for node_id in internal:
            subtree_leaves = _leaves(current, node_id)
            if subtree_leaves < 2:
                continue
            node_risk = current.nodes[node_id].n_samples / n_total * current.nodes[node_id].gini
            sub_risk = _risk(current, node_id, n_total)
            value = (node_risk - sub_risk) / (subtree_leaves - 1)
            scores.append((value, node_id))
        if not scores:
            break
        value, node_id = min(scores, key=lambda item: (item[0], item[1]))
        alpha = max(alpha, value)
        current = prune_at(current, node_id)
        train_acc, val_acc = evaluate(current)
        steps.append(PruneStep(alpha, current, node_id, len(cart.leaves(current)),
                                cart.depth(current), train_acc, val_acc))
    return steps


def best_step(steps, key="val_accuracy"):
    """在 α 序列中选验证集表现最好的树；平局取叶节点更少者。"""
    return max(steps, key=lambda s: (s.__getattribute__(key), -s.leaves))


def select_alpha(steps, alpha: float):
    """选择 α 不超过给定值且最大的那棵树。"""
    chosen = steps[0]
    for step in steps:
        if step.alpha <= alpha + 1e-12:
            chosen = step
    return chosen
