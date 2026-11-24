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


# α 序列中的一步：剪枝后得到的树、被剪的子树根、叶数、深度，以及训练 / 验证准确率
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
    """子树风险 R(T)：各叶节点按样本比例加权的 Gini 之和。"""
    node = tree.nodes[node_id]
    if node.is_leaf:
        return node.n_samples / n_total * node.gini
    return _risk(tree, node.left, n_total) + _risk(tree, node.right, n_total)


def _leaves(tree: Tree, node_id: int) -> int:
    """子树里的叶节点数 |T|。"""
    node = tree.nodes[node_id]
    if node.is_leaf:
        return 1
    return _leaves(tree, node.left) + _leaves(tree, node.right)


def prune_at(tree: Tree, node_id: int) -> Tree:
    """把 node_id 的子树整体替换为叶节点。"""
    node = tree.nodes[node_id]
    # 复制一份节点表，删掉 node_id 的所有后代
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
    # node_id 自己变成叶节点：保留样本数 / 类别计数 / 多数类，原因标为 pruned
    nodes[node_id] = type(node)(
        id=node.id, depth=node.depth, n_samples=node.n_samples,
        counts=node.counts, gini=node.gini, prediction=node.prediction,
        reason="pruned",
    )
    return Tree(nodes=nodes, root=tree.root, feature_names=tree.feature_names,
                class_names=tree.class_names, n_features=tree.n_features,
                params=dict(tree.params, pruned=True))


def prune_sequence(tree: Tree, x_train, y_train, x_val, y_val):
    """返回完整的 α 序列，首项为未剪枝树。

    循环直到只剩根节点：
    1. 对每个内部节点 t 算弱链接系数 α_eff(t) = (R(t) − R(T_t)) / (|T_t| − 1)
       —— 即「合并这棵子树，平均每少一个叶节点要多付多少风险」；
    2. 剪掉 α_eff 最小的子树（最便宜的那棵），α 取历史最大值保证单调不减；
    3. 在训练集 / 验证集上评估剪枝后的树，存成一步。
    """
    n_total = tree.nodes[tree.root].n_samples
    current = tree
    steps = []
    alpha = 0.0

    # 用训练集和验证集分别算准确率（测试集不参与选 α）
    def evaluate(model):
        train = float(np.mean(cart.predict(model, x_train)[0] == y_train))
        val = float(np.mean(cart.predict(model, x_val)[0] == y_val))
        return train, val

    # 第 0 步：未剪枝的原始树
    train_acc, val_acc = evaluate(current)
    steps.append(PruneStep(0.0, current, None, len(cart.leaves(current)),
                            cart.depth(current), train_acc, val_acc))
    # 每轮剪一棵子树，直到没有内部节点
    while True:
        internal = [n.id for n in current.nodes.values() if not n.is_leaf]
        if not internal:
            break
        # 对所有内部节点算 α_eff
        scores = []
        for node_id in internal:
            subtree_leaves = _leaves(current, node_id)
            # 只有 1 个叶的子树不需要剪
            if subtree_leaves < 2:
                continue
            # R(t)：把 t 直接当叶节点时的风险
            node_risk = current.nodes[node_id].n_samples / n_total * current.nodes[node_id].gini
            # R(T_t)：保留 t 下面整棵子树的风险
            sub_risk = _risk(current, node_id, n_total)
            # α_eff = 风险增加量 / 少掉的叶节点数
            value = (node_risk - sub_risk) / (subtree_leaves - 1)
            scores.append((value, node_id))
        if not scores:
            break
        # 选 α_eff 最小的；平局取节点编号小的
        value, node_id = min(scores, key=lambda item: (item[0], item[1]))
        # α 序列单调不减
        alpha = max(alpha, value)
        current = prune_at(current, node_id)
        # 评估剪枝后的树并记录这一步
        train_acc, val_acc = evaluate(current)
        steps.append(PruneStep(alpha, current, node_id, len(cart.leaves(current)),
                                cart.depth(current), train_acc, val_acc))
    return steps


def best_step(steps, key="val_accuracy"):
    """在 α 序列中选验证集表现最好的树；平局取叶节点更少者（key 可换成别的指标）。"""
    return max(steps, key=lambda s: (s.__getattribute__(key), -s.leaves))


def select_alpha(steps, alpha: float):
    """选择 α 不超过给定值且最大的那棵树。"""
    chosen = steps[0]
    for step in steps:
        if step.alpha <= alpha + 1e-12:
            chosen = step
    return chosen
