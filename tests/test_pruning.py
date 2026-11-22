import numpy as np

from algorithms import cart, pruning
from core.metrics import classification_report, confusion_matrix


def _tree():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 4))
    y = np.where(x[:, 0] > 0.2, 2, np.where(x[:, 1] > 0, 1, 0))
    return x, y, cart.fit(x, y, max_depth=5, record=False)[0]


def test_prune_sequence_is_increasing_and_shrinks():
    x, y, tree = _tree()
    steps = pruning.prune_sequence(tree, x[:80], y[:80], x[80:], y[80:])
    assert steps[0].alpha == 0
    assert steps[0].leaves == len(cart.leaves(tree))
    assert steps[-1].leaves == 1
    assert all(a.alpha <= b.alpha for a, b in zip(steps, steps[1:]))
    assert all(a.leaves >= b.leaves for a, b in zip(steps, steps[1:]))
    assert steps[-1].tree.root in steps[-1].tree.nodes


def test_prune_at_removes_descendants_only():
    _, _, tree = _tree()
    parent = next(n.id for n in tree.nodes.values() if not n.is_leaf)
    pruned = pruning.prune_at(tree, parent)
    assert pruned.nodes[parent].is_leaf
    assert tree.nodes[parent].is_leaf is False
    assert len(pruned.nodes) < len(tree.nodes)
    for node in pruned.nodes.values():
        assert node.id in pruned.nodes


def test_best_step_prefers_validation_then_fewer_leaves():
    x, y, tree = _tree()
    steps = pruning.prune_sequence(tree, x[:80], y[:80], x[80:], y[80:])
    best = pruning.best_step(steps)
    assert best.val_accuracy >= max(s.val_accuracy for s in steps) - 1e-12
    chosen = pruning.select_alpha(steps, best.alpha)
    assert chosen.alpha <= best.alpha + 1e-12
    assert chosen.leaves >= best.leaves


def test_confusion_matrix_and_report():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    report = classification_report(cm, ["a", "b", "c"])
    assert report["accuracy"] == 4 / 6
    assert report["classes"][0]["recall"] == 0.5
    assert report["classes"][1]["precision"] == 2 / 3
    assert report["macro"]["recall"] == (0.5 + 1.0 + 0.5) / 3
