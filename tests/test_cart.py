import numpy as np
import pytest

from algorithms import cart


def naive_best(x, y, n_classes, min_leaf=1):
    best = None
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        for a, b in zip(values[:-1], values[1:]):
            threshold = (a + b) / 2
            left = y[x[:, feature] <= threshold]
            right = y[x[:, feature] > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            weighted = (len(left) * cart.gini(np.bincount(left, minlength=n_classes)) +
                        len(right) * cart.gini(np.bincount(right, minlength=n_classes))) / len(y)
            if best is None or weighted < best[0] - 1e-12:
                best = (weighted, feature, threshold)
    return best


def test_gini_extremes():
    assert cart.gini([10, 0, 0]) == 0
    assert cart.gini([5, 5]) == pytest.approx(0.5)
    assert cart.gini([3, 3, 3]) == pytest.approx(2 / 3)


def test_split_search_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 3))
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    split, top, evaluated = cart.candidate_splits(x, y, 3, min_samples_leaf=2)
    weighted, feature, threshold = naive_best(x, y, 3, 2)
    assert split.feature == feature
    assert split.threshold == pytest.approx(threshold)
    assert split.weighted_gini == pytest.approx(weighted)
    assert split.n_left + split.n_right == 60
    assert evaluated > 0 and len(top) <= 8
    assert all(c.n_left >= 2 and c.n_right >= 2 for c in top)


def test_thresholds_lie_between_distinct_values_and_gain_nonnegative():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    split, _, _ = cart.candidate_splits(x, y, 2)
    values = np.unique(x[:, split.feature])
    assert any(abs(split.threshold - (a + b) / 2) < 1e-12 for a, b in zip(values[:-1], values[1:]))
    assert split.gain >= -1e-12


def test_tree_respects_limits_and_grows_until_pure():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    tree, frames = cart.fit(x, y, max_depth=2, record=True)
    assert cart.depth(tree) <= 3
    assert all(node.depth <= 2 for node in tree.nodes.values())
    assert all(node.is_leaf or node.n_samples >= node.counts[0] for node in tree.nodes.values())
    assert frames[0]["event"] in ("split", "leaf")
    unlimited, _ = cart.fit(x, y, record=False)
    labels, _ = cart.predict(unlimited, x)
    assert np.mean(labels == y) == 1.0


def test_min_samples_leaf_is_respected():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.array([0] * 10 + [1] * 10)
    tree, _ = cart.fit(x, y, min_samples_leaf=4, record=False)
    for node in tree.nodes.values():
        if node.is_leaf:
            assert node.n_samples >= 4


def test_predict_and_explain_paths_are_consistent():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(90, 5))
    y = rng.integers(0, 3, size=90)
    tree, _ = cart.fit(x, y, max_depth=4, record=False)
    labels, paths = cart.predict(tree, x)
    for row, label, path in zip(x, labels, paths):
        steps = cart.explain(tree, row)
        assert [s["node"] for s in steps] == path
        assert steps[-1]["prediction"] == label
    assert all(tree.nodes[p].is_leaf for p in paths[-1:][0][-1:])


@pytest.mark.parametrize("params", [{"max_depth": -1}, {"min_samples_leaf": 0}, {"min_gain": -1}])
def test_invalid_params(params):
    with pytest.raises(ValueError):
        cart.fit(np.zeros((4, 2)), np.zeros(4, dtype=int), **params)
