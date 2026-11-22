from pathlib import Path

import numpy as np
import pytest

from core.datasets import load_iris, stratified_split

ROOT = Path(__file__).resolve().parents[1] / "datasets"


def test_iris_shape_and_balance():
    data = load_iris(ROOT)
    assert data.features.shape == (150, 4)
    assert np.bincount(data.labels).tolist() == [50, 50, 50]
    assert data.feature_names == ("sepal_length", "sepal_width", "petal_length", "petal_width")
    assert np.allclose(data.features[0], [5.1, 3.5, 1.4, 0.2])


def test_stratified_split_sizes_and_disjoint():
    data = load_iris(ROOT)
    train, val, test = stratified_split(data.labels, seed=7)
    assert (len(train), len(val), len(test)) == (90, 30, 30)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(150))
    for part in (train, val, test):
        assert np.bincount(data.labels[part], minlength=3).tolist() == [len(part) // 3] * 3


def test_split_is_reproducible_and_seed_dependent():
    data = load_iris(ROOT)
    a = stratified_split(data.labels, seed=42)
    b = stratified_split(data.labels, seed=42)
    c = stratified_split(data.labels, seed=43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_matches_sklearn_when_available():
    sklearn = pytest.importorskip("sklearn.datasets")
    data = load_iris(ROOT)
    bunch = sklearn.load_iris()
    np.testing.assert_allclose(data.features, bunch.data)
    np.testing.assert_array_equal(data.labels, bunch.target)
    assert list(data.target_names) == list(bunch.target_names)
