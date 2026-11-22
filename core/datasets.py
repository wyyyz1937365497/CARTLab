"""Iris 数据集读取与分层划分（纯 NumPy）。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
CLASSES = ("setosa", "versicolor", "virginica")
LABELS_ZH = ("山鸢尾", "变色鸢尾", "维吉尼亚鸢尾")


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    source: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray


def load_iris(root: Path) -> Dataset:
    """读取随程序分发的 sklearn Iris 数据副本。"""
    path = Path(root) / "iris.csv"
    rows = []
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        if header != [*FEATURES, "target"]:
            raise ValueError(f"iris.csv 列名异常：{header}")
        for number, line in enumerate(stream, start=2):
            line = line.strip()
            if not line:
                continue
            cells = line.split(",")
            if len(cells) != 5:
                raise ValueError(f"iris.csv 第 {number} 行列数错误")
            rows.append([float(c) for c in cells])
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (150, 5):
        raise ValueError("iris.csv 应为 150 行 5 列")
    labels = values[:, 4].astype(np.intp)
    if not np.array_equal(np.unique(labels), np.arange(3)):
        raise ValueError("iris.csv 标签应为 0/1/2")
    return Dataset(
        name="Iris",
        description="鸢尾花的四个形态特征与三个品种，共 150 个样本，每类 50 个。",
        source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
        feature_names=FEATURES,
        target_names=CLASSES,
        features=np.ascontiguousarray(values[:, :4]),
        labels=labels,
    )


def stratified_split(labels, ratios=(0.6, 0.2, 0.2), seed=42):
    """按类别分层打乱后切分，返回 train/val/test 下标。"""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("划分比例之和必须为 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        n = len(members)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        parts[0].append(members[:n_train])
        parts[1].append(members[n_train:n_train + n_val])
        parts[2].append(members[n_train + n_val:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)
