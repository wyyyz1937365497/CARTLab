# 数据来源

- Iris 数据集：R. A. Fisher (1936)，150 个样本、4 个特征、3 个类别。
  scikit-learn 通过 `sklearn.datasets.load_iris()` 提供该数据（BSD-3-Clause）。
  本项目的 `iris.csv` 取自 scikit-learn 仓库 `sklearn/datasets/data/iris.csv`，
  转换为「特征名 + target」的 CSV 表头，数值与样本顺序保持不变。
- 校验：`tests/test_data.py::test_matches_sklearn_when_available` 在安装 scikit-learn 时
  逐元素比较特征与标签；未安装时跳过。

引用：Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
Annals of Eugenics, 7(2), 179–188.
