# CARTLab · 决策树实验室

手写 CART 分类树 + Iris 数据集的离线桌面实验程序。实现 Gini 指数、基于排序扫描的阈值搜索、递归建树、代价复杂度后剪枝，并提供建树回放、候选划分、混淆矩阵、剪枝曲线与逐样本推理路径。

## 运行

```sh
uv sync --locked --group linux
uv run --locked --group linux python main.py
```

已有虚拟环境时：

```sh
.venv/bin/python main.py
```

Linux 使用 Qt WebEngine；Windows 使用 Edge WebView2。Windows 打包见 `scripts/build_windows.ps1`：`uv sync --locked --group dev` 后用 PyInstaller 生成 `dist\CARTLab\CARTLab.exe`，运行环境需已安装 WebView2 Runtime（Windows 11 自带）。

## 使用

1. 左栏选择参数：`max_depth` 留空表示不限制，`min_samples_leaf`、`min_gain` 控制停止条件，勾选「代价复杂度后剪枝」时用验证集选择 α。
2. 点击「运行训练」。中间为决策树，时间轴可回放建树过程；点击节点查看该节点的候选划分。
3. 右栏为指标、逐类精确率/召回率/F1、混淆矩阵、PCA 散点与测试样本推理路径。
4. 「运行对比」比较基线、预剪枝与后剪枝，并绘出 α 曲线。
5. 顶栏可切换明暗主题、导出树图 PNG 与完整结果 JSON。

## 验证

```sh
uv run --locked pytest -q
uv run --locked python scripts/run_experiments.py ./experiment_results
uv run --locked --group linux python scripts/build_report.py
```

测试覆盖 Gini 计算、阈值搜索与暴力枚举一致性、停止条件、推理路径、剪枝序列单调性、混淆矩阵与指标、数据规模与分层划分。

- [数据来源](datasets/SOURCES.md)
- [实验报告](docs/experiment_report.md)
