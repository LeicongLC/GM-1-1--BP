# GM(1,1)-BP Forecasting

基于灰色预测模型 GM(1,1) 与 BP 神经网络的年度数据预测项目。仓库同时保留原始 Jupyter Notebook，并提供一个可重复运行的 Python 命令行流程，便于复现实验、检查指标和生成图表。

## 项目结构

```text
.
├── data.csv                 # 输入数据：year、X1-X6、Y
├── GM-BP.ipynb              # 原始 notebook，保留用于学习和过程展示
├── src/
│   └── gm_bp_pipeline.py    # 可复现的 GM(1,1)-BP 流程
├── 结果/                    # 原始实验结果
├── results/                 # 脚本运行后生成的结果（默认被 git 忽略）
├── requirements.txt
└── .gitignore
```

## 环境准备

建议使用 Python 3.7 或更高版本，并安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

流程使用 `scikit-learn` 的 `MLPRegressor` 实现 BP 风格的反向传播网络，不再依赖 PyTorch，降低环境配置成本。当前依赖约束已按 Python 3.7 兼容范围设置。

## 快速运行

在仓库根目录执行：

```bash
python -m src.gm_bp_pipeline --data data.csv --output results
```

常用参数：

```bash
python -m src.gm_bp_pipeline \
  --data data.csv \
  --output results \
  --future-steps 10 \
  --test-size 0.3 \
  --random-state 42
```

流程会先按年份排序，再使用较早年份训练、较晚年份测试，避免随机打乱时间序列。GM(1,1) 用于外推 `X1`-`X6` 的未来值，BP 网络使用这些特征预测目标列 `Y`。

## 输出文件

脚本会在 `results/` 中生成：

- `metrics.csv`：MSE、RMSE、MAE、R² 和解释方差
- `test_predictions.csv`：测试集真实值、预测值和绝对误差
- `gm_feature_forecast.csv`：GM(1,1) 对各输入特征的未来预测
- `predicted_values.csv`：未来年份的 `Y` 预测
- `loss.png`：BP 训练损失曲线
- `actual_vs_predicted.png`：测试集真实值与预测值对比
- `future_forecast.png`：未来预测曲线

## 数据格式

输入 CSV 必须包含以下列：

```text
year,X1,X2,X3,X4,X5,X6,Y
```

GM(1,1) 要求用于外推的 `X1`-`X6` 为正数，数据不能在必需列中包含缺失值。

## Notebook

`GM-BP.ipynb` 是原始实验 notebook，适合逐单元查看数据处理、模型训练和可视化过程。推荐使用 `src/gm_bp_pipeline.py` 作为批量运行和结果复现入口。
