# 机器学习项目：数据预处理与GM-BP模型

## 项目概述

此项目包括数据的预处理、使用GM(1,1)和BP神经网络模型进行数据分析和预测，以及结果的可视化展示。项目旨在展示如何结合传统的机器学习方法和深度学习技术来进行时间序列预测。

## 环境准备

本项目使用Python进行开发，需要安装以下库：

- pandas
- sklearn
- torch
- matplotlib
- numpy

您可以使用pip安装上述依赖：

```bash
pip install pandas scikit-learn torch matplotlib numpy
```

## 文件结构

- `data.csv`：原始数据文件
- `1.1.ipynb`：主要的Jupyter笔记本，包含项目代码

## 如何运行

1. 确保您已安装所有依赖。
2. 将数据文件`data.csv`放在与笔记本相同的目录下。
3. 通过Jupyter Notebook或JupyterLab打开`GM-BP.ipynb`。
4. 按顺序运行笔记本中的单元格。

## 项目细节

### 数据预处理

项目从加载数据集开始，包括数据清洗、特征标准化，并使用`train_test_split`函数划分数据集。

### 模型定义

- **GM(1,1)模型**：用于预测单个特征列的未来值。
- **BP神经网络模型**：一个深度学习模型，用于根据输入特征预测输出。

### 训练与评估

使用PyTorch框架实例化BP模型，训练模型，并评估其性能。性能评估包括MSE、RMSE、MAE、R²分数和解释方差分数。

### 结果可视化

项目使用matplotlib库绘制训练过程中的损失下降图、真实值与预测值的散点图，以及未来值的预测折线图。
