# ai-models-and-algorithms-final-25f

本仓库包含了期终试卷的第八题（强化学习）和第九题（回归预测）的解题代码。

## 目录结构

*   `8-rl/`: 第八题 - 迷宫求解 (Reinforcement Learning)
*   `9-regression/`: 第九题 - 回归模型预测 (Regression Analysis)

---

## 8-rl: 迷宫求解 (Q-Learning)

该模块实现了一个基于 Q-Learning 的强化学习智能体，能够自动识别迷宫图像并规划从起点到终点的最佳路径。

### 1. 环境配置

请确保已安装 Python 3.8+。在终端中进入 `8-rl` 目录并安装依赖：

```bash
cd 8-rl
pip install -r requirements.txt
```

主要依赖库包括：`numpy`, `opencv-python`, `matplotlib`。

### 2. 运行方式

确保 `maze.jpg` 文件位于 `8-rl` 目录下，然后运行主程序：

```bash
python src/main.py
```

### 3. 程序说明

*   **图像处理**: 程序首先读取 `maze.jpg`，将其转换为网格环境。
*   **目标选择**:
    *   默认情况下，程序会自动寻找距离起点最远的有效点作为终点。
    *   您也可以选择手动输入终点坐标。
*   **训练过程**: 智能体会进行 10,000 Episode的训练。
*   **结果输出**:
    *   控制台输出训练进度、最佳路径长度。
    *   生成可视化结果图 `maze_solution.png`，展示识别出的迷宫、起点、终点及规划路径。

---

## 9-regression: 回归预测

基于 Gradient Boosting 构建了一个回归模型，用于对给定数据进行预测和评估。

### 1. 环境配置

在终端中进入 `9-regression` 目录并安装依赖：

```bash
cd 9-regression
pip install -r requirements.txt
```

### 2. 运行方式

确保数据文件 `回归预测.xlsx` 位于 `9-regression` 目录下，然后运行脚本：

```bash
python regression_model.py
```

### 3. 程序说明

*   **数据加载**: 程序读取 `回归预测.xlsx`。
    *   Sheet 1: 训练数据集。
    *   Sheet 2: 测试数据集。
    *   前 31 列作为特征（第 31 列为分类变量，其余为数值），第 32 列为目标值。
*   **模型管道**:
    *   预处理：数值特征标准化 ，分类特征独热编码。
    *   模型：梯度提升回归器 (GradientBoostingRegressor)。
*   **超参数调优**: 使用网格搜索 (GridSearchCV) 寻找最佳参数（树数量、学习率、最大深度等）。
*   **评估指标**: 在测试集上计算并输出：
    *   平方相对误差均值 (Mean Squared Relative Error)
    *   平方相对误差方差 (Variance of Squared Relative Error)
    *   均方误差 (MSE)
