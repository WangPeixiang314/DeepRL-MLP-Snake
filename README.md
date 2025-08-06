# 基于深度强化学习的贪吃蛇AI

这是一个使用深度Q网络（DQN）和优先级经验回放训练的贪吃蛇AI项目。AI能够通过自我对弈学习如何玩贪吃蛇游戏，并逐步提高游戏表现。

## 项目特点

- 使用深度Q网络（DQN）算法
- 实现优先级经验回放机制
- 采用多层感知机（MLP）作为神经网络结构
- 支持超参数优化
- 实时可视化训练过程
- 防自杀机制，避免AI做出明显的自杀行为

## 环境依赖

- Python 3.11+
- PyTorch 2.0+
- NumPy 1.24+
- Pygame 2.5+
- Numba 0.58+
- Matplotlib 3.7+

## 安装步骤

1. 克隆项目代码：
   ```bash
   git clone <repository-url>
   cd DeepRL-MLP-Snake
   ```

2. 创建虚拟环境（推荐）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # 或
   .venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 训练模型

运行以下命令开始训练：
```bash
python train.py
```

训练参数可以在 `config.py` 文件中进行调整。

### 超参数优化

项目支持使用Optuna进行超参数优化：
```bash
python optimize_hyperparameters.py
```

### 查看训练结果

训练过程中会实时显示图表，展示以下指标：
- 分数（蛇的长度）
- 总奖励
- 步数
- 训练损失

## 项目结构

```
DeepRL-MLP-Snake/
├── agent.py          # DQN智能体实现
├── config.py         # 配置参数
├── direction.py      # 方向枚举定义
├── game.py           # 游戏逻辑实现
├── help.py           # 辅助函数（使用numba加速）
├── memory.py         # 优先级经验回放实现
├── model.py          # 神经网络模型定义
├── train.py          # 训练主程序
├── training.py       # 训练统计和可视化
├── optimize_hyperparameters.py  # 超参数优化
├── requirements.txt  # 项目依赖
├── models/           # 训练好的模型文件
└── README.md         # 项目说明文件
```

## 技术细节

### 状态表示

AI的状态表示包括以下特征：
- 前方、右方、左方的危险检测
- 蛇头位置
- 食物位置
- 蛇头与食物的相对距离
- 曼哈顿距离
- 当前方向的独热编码
- 8个方向的危险检测
- 边界距离
- 蛇身长度
- 空闲空间比例
- 局部网格视图
- 动作历史

### 奖励机制

- 吃到食物：+16分
- 碰撞惩罚：-10分
- 向食物靠近：+0.1 * 距离差
- 每步惩罚：-0.01分

### 网络结构

默认使用5层MLP网络：
- 输入层：根据状态特征数量确定
- 隐藏层：128 -> 64 -> 32 -> 16 -> 8
- 输出层：3（直行、右转、左转）

## 许可证

本项目采用MIT许可证，详情请见 [LICENSE](LICENSE) 文件。