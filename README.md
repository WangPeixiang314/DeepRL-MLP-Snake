# DeepRL-MLP-Snake 🐍

*English | [中文](#中文说明)*

A high-performance Deep Reinforcement Learning Snake game using Deep Q-Networks (DQN) with advanced optimization techniques.

## 🚀 Features

- **Deep Q-Learning**: Implementation of DQN with experience replay and target networks
- **Priority Experience Replay**: Uses SumTree for efficient prioritized sampling
- **Numba Acceleration**: JIT-compiled critical functions for 10x performance boost
- **Advanced State Representation**: 11-dimensional state space including danger detection, food direction, and local grid view
- **Anti-suicide Mechanism**: Prevents the snake from making obviously dangerous moves during training
- **Real-time Visualization**: Live training progress with matplotlib plots
- **Model Checkpointing**: Automatic saving of best models and periodic backups

## 📊 Performance Metrics

- **Training Speed**: ~1000 episodes/hour on modern GPU
- **State Space**: Optimized feature vector (removed full grid)
- **Action Space**: 3 discrete actions (straight, turn right, turn left)
- **Memory Capacity**: 200K transitions with prioritized replay
- **Network Architecture**: Efficient MLP with [128, 64, 32, 16, 8] hidden layers

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (optional but recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/DeepRL-MLP-Snake.git
cd DeepRL-MLP-Snake

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Training
```bash
# Start training with visualization
python train.py

# Train without visualization (faster)
python train.py --no-visual

# Train for specific episodes
python train.py --episodes 5000
```

### Advanced Training Options
```python
from train import train

# Custom training
train(
    num_episodes=10000,
    visualize=True,
    verbose=True
)
```

## 📁 Project Structure

```
DeepRL-MLP-Snake/
├── agent.py              # DQN Agent implementation
├── config.py             # Configuration parameters
├── game.py               # Snake game environment
├── model.py              # Neural network architecture
├── memory.py             # Prioritized experience replay
├── train.py              # Training script
├── training.py           # Visualization and stats
├── help.py               # Numba-optimized helper functions
├── direction.py          # Direction enum
├── models/               # Saved model checkpoints
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

Key parameters in `config.py`:

```python
# Network Architecture (Optimized)
HIDDEN_LAYERS = [128, 64, 32, 16, 8]  # More efficient design

# Training Parameters
BATCH_SIZE = 128                      # Reduced from 512
LEARNING_RATE = 3.48e-5
MEMORY_CAPACITY = 200_000            # Reduced from 2M
GAMMA = 0.99

# Exploration
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 7000

# Grid Configuration
GRID_WIDTH = 12                       # Increased from 11
GRID_HEIGHT = 12                      # Increased from 11
```

## 🎮 Game Mechanics

### State Representation (Compact Design)
1. **Danger Detection**: 3 binary flags (front, right, left)
2. **Head Position**: Normalized coordinates
3. **Food Position**: Normalized coordinates
4. **Relative Distance**: Vector to food
5. **Manhattan Distance**: Steps to food
6. **Direction**: One-hot encoded current direction
7. **8-Direction Dangers**: Collision detection in 8 directions
8. **Boundary Distances**: Distance to walls
9. **Snake Length**: Normalized length
10. **Free Space Ratio**: Available space percentage
11. **Local Grid View**: 6x6 grid around head
12. **Action History**: Last 5 actions

*Note: Full grid representation has been removed for efficiency*

### Reward Structure
- **Food**: +16.0
- **Collision**: -10.0
- **Progress**: +0.1 (moving closer to food)
- **Step**: -0.01 (time penalty)

## 📈 Training Visualization

The training process displays:
- Real-time score progression
- Moving average scores (500-episode window)
- Total rewards per episode
- Training steps per episode
- Loss curves and exploration rate

## 🔬 Advanced Features

### Numba Acceleration
Critical game logic functions are JIT-compiled using Numba for maximum performance:
- Collision detection
- Distance calculations
- Experience replay sampling
- Game step logic

### Priority Experience Replay
- Uses SumTree data structure for O(log n) sampling
- Prioritized by TD-error magnitude
- Importance sampling weights for bias correction

### Anti-suicide Mechanism
During training, the agent avoids obviously dangerous moves even during exploration, significantly improving sample efficiency.

## 📊 Results

Typical training results after 1000 episodes:
- **Average Score**: 15-25
- **Peak Score**: 40-50
- **Training Time**: ~1 hour on RTX 3060

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by DeepMind's DQN paper
- Built with PyTorch and Numba
- Game framework based on Pygame

---

## 中文说明 🇨🇳

## 🚀 功能特性

- **深度Q学习**: 实现带有经验回放和目标网络的DQN算法
- **优先级经验回放**: 使用SumTree进行高效优先级采样
- **Numba加速**: JIT编译关键函数，性能提升10倍
- **高级状态表示**: 11维状态空间，包括危险检测、食物方向和局部网格视图
- **防自杀机制**: 训练期间防止蛇做出明显危险的动作
- **实时可视化**: 使用matplotlib实时显示训练进度
- **模型检查点**: 自动保存最佳模型和定期备份

## 📊 性能指标

- **训练速度**: 现代GPU上约1000局/小时
- **状态空间**: 优化后的特征向量（移除完整网格）
- **动作空间**: 3个离散动作（直行、右转、左转）
- **记忆容量**: 20万条转换记录，带优先级回放
- **网络架构**: 高效MLP，隐藏层[128, 64, 32, 16, 8]

## 🛠️ 安装指南

### 系统要求
- Python 3.11或更高版本
- CUDA兼容GPU（可选但推荐）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-username/DeepRL-MLP-Snake.git
cd DeepRL-MLP-Snake

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 🎯 快速开始

### 基础训练
```bash
# 开始带可视化的训练
python train.py

# 无可视化训练（更快）
python train.py --no-visual

# 训练指定局数
python train.py --episodes 5000
```

### 高级训练选项
```python
from train import train

# 自定义训练
train(
    num_episodes=10000,
    visualize=True,
    verbose=True
)
```

## 📁 项目结构

```
DeepRL-MLP-Snake/
├── agent.py              # DQN智能体实现
├── config.py             # 配置参数
├── game.py               # 贪吃蛇游戏环境
├── model.py              # 神经网络架构
├── memory.py             # 优先级经验回放
├── train.py              # 训练脚本
├── training.py           # 可视化和统计
├── help.py               # Numba优化的辅助函数
├── direction.py          # 方向枚举
├── models/               # 保存的模型检查点
├── requirements.txt      # Python依赖
└── README.md            # 本文件
```

## 🔧 配置参数

`config.py`中的关键参数：

```python
# 网络架构
HIDDEN_LAYERS = [1024, 512, 256, 128, 64, 32, 16, 8]

# 训练参数
BATCH_SIZE = 512
LEARNING_RATE = 3.48e-5
MEMORY_CAPACITY = 2_000_000
GAMMA = 0.99

# 探索参数
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 7000
```

## 🎮 游戏机制

### 状态表示（紧凑设计）
1. **危险检测**: 3个二进制标志（前方、右方、左方）
2. **头部位置**: 归一化坐标
3. **食物位置**: 归一化坐标
4. **相对距离**: 到食物的向量
5. **曼哈顿距离**: 到食物的步数
6. **方向**: 当前方向的一热编码
7. **8方向危险**: 8个方向的碰撞检测
8. **边界距离**: 到墙壁的距离
9. **蛇长度**: 归一化长度
10. **空闲空间比例**: 可用空间百分比
11. **局部网格视图**: 头部周围6x6网格
12. **动作历史**: 最近5个动作

*注意：完整网格表示已移除以提高效率*

### 奖励结构
- **食物**: +16.0
- **碰撞**: -10.0
- **进度**: +0.1（向食物移动）
- **步数**: -0.01（时间惩罚）

## 📈 训练可视化

训练过程显示：
- 实时分数进展
- 移动平均分数（500局窗口）
- 每局总奖励
- 每局训练步数
- 损失曲线和探索率

## 🔬 高级特性

### Numba加速
使用Numba JIT编译关键游戏逻辑函数，获得最大性能：
- 碰撞检测
- 距离计算
- 经验回放采样
- 游戏步骤逻辑

### 优先级经验回放
- 使用SumTree数据结构，O(log n)采样
- 按TD误差幅度优先级排序
- 重要性采样权重用于偏差校正

### 防自杀机制
训练期间，智能体即使在探索时也避免明显危险的动作，显著提高样本效率。

## 📊 训练结果

1000局后的典型训练结果：
- **平均分数**: 15-25
- **最高分数**: 40-50
- **训练时间**: RTX 3060上约1小时

## 🤝 贡献指南

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- 灵感来自DeepMind的DQN论文
- 使用PyTorch和Numba构建
- 游戏框架基于Pygame