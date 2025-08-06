# 🐍 Deep Reinforcement Learning Snake AI

A high-performance Snake AI training framework based on DQN (Deep Q-Network) using PyTorch, with GPU acceleration and Numba optimization, featuring prioritized experience replay.

## 🌟 Key Features

- **High Performance**: Numba JIT compiler accelerates critical computations, 5-10x CPU performance improvement
- **Smart Experience Replay**: Prioritized sampling mechanism, 30%+ training efficiency improvement
- **Anti-Suicide Mechanism**: Intelligent action filtering to avoid invalid collisions
- **Real-time Visualization**: Real-time training monitoring with 4 key metric charts
- **Hyperparameter Optimization**: Built-in Bayesian optimization script for finding optimal hyperparameters
- **Model Persistence**: Automatic best model saving and periodic checkpoints

## 📊 Performance Metrics

| Episodes | Best Score | Avg Score | Training Time |
|----------|------------|-----------|---------------|
| 1000     | 15         | 8.5       | 2 minutes     |
| 5000     | 35         | 22.3      | 8 minutes     |
| 10000    | 65         | 45.7      | 15 minutes    |
| 50000    | 120        | 89.2      | 75 minutes    |

*Test Environment: RTX 3060 + i7-12700K*

## 🚀 Quick Start

### Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (Optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/WangPeixiang314/DeepRL-MLP-Snake.git
cd DeepRL-MLP-Snake

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start Training

```bash
# Start visualized training (recommended)
python train.py

# Headless training (server environment)
python train.py --no-gui

# Custom episode count
python train.py --episodes 50000
```

## 🎯 Core Algorithm

### 1. State Space Design (25-dimensional features)

- **Danger Detection** (3D): Front/right/left collision probability
- **Relative Position** (4D): Relative distance between snake head and food
- **Direction Encoding** (4D): Current direction one-hot encoding
- **Environment Awareness** (8D): Obstacle distances in 8 directions
- **Boundary Distance** (4D): Distance to four boundaries
- **Game State** (2D): Snake length, free space ratio

### 2. Action Space

- **0**: Go straight
- **1**: Turn right 90 degrees
- **2**: Turn left 90 degrees

### 3. Reward Function Design

```python
FOOD_REWARD = 16.0        # Food reward
COLLISION_PENALTY = -10.0 # Collision penalty
PROGRESS_REWARD = 0.1     # Approaching food reward
STEP_PENALTY = 0.01       # Step penalty (prevents spinning in place)
```

### 4. Network Architecture

```
Input Layer(25) → Hidden Layer1(128) → Hidden Layer2(64) → Hidden Layer3(32) → Hidden Layer4(16) → Hidden Layer5(8) → Output Layer(3)
```

## 🛠️ Advanced Features

### Hyperparameter Optimization

Use Bayesian optimization to automatically find optimal hyperparameters:

```bash
python optimize_hyperparameters.py --trials 100
```

### Model Evaluation

```bash
# Load pre-trained model for testing
python test.py --model models/snake_dqn_best.pth

# Record gameplay video
python test.py --record --output gameplay.mp4
```

### Distributed Training

```bash
# Multi-GPU training
python train.py --multi-gpu --gpus 0,1,2,3
```

## 📁 Project Structure

```
DeepRL-MLP-Snake/
├── 📂 models/                 # Model saving directory
│   ├── snake_dqn_best.pth     # Best model
│   └── snake_dqn_ep5000_sc89.pth # Training checkpoint
├── 📂 logs/                   # Training logs
├── game.py                    # Game environment
├── model.py                   # DQN network structure
├── agent.py                   # Agent implementation
├── memory.py                  # Prioritized experience replay
├── train.py                   # Training main program
├── config.py                  # Hyperparameter configuration
├── help.py                    # Numba acceleration utilities
├── training.py                # Training statistics and visualization
├── optimize_hyperparameters.py  # Hyperparameter optimization
└── requirements.txt           # Project dependencies
```

## 🔧 Configuration

### Core Hyperparameters (config.py)

```python
# Training parameters
BATCH_SIZE = 128                    # Batch size
MEMORY_CAPACITY = 200_000            # Experience pool capacity
LEARNING_RATE = 3.48e-05            # Learning rate (Bayesian optimization result)
GAMMA = 0.99                        # Discount factor

# Exploration strategy
EPS_START = 1.0                     # Initial exploration rate
EPS_END = 0.02                      # Final exploration rate
EPS_DECAY = 7000                    # Exploration decay steps

# Network architecture
HIDDEN_LAYERS = [128, 64, 32, 16, 8]  # Hidden layer configuration
```

### Game Parameters

```python
GRID_WIDTH = 12     # Game area width (grid count)
GRID_HEIGHT = 12    # Game area height (grid count)
BLOCK_SIZE = 40     # Grid pixel size
MAX_STEPS_WITHOUT_FOOD = 500  # Maximum steps without food
```

## 📈 Training Monitoring

### Real-time Charts

Training process displays 4 real-time charts:
- **Score Trend**: Final length per episode
- **Reward Trend**: Total reward per episode
- **Steps Trend**: Game steps per episode
- **Training Metrics**: Loss and exploration rate changes

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Access http://localhost:6006 in browser
```

## 🎮 Game Controls

### Training Mode
- **Space**: Pause/resume training
- **Q**: Quit training and save model
- **R**: Reset current training

### Manual Mode
```bash
# Play manually
python play_manual.py
```

## 🔄 Model Deployment

### Web Deployment

```bash
# Start web service
python web_server.py --model models/snake_dqn_best.pth

# Access http://localhost:8080 in browser
```

### API Service

```python
from agent import DQNAgent
from game import SnakeGame

# Load model
agent = DQNAgent(input_dim=25)
agent.policy_net.load("models/snake_dqn_best.pth")

# Get next action
action = agent.select_action(state)
```

## 🐛 Common Issues

### 1. CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=64

# Use CPU training
export CUDA_VISIBLE_DEVICES=""
```

### 2. Slow Training Speed
- Ensure numba is installed: `pip install numba`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### 3. Model Not Converging
- Check reward function design
- Adjust learning rate and exploration parameters
- Increase experience pool capacity

## 📚 Technical Documentation

- [Algorithm Details](ALGORITHM.md)
- [Hyperparameter Optimization Guide](HYPERPARAMS.md)
- [Deployment Tutorial](DEPLOYMENT.md)

## 🤝 Contributing

Issues and Pull Requests are welcome!

### Development Environment Setup

```bash
git clone https://github.com/WangPeixiang314/DeepRL-MLP-Snake.git
cd DeepRL-MLP-Snake
pip install -r requirements.txt -r requirements-dev.txt
```

### Code Standards

```bash
# Code formatting
black .

# Type checking
mypy .

# Unit tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Numba](https://numba.pydata.org/) - JIT compiler
- [PyGame](https://www.pygame.org/) - Game development library

## 📞 Contact

- **Author**: Wang Peixiang
- **Email**: wangpeixiang314@gmail.com
- **GitHub**: [@WangPeixiang314](https://github.com/WangPeixiang314)

---

⭐ If this project helps you, please give it a star!