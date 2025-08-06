# Snake AI based on Deep Reinforcement Learning

This is a Snake AI project trained using Deep Q-Network (DQN) with prioritized experience replay. The AI can learn to play the Snake game through self-play and gradually improve its performance.

## Project Features

- Uses Deep Q-Network (DQN) algorithm
- Implements prioritized experience replay mechanism
- Employs Multi-Layer Perceptron (MLP) as neural network structure
- Supports hyperparameter optimization
- Real-time visualization of training process
- Anti-suicide mechanism to prevent obvious self-destructive actions

## Environment Dependencies

- Python 3.11+
- PyTorch 2.0+
- NumPy 1.24+
- Pygame 2.5+
- Numba 0.58+
- Matplotlib 3.7+

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepRL-MLP-Snake
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the following command to start training:
```bash
python train.py
```

Training parameters can be adjusted in the `config.py` file.

### Hyperparameter Optimization

The project supports hyperparameter optimization using Optuna:
```bash
python optimize_hyperparameters.py
```

### Viewing Training Results

Charts are displayed in real-time during training, showing the following metrics:
- Score (snake length)
- Total reward
- Steps
- Training loss

## Project Structure

```
DeepRL-MLP-Snake/
├── agent.py          # DQN agent implementation
├── config.py         # Configuration parameters
├── direction.py      # Direction enumeration definition
├── game.py           # Game logic implementation
├── help.py           # Helper functions (accelerated with numba)
├── memory.py         # Prioritized experience replay implementation
├── model.py          # Neural network model definition
├── train.py          # Main training program
├── training.py       # Training statistics and visualization
├── optimize_hyperparameters.py  # Hyperparameter optimization
├── requirements.txt  # Project dependencies
├── models/           # Trained model files
└── README_EN.md      # Project description file
```

## Technical Details

### State Representation

The AI's state representation includes the following features:
- Danger detection in front, right, and left directions
- Snake head position
- Food position
- Relative distance between snake head and food
- Manhattan distance
- One-hot encoding of current direction
- Danger detection in 8 directions
- Boundary distances
- Snake body length
- Free space ratio
- Local grid view
- Action history

### Reward Mechanism

- Eating food: +16 points
- Collision penalty: -10 points
- Moving toward food: +0.1 * distance difference
- Step penalty: -0.01 points per step

### Network Architecture

Default 5-layer MLP network:
- Input layer: Determined by state feature count
- Hidden layers: 128 -> 64 -> 32 -> 16 -> 8
- Output layer: 3 (go straight, turn right, turn left)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.