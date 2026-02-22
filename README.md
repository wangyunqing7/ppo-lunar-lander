# PPO Lunar Lander

Implementation of Proximal Policy Optimization (PPO) for the LunarLander-v2 reinforcement learning environment.

## Overview

This project implements the PPO algorithm to train an agent to successfully land a lunar lander in the [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment from Gymnasium. The implementation includes:

- Actor-Critic neural networks with configurable architecture
- GAE (Generalized Advantage Estimation) for advantage computation
- TensorBoard logging for training visualization
- Model checkpointing and evaluation utilities
- Training and evaluation scripts

## Requirements

- Python 3.12+
- UV package manager (recommended) or pip

## Installation

### Using UV (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/wangyunqing7/ppo-lunar-lander.git
cd ppo-lunar-lander

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/wangyunqing7/ppo-lunar-lander.git
cd ppo-lunar-lander

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Project Structure

```
ppo-lunar-lander/
├── src/
│   └── ppo_lunar_lander/
│       ├── agents/
│       │   ├── __init__.py
│       │   └── ppo.py           # PPO agent implementation
│       ├── models/
│       │   ├── __init__.py
│       │   └── networks.py      # Actor-Critic networks
│       └── __init__.py
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── visualize.py                 # Visualization script
├── checkpoints/                 # Saved model checkpoints (created during training)
├── logs/                        # TensorBoard logs (created during training)
├── pyproject.toml              # Project configuration
└── README.md
```

## Usage

### Training

Train a PPO agent from scratch:

```bash
# Basic training with default hyperparameters
python train.py

# Training with custom hyperparameters
python train.py \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --num-steps 2048 \
    --batch-size 64 \
    --num-epochs 10 \
    --hidden-dim 256 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2

# Training on GPU (if available)
python train.py --device cuda
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--env-id` | LunarLander-v2 | Gymnasium environment ID |
| `--seed` | 42 | Random seed for reproducibility |
| `--total-timesteps` | 1,000,000 | Total training timesteps |
| `--learning-rate` | 3e-4 | Learning rate for optimizer |
| `--num-steps` | 2048 | Steps per policy update |
| `--batch-size` | 64 | Mini-batch size for optimization |
| `--num-epochs` | 10 | Optimization epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda parameter |
| `--clip-epsilon` | 0.2 | PPO clipping parameter |
| `--c1` | 1.0 | Value function loss coefficient |
| `--c2` | 0.01 | Entropy bonus coefficient |
| `--hidden-dim` | 256 | Hidden layer size |
| `--num-hidden` | 2 | Number of hidden layers |
| `--save-dir` | checkpoints | Directory to save models |
| `--log-dir` | logs | Directory for TensorBoard logs |
| `--save-freq` | 100,000 | Model save frequency (timesteps) |
| `--eval-freq` | 10,000 | Evaluation frequency |
| `--device` | auto | Device (cpu/cuda/auto) |

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006`

### Evaluation

Evaluate a trained model:

```bash
# Evaluate without rendering
python evaluate.py --model-path checkpoints/ppo_final.pt --num-episodes 100

# Evaluate with rendering (watch the agent)
python evaluate.py --model-path checkpoints/ppo_final.pt --num-episodes 10 --render
```

### Visualization

Generate training plots from TensorBoard logs:

```bash
python visualize.py --log-dir logs --output training_plot.png --smooth 10
```

## Algorithm Details

### PPO (Proximal Policy Optimization)

PPO is a policy gradient method that uses a clipped surrogate objective to prevent large policy updates. The key components:

1. **Actor-Critic Architecture**:
   - Actor: Outputs a probability distribution over actions
   - Critic: Estimates the state value function V(s)

2. **GAE (Generalized Advantage Estimation)**:
   - Computes advantages with bias-variance trade-off
   - λ parameter controls the trade-off (0.95 default)

3. **Clipped Surrogate Objective**:
   - Prevents destructive large policy updates
   - ε parameter controls the clipping range (0.2 default)

4. **Entropy Bonus**:
   - Encourages exploration by penalizing low-entropy policies

### Network Architecture

- Input: State observation (8 dimensions for LunarLander-v2)
- Hidden layers: Fully connected with Tanh activation
- Output:
  - Actor: Action logits (4 discrete actions)
  - Critic: State value (scalar)

### Hyperparameters

The default hyperparameters are tuned for LunarLander-v2:

- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO clip (ε): 0.2
- Value loss coefficient: 1.0
- Entropy coefficient: 0.01
- Max gradient norm: 0.5
- Batch size: 64
- Optimization epochs: 10

## Expected Results

With the default hyperparameters and 1M timesteps of training:

- **Average reward**: Should reach 200+ (success threshold)
- **Success rate**: >80% of episodes end with reward >200
- **Training time**: ~30-60 minutes on CPU, ~5-10 minutes on GPU

The environment considers a landing successful if:
- The lander touches the ground between the two flags
- The lander is not moving
- The lander is upright (angle near 0)
- Reward > 200

## Troubleshooting

**Training is unstable**:
- Reduce learning rate (try 1e-4)
- Increase number of optimization epochs (try 20)
- Adjust entropy coefficient (try 0.005 or 0.02)

**Agent not learning**:
- Check TensorBoard logs for policy collapse (entropy → 0)
- Increase entropy coefficient for more exploration
- Verify environment is rendering correctly with `--render`

**Out of memory errors**:
- Reduce batch size
- Reduce number of steps per update
- Use CPU instead of GPU

## License

This project is open source and available under the MIT License.

## References

- [Schulman et al., "Proximal Policy Optimization Algorithms" (2017)](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- OpenAI for the PPO algorithm
- Gymnasium team for the LunarLander environment
- PyTorch team for the deep learning framework
