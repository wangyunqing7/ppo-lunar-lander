"""Quick test script to verify installation."""

import sys

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

try:
    import gymnasium as gym
    print(f"✓ Gymnasium {gym.__version__}")

    # Test environment creation
    env = gym.make("LunarLander-v2")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test step
    state, info = env.reset()
    print(f"  Initial state shape: {state.shape}")

    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"  Next state shape: {next_state.shape}")
    print(f"  Reward: {reward}")

    env.close()
    print("\n✓ Environment test passed!")

except Exception as e:
    print(f"✗ Gymnasium error: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    print("✓ TensorBoard available")
except ImportError:
    print("✗ TensorBoard not available")
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
