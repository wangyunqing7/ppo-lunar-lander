"""Evaluate a trained PPO agent on LunarLander-v2."""

import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from ppo_lunar_lander.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")

    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--env-id", type=str, default="LunarLander-v2", help="Gymnasium environment ID")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)")

    return parser.parse_args()


def get_device(device: str) -> str:
    """Get the device to use."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def evaluate(args):
    """Evaluate the agent."""
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create environment
    env = gym.make(args.env_id, render_mode="human" if args.render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    agent.load(args.model_path)
    agent.set_eval_mode()

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in tqdm(range(args.num_episodes), desc="Evaluating"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Count successful landings (reward > 200)
        if episode_reward > 200:
            success_count += 1

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"Success rate (>200 reward): {success_count / args.num_episodes * 100:.1f}%")
    print("=" * 60)

    # Close environment
    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
