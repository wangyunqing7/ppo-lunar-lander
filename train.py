"""Training script for PPO on LunarLander-v2."""

import os
import argparse
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo_lunar_lander.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v2")

    # Environment settings
    parser.add_argument(
        "--env-id", type=str, default="LunarLander-v2", help="Gymnasium environment ID"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Optimization epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")

    # PPO settings
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--c1", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--c2", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Network settings
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num-hidden", type=int, default=2, help="Number of hidden layers")

    # Saving and logging
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for tensorboard logs")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Save frequency (timesteps)")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(device: str) -> str:
    """Get the device to use for training."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def evaluate_agent(agent: PPOAgent, env: gym.Env, num_episodes: int = 10) -> dict:
    """
    Evaluate the agent.

    Args:
        agent: PPO agent
        env: Environment
        num_episodes: Number of episodes to run

    Returns:
        Dictionary with evaluation metrics
    """
    agent.set_eval_mode()

    episode_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
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

    agent.set_train_mode()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
    }


def train(args):
    """Main training loop."""
    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create environment
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        c1=args.c1,
        c2=args.c2,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    recent_rewards = []

    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_length = 0

    pbar = tqdm(total=args.total_timesteps, desc="Training")

    while global_step < args.total_timesteps:
        # Collect rollouts
        for _ in range(args.num_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store transition
            agent.buffer.add(state, action, log_prob, reward, value, done or truncated)

            # Update tracking
            episode_reward += reward
            episode_length += 1
            global_step += 1

            # Check if episode ended
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                recent_rewards.append(episode_reward)

                # Keep only last 100 episodes for rolling average
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                # Log episode metrics
                if len(episode_rewards) % 10 == 0:
                    writer.add_scalar("episode/reward", episode_reward, global_step)
                    writer.add_scalar("episode/length", episode_length, global_step)
                    writer.add_scalar(
                        "episode/mean_reward_100", np.mean(recent_rewards), global_step
                    )

                # Reset episode
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

            # Update progress bar
            pbar.update(1)
            if len(recent_rewards) > 0:
                pbar.set_postfix(
                    {
                        "reward": f"{np.mean(recent_rewards):.2f}",
                        "episodes": len(episode_rewards),
                    }
                )

            # Stop if we've reached total timesteps
            if global_step >= args.total_timesteps:
                break

        # Compute advantages and returns
        last_value = 0.0 if (done or truncated) else agent.policy.critic(
            torch.FloatTensor(state).unsqueeze(0).to(device)
        ).item()
        agent.buffer.compute_advantages_and_returns(last_value)

        # Update policy
        metrics = agent.update(num_epochs=args.num_epochs, batch_size=args.batch_size)

        # Log training metrics
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("train/entropy", metrics["entropy"], global_step)

        # Evaluate agent
        if global_step % args.eval_freq == 0 or global_step >= args.total_timesteps:
            eval_metrics = evaluate_agent(agent, eval_env, args.eval_episodes)
            print(
                f"\nEvaluation at step {global_step}: "
                f"Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}"
            )

            writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], global_step)
            writer.add_scalar("eval/std_reward", eval_metrics["std_reward"], global_step)
            writer.add_scalar("eval/min_reward", eval_metrics["min_reward"], global_step)
            writer.add_scalar("eval/max_reward", eval_metrics["max_reward"], global_step)

        # Save model
        if global_step % args.save_freq == 0 or global_step >= args.total_timesteps:
            save_path = os.path.join(args.save_dir, f"ppo_{global_step}.pt")
            agent.save(save_path)
            print(f"\nModel saved to {save_path}")

    pbar.close()

    # Final evaluation and save
    print("\nTraining completed!")
    eval_metrics = evaluate_agent(agent, eval_env, args.eval_episodes)
    print(f"Final Evaluation: Mean Reward: {eval_metrics['mean_reward']:.2f}")

    # Save final model
    final_save_path = os.path.join(args.save_dir, "ppo_final.pt")
    agent.save(final_save_path)
    print(f"Final model saved to {final_save_path}")

    writer.close()

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
