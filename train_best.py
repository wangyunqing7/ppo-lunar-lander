"""Optimized training script for best performance on LunarLander-v2."""

import os
import argparse
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo_lunar_lander.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized training for LunarLander-v2")

    parser.add_argument("--env-id", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--c1", type=float, default=0.5)  # Lower value loss coefficient
    parser.add_argument("--c2", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed."""
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
    """Get device."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def evaluate_agent(agent: PPOAgent, env: gym.Env, num_episodes: int = 20) -> dict:
    """Evaluate agent."""
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
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

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

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    episode_rewards = []
    recent_rewards = []

    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_length = 0

    best_mean_reward = -float('inf')

    pbar = tqdm(total=args.total_timesteps, desc="Training")

    while global_step < args.total_timesteps:
        for _ in range(args.num_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.add(state, action, log_prob, reward, value, done or truncated)

            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done or truncated:
                episode_rewards.append(episode_reward)
                recent_rewards.append(episode_reward)

                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                if len(episode_rewards) % 10 == 0:
                    writer.add_scalar("episode/reward", episode_reward, global_step)
                    writer.add_scalar("episode/length", episode_length, global_step)
                    if len(recent_rewards) > 0:
                        writer.add_scalar(
                            "episode/mean_reward_100", np.mean(recent_rewards), global_step
                        )

                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

            pbar.update(1)
            if len(recent_rewards) > 0:
                pbar.set_postfix({
                    "reward": f"{np.mean(recent_rewards):.1f}",
                    "best": f"{best_mean_reward:.1f}",
                    "episodes": len(episode_rewards),
                })

            if global_step >= args.total_timesteps:
                break

        last_value = 0.0 if (done or truncated) else agent.policy.critic(
            torch.FloatTensor(state).unsqueeze(0).to(device)
        ).item()
        agent.buffer.compute_advantages_and_returns(last_value)

        metrics = agent.update(num_epochs=args.num_epochs, batch_size=args.batch_size)

        writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("train/entropy", metrics["entropy"], global_step)

        if global_step % args.eval_freq == 0 or global_step >= args.total_timesteps:
            eval_metrics = evaluate_agent(agent, eval_env, args.eval_episodes)
            current_mean = eval_metrics['mean_reward']

            if current_mean > best_mean_reward:
                best_mean_reward = current_mean
                save_path = os.path.join(args.save_dir, "ppo_best.pt")
                agent.save(save_path)
                print(f"\n✓ New best model! Mean reward: {current_mean:.2f}")
                print(f"  Saved to: {save_path}")

            print(f"\nEvaluation at step {global_step}:")
            print(f"  Mean Reward: {current_mean:.2f} +/- {eval_metrics['std_reward']:.2f}")
            print(f"  Best: {best_mean_reward:.2f}")
            print(f"  Min/Max: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")

            writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], global_step)
            writer.add_scalar("eval/std_reward", eval_metrics["std_reward"], global_step)
            writer.add_scalar("eval/best_reward", best_mean_reward, global_step)

        if global_step % args.save_freq == 0 or global_step >= args.total_timesteps:
            save_path = os.path.join(args.save_dir, f"ppo_{global_step}.pt")
            agent.save(save_path)

    pbar.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    eval_metrics = evaluate_agent(agent, eval_env, args.eval_episodes)
    print(f"Final Evaluation:")
    print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
    print(f"  Best Reward Achieved: {best_mean_reward:.2f}")
    print(f"  Success Rate (>200): {np.mean([r > 200 for r in episode_rewards[-100:]]) * 100:.1f}%")

    final_save_path = os.path.join(args.save_dir, "ppo_final.pt")
    agent.save(final_save_path)
    print(f"\nFinal model saved to: {final_save_path}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'ppo_best.pt')}")

    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    import torch  # Import at runtime
    args = parse_args()
    train(args)
