"""Visualize training progress from tensorboard logs."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize training logs")

    parser.add_argument("--log-dir", type=str, default="logs", help="Path to tensorboard logs")
    parser.add_argument("--output", type=str, default="training_plot.png", help="Output image path")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")

    return parser.parse_args()


def load_tensorboard_logs(log_dir: str):
    """Load data from tensorboard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()

    data = {}
    for tag in tags["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {"steps": [e.step for e in events], "values": [e.value for e in events]}

    return data


def smooth(data: list, window: int) -> list:
    """Apply moving average smoothing."""
    if window <= 1:
        return data

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))

    return smoothed


def plot_training_logs(data: dict, output_path: str, smooth_window: int = 10):
    """Plot training metrics."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Progress", fontsize=16)

    # Plot 1: Episode Rewards
    if "episode/reward" in data:
        ax = axes[0, 0]
        steps = data["episode/reward"]["steps"]
        values = data["episode/reward"]["values"]

        ax.plot(steps, values, alpha=0.3, color="blue", label="Raw")
        ax.plot(steps, smooth(values, smooth_window), color="blue", linewidth=2, label=f"Smoothed (window={smooth_window})")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Mean Reward (100 episodes)
    if "episode/mean_reward_100" in data:
        ax = axes[0, 1]
        steps = data["episode/mean_reward_100"]["steps"]
        values = data["episode/mean_reward_100"]["values"]

        ax.plot(steps, values, color="green", linewidth=2)
        ax.axhline(y=200, color="red", linestyle="--", label="Success threshold (200)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean Reward (100 episodes)")
        ax.set_title("Mean Reward (Rolling Average)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Training Losses
    ax = axes[1, 0]
    has_loss_data = False

    if "train/policy_loss" in data:
        steps = data["train/policy_loss"]["steps"]
        values = smooth(data["train/policy_loss"]["values"], smooth_window)
        ax.plot(steps, values, label="Policy Loss", linewidth=2)
        has_loss_data = True

    if "train/value_loss" in data:
        steps = data["train/value_loss"]["steps"]
        values = smooth(data["train/value_loss"]["values"], smooth_window)
        ax.plot(steps, values, label="Value Loss", linewidth=2)
        has_loss_data = True

    if has_loss_data:
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Evaluation Metrics
    if "eval/mean_reward" in data:
        ax = axes[1, 1]
        steps = data["eval/mean_reward"]["steps"]
        values = data["eval/mean_reward"]["values"]
        stds = data.get("eval/std_reward", {}).get("values", [0] * len(values))

        ax.plot(steps, values, marker="o", color="purple", linewidth=2, label="Mean Reward")
        ax.fill_between(
            steps,
            np.array(values) - np.array(stds),
            np.array(values) + np.array(stds),
            alpha=0.3,
            color="purple",
        )
        ax.axhline(y=200, color="red", linestyle="--", label="Success threshold (200)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Evaluation Mean Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.close()


def main():
    """Main function."""
    args = parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' does not exist.")
        return

    print(f"Loading logs from {args.log_dir}...")
    data = load_tensorboard_logs(args.log_dir)

    if not data:
        print("No data found in tensorboard logs.")
        return

    print(f"Found {len(data)} scalar tags:")
    for tag in data.keys():
        print(f"  - {tag}: {len(data[tag]['steps'])} data points")

    print(f"\nGenerating plot...")
    plot_training_logs(data, args.output, args.smooth)


if __name__ == "__main__":
    main()
