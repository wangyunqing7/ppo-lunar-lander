"""Record trained agent performance as GIF or MP4 video."""

import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import os

from ppo_lunar_lander.agents import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record agent performance as video")

    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Gymnasium environment ID")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--output", type=str, default="agent_demo.gif", help="Output file path (use .gif or .mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--resize", type=int, default=512, help="Resize frames to this size")

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


def record_as_gif(frames, output_path, fps=30, resize=512):
    """Save frames as GIF using Pillow."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Install with: pip install pillow")
        return False

    print(f"Creating GIF with {len(frames)} frames...")

    # Resize frames if needed
    if resize and frames[0].shape[0] != resize:
        print(f"Resizing frames to {resize}x{resize}...")
        resized_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = img.resize((resize, resize), Image.Resampling.LANCZOS)
            resized_frames.append(np.array(img))
        frames = resized_frames

    # Save as GIF
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
        optimize=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ GIF saved to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.1f} seconds")
    return True


def record_as_mp4(frames, output_path, fps=30, resize=512):
    """Save frames as MP4 using OpenCV."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return False

    print(f"Creating MP4 with {len(frames)} frames...")

    # Resize frames if needed
    if resize and frames[0].shape[0] != resize:
        print(f"Resizing frames to {resize}x{resize}...")
        frames = [cv2.resize(frame, (resize, resize)) for frame in frames]

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ MP4 saved to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.1f} seconds")
    return True


def record_agent(args):
    """Record agent performance."""
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create environment with rgb_array rendering
    env = gym.make(args.env_id, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent and load model
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    print(f"Loading model from {args.model_path}")
    agent.load(args.model_path)
    agent.set_eval_mode()

    # Record episodes
    all_frames = []
    episode_rewards = []

    for episode in tqdm(range(args.num_episodes), desc="Recording episodes"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        frames = []

        while not (done or truncated):
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Select action
            action, _, _ = agent.select_action(state)

            # Take action
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        all_frames.extend(frames)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    # Save video
    print(f"\nTotal frames recorded: {len(all_frames)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")

    output_ext = os.path.splitext(args.output)[1].lower()

    if output_ext == '.gif':
        success = record_as_gif(all_frames, args.output, args.fps, args.resize)
    elif output_ext == '.mp4':
        success = record_as_mp4(all_frames, args.output, args.fps, args.resize)
    else:
        # Default to GIF
        print(f"Unknown extension '{output_ext}', defaulting to GIF")
        output_path = args.output if args.output.endswith('.gif') else args.output + '.gif'
        success = record_as_gif(all_frames, output_path, args.fps, args.resize)

    if success:
        print(f"\n✓ Recording completed successfully!")
        print(f"  Output: {args.output}")
    else:
        print(f"\n✗ Failed to create video")


if __name__ == "__main__":
    args = parse_args()
    record_agent(args)
