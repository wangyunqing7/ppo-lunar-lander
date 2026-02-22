"""PPO Agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque

from ..models.networks import ActorCritic


class RolloutBuffer:
    """Buffer to store rollout data for PPO."""

    def __init__(self, buffer_size: int = 2048, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages_and_returns(self, last_value: float = 0.0):
        """
        Compute advantages and returns using GAE (Generalized Advantage Estimation).

        Args:
            last_value: Value estimate of the next state (for bootstrapping)
        """
        advantages = []
        returns = []
        gae = 0.0

        # Process in reverse order
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Get randomized batches from the buffer.

        Args:
            batch_size: Size of each batch

        Returns:
            Tuple of batched tensors (states, actions, log_probs, advantages, returns)
        """
        indices = np.random.permutation(len(self.states))

        for start in range(0, len(self.states), batch_size):
            end = min(start + batch_size, len(self.states))
            batch_indices = indices[start:end]

            yield (
                torch.FloatTensor(np.array([self.states[i] for i in batch_indices])),
                torch.LongTensor(np.array([self.actions[i] for i in batch_indices])),
                torch.FloatTensor(np.array([self.log_probs[i] for i in batch_indices])),
                torch.FloatTensor(np.array([self.advantages[i] for i in batch_indices])),
                torch.FloatTensor(np.array([self.returns[i] for i in batch_indices])),
            )

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.states)


class PPOAgent:
    """PPO Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,  # Value function coefficient
        c2: float = 0.01,  # Entropy coefficient
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of neurons in hidden layers
            num_hidden: Number of hidden layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            c1: Value function loss coefficient
            c2: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, num_hidden).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action using the current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.policy.get_action(state_tensor)
        return action, log_prob, value

    def update(self, num_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Args:
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size for optimization

        Returns:
            Dictionary of training metrics
        """
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(self.buffer.advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(self.buffer.returns)).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Optimize policy for multiple epochs
        for _ in range(num_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)

                # Compute policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Compute total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy.mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Clear buffer
        self.buffer.clear()

        # Return average metrics
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def set_eval_mode(self):
        """Set policy to evaluation mode."""
        self.policy.eval()

    def set_train_mode(self):
        """Set policy to training mode."""
        self.policy.train()
