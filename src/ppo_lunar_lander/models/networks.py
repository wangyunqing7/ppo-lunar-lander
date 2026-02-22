"""Actor-Critic networks for PPO algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """Actor network that outputs action probabilities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ):
        """
        Initialize Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of discrete actions)
            hidden_dim: Number of neurons in hidden layers
            num_hidden: Number of hidden layers
        """
        super().__init__()

        layers = []
        input_dim = state_dim

        # Build hidden layers
        for _ in range(num_hidden):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            state: Current state tensor

        Returns:
            Action logits (before softmax)
        """
        return self.network(state)

    def get_action_dist(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get action distribution for given state.

        Args:
            state: Current state tensor

        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """Critic network that estimates state value."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ):
        """
        Initialize Critic network.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layers
            num_hidden: Number of hidden layers
        """
        super().__init__()

        layers = []
        input_dim = state_dim

        # Build hidden layers
        for _ in range(num_hidden):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state: Current state tensor

        Returns:
            State value estimate
        """
        return self.network(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of neurons in hidden layers
            num_hidden: Number of hidden layers
        """
        super().__init__()

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, num_hidden)
        self.critic = CriticNetwork(state_dim, hidden_dim, num_hidden)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Forward pass through both actor and critic.

        Args:
            state: Current state tensor

        Returns:
            Tuple of (action distribution, state value)
        """
        action_dist = self.actor.get_action_dist(state)
        value = self.critic(state)
        return action_dist, value

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Current state tensor

        Returns:
            Tuple of (action, log_prob, state_value)
        """
        with torch.no_grad():
            action_dist, value = self.forward(state)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Categorical]:
        """
        Evaluate actions for PPO update.

        Args:
            state: Batch of states
            action: Batch of actions taken

        Returns:
            Tuple of (log_probs, state_values, action_distribution)
        """
        action_dist, value = self.forward(state)
        log_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return log_probs, value, entropy
