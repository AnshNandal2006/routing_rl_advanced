"""Neural architectures for Deep Q-Learning (DQN / Double / Dueling)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DuelingDQN(nn.Module):
    """
    Dueling architecture:
      V(s)  +  (A(s,a) - mean_a A(s,a))
    Improves learning of state values independent of actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: List[int] = None,
        dueling: bool = True,
    ):
        super().__init__()
        hidden = hidden or [128, 128, 64]
        self.dueling = dueling
        self.action_dim = action_dim

        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.feature = nn.Sequential(*layers)

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(prev, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, action_dim),
            )
        else:
            self.head = nn.Linear(prev, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        if self.dueling:
            v = self.value_stream(feats)
            a = self.advantage_stream(feats)
            return v + (a - a.mean(dim=-1, keepdim=True))
        return self.head(feats)
