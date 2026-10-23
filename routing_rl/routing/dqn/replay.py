"""Experience replay buffer with optional prioritization (simplified)."""
from __future__ import annotations
import random
from collections import deque
from typing import Deque, Tuple, List, Any
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,          # valid action mask at next state
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
