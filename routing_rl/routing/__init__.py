"""Routing policies.

DQNPolicy needs PyTorch, so it is imported lazily: shortest-path, congestion-aware and
tabular Q-learning work in an environment without torch. Asking for DQNPolicy without torch
raises a clear ImportError instead of breaking every other policy.
"""
from .base import RoutingPolicy
from .shortest_path import ShortestPathPolicy
from .congestion_aware import CongestionAwarePolicy
from .tabular_q import TabularQPolicy

# DQNPolicy is deliberately not listed here so `from routing_rl.routing import *` stays torch-free.
__all__ = ["RoutingPolicy", "ShortestPathPolicy", "CongestionAwarePolicy", "TabularQPolicy"]


def __getattr__(name):
    if name == "DQNPolicy":
        try:
            from .dqn.agent import DQNPolicy
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] == "torch":
                raise ImportError(
                    "The 'dqn' routing method requires PyTorch. Install it with `pip install torch` "
                    "or choose shortest_path, congestion_aware or tabular_q."
                ) from exc
            raise
        return DQNPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["DQNPolicy"])