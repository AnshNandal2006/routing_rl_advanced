"""
Deep Q-Network routing policy (Double + Dueling DQN).

State features (per decision):
  - one-hot current router
  - one-hot destination
  - local queue fraction + level
  - neighbor queue fractions (padded)
  - neighbor link residual capacities (padded)
  - hop count so far (normalized)
  - packet priority flag

Action space is the ordered list of neighbors (padded to max_degree).
Invalid actions are masked.
"""
from __future__ import annotations
import random
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from routing_rl.routing.base import RoutingPolicy
from routing_rl.network.packet import Packet
from routing_rl.network.router import Router
from routing_rl.config import DQNConfig
from .network import DuelingDQN
from .replay import ReplayBuffer


class DQNPolicy(RoutingPolicy):
    def __init__(
        self,
        routers: Dict[str, Router],
        links: Dict[Tuple[str, str], Any],
        cfg: Optional[DQNConfig] = None,
        reward_cfg: Optional[Dict] = None,
        train: bool = True,
    ):
        super().__init__(routers, links, reward_cfg)
        self.cfg = cfg or DQNConfig()
        self.train_mode = train
        self.device = torch.device(self.cfg.device)

        # Topology metadata
        self.router_ids = sorted(routers.keys())
        self.rid2idx = {r: i for i, r in enumerate(self.router_ids)}
        self.n_routers = len(self.router_ids)
        self.max_degree = max(len(r.neighbors) for r in routers.values()) or 1

        # Ordered neighbor lists (stable action indexing)
        self.neighbor_lists: Dict[str, List[str]] = {
            rid: list(r.neighbors) for rid, r in routers.items()
        }

        # State dimension
        # one-hot router + one-hot dest + queue_frac + queue_level_norm
        # + max_degree neighbor_q + max_degree residual + hop_norm + priority
        self.state_dim = (
            self.n_routers          # current router one-hot
            + self.n_routers        # destination one-hot
            + 2                     # queue_frac, queue_level
            + self.max_degree       # neighbor queue fractions
            + self.max_degree       # residual capacities
            + 2                     # hop_norm, priority
        )
        self.action_dim = self.max_degree

        self.policy_net = DuelingDQN(
            self.state_dim, self.action_dim, self.cfg.hidden_dims, self.cfg.dueling
        ).to(self.device)
        self.target_net = DuelingDQN(
            self.state_dim, self.action_dim, self.cfg.hidden_dims, self.cfg.dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)

        self.epsilon = self.cfg.epsilon_start
        self.step_count = 0
        self.train_steps = 0
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []

        # Reward coeffs
        rc = self.reward_cfg or {}
        self.r_hop = rc.get("hop_penalty", -1.0)
        self.r_queue = rc.get("queue_penalty_scale", -2.5)
        self.r_delivery = rc.get("delivery_bonus", 40.0)
        self.r_drop = rc.get("drop_penalty", -60.0)
        self.r_latency = rc.get("latency_scale", -0.5)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _encode_state(self, router: Router, packet: Packet) -> np.ndarray:
        s = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0
        # current router one-hot
        s[idx + self.rid2idx[router.id]] = 1.0
        idx += self.n_routers
        # destination one-hot
        if packet.dst in self.rid2idx:
            s[idx + self.rid2idx[packet.dst]] = 1.0
        idx += self.n_routers
        # local queue
        s[idx] = router.queue_fraction()
        s[idx + 1] = router.queue_level() / 2.0
        idx += 2
        # neighbor features
        neighs = self.neighbor_lists[router.id]
        for i in range(self.max_degree):
            if i < len(neighs):
                n = neighs[i]
                s[idx + i] = self.routers[n].queue_fraction()
                link = self.links.get((router.id, n))
                s[idx + self.max_degree + i] = (
                    link.residual_capacity_ratio() if link and link.is_up else 0.0
                )
            # else zeros
        idx += 2 * self.max_degree
        # hop & priority
        s[idx] = min(1.0, packet.hop_count / 10.0)
        s[idx + 1] = 1.0 if packet.is_priority else 0.0
        return s

    def _valid_mask(self, router: Router, packet: Packet) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.float32)
        neighs = self.neighbor_lists[router.id]
        for i, n in enumerate(neighs):
            link = self.links.get((router.id, n))
            if link and link.is_up and n not in packet.path:
                mask[i] = 1.0
        # Fallback: allow visited if nothing else (rare)
        if mask.sum() == 0:
            for i, n in enumerate(neighs):
                link = self.links.get((router.id, n))
                if link and link.is_up:
                    mask[i] = 1.0
        return mask

    def _action_to_hop(self, router: Router, action_idx: int) -> Optional[str]:
        neighs = self.neighbor_lists[router.id]
        if 0 <= action_idx < len(neighs):
            return neighs[action_idx]
        return None

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def select_next_hop(
        self, packet: Packet, router: Router, now: float
    ) -> Optional[str]:
        state = self._encode_state(router, packet)
        mask = self._valid_mask(router, packet)
        if mask.sum() == 0:
            return None

        if self.train_mode and random.random() < self.epsilon:
            # explore among valid
            valid_idxs = np.where(mask > 0)[0]
            action = int(random.choice(valid_idxs))
        else:
            with torch.no_grad():
                t = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q = self.policy_net(t).cpu().numpy()[0]
                q[mask == 0] = -1e9
                action = int(np.argmax(q))

        next_hop = self._action_to_hop(router, action)
        if next_hop is None:
            return None

        packet.pending_state = state
        packet.pending_action = action
        packet.pending_router = router.id
        # also store mask for next state later
        packet._pending_mask = mask  # type: ignore
        return next_hop

    def on_hop_forward(
        self, packet: Packet, router_id: str, next_hop: str, now: float
    ) -> None:
        if not self.train_mode or packet.pending_state is None:
            return
        next_r = self.routers[next_hop]
        next_state = self._encode_state(next_r, packet)
        next_mask = self._valid_mask(next_r, packet)

        reward = self.r_hop + self.r_queue * next_r.queue_level()
        if packet.is_priority:
            reward *= 1.2
        # mild latency shaping
        reward += self.r_latency * packet.age(now) * 0.1

        self.buffer.push(
            packet.pending_state,
            int(packet.pending_action),
            reward,
            next_state,
            False,
            next_mask,
        )
        self.reward_history.append(reward)
        self._maybe_train()
        self._decay_epsilon()

    def on_packet_delivered(self, packet: Packet, now: float) -> None:
        if not self.train_mode or packet.pending_state is None:
            return
        bonus = self.r_delivery
        if packet.is_priority:
            bonus *= 1.4
        # extra bonus for low latency
        latency = packet.age(now)
        bonus += max(0, 10.0 - latency * 2)
        dummy_next = np.zeros(self.state_dim, dtype=np.float32)
        dummy_mask = np.zeros(self.action_dim, dtype=np.float32)
        self.buffer.push(
            packet.pending_state,
            int(packet.pending_action),
            bonus,
            dummy_next,
            True,
            dummy_mask,
        )
        self.reward_history.append(bonus)
        self._maybe_train()

    def on_packet_dropped(self, packet: Packet, reason: str, now: float) -> None:
        if not self.train_mode or packet.pending_state is None:
            return
        dummy_next = np.zeros(self.state_dim, dtype=np.float32)
        dummy_mask = np.zeros(self.action_dim, dtype=np.float32)
        self.buffer.push(
            packet.pending_state,
            int(packet.pending_action),
            self.r_drop,
            dummy_next,
            True,
            dummy_mask,
        )
        self.reward_history.append(self.r_drop)
        self._maybe_train()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _maybe_train(self) -> None:
        self.step_count += 1
        if len(self.buffer) < self.cfg.min_buffer:
            return
        if self.step_count % self.cfg.train_every != 0:
            return
        self._train_step()
        if self.train_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self) -> None:
        batch = self.buffer.sample(self.cfg.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(np.stack([b[5] for b in batch])).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: action selection from policy, evaluation from target
                next_q_policy = self.policy_net(next_states)
                next_q_policy[masks == 0] = -1e9
                best_actions = next_q_policy.argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                next_q_target = self.target_net(next_states)
                next_q_target[masks == 0] = -1e9
                next_q = next_q_target.max(dim=1)[0]
            targets = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        self.train_steps += 1
        self.loss_history.append(float(loss.item()))

    def _decay_epsilon(self) -> None:
        if self.epsilon > self.cfg.epsilon_end:
            self.epsilon = max(
                self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay
            )

    def on_link_change(self) -> None:
        # Neighbor lists may need refresh if topology truly changes
        for rid, r in self.routers.items():
            self.neighbor_lists[rid] = list(r.neighbors)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "train_steps": self.train_steps,
            "avg_loss": float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
            "avg_reward": float(np.mean(self.reward_history[-200:])) if self.reward_history else 0.0,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", self.cfg.epsilon_end)
        self.train_mode = False
