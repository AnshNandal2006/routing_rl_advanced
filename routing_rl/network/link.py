"""Directional link: token-bucket capacity, propagation delay, failures, measured utilization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from .packet import Packet


@dataclass
class InTransitPacket:
    packet: Packet
    send_time: float
    arrival_time: float


class Link:
    """Store-and-forward link.

    Failure semantics: when a link goes down, every packet in flight is removed from the link and
    kept in an internal "lost" list. The simulator drains it with ``pop_lost()`` so each packet is
    counted exactly once (reason ``link_failure``) and the routing policy is told about it.
    """

    MAX_IN_TRANSIT = 800

    def __init__(self, src: str, dst: str, capacity_pps: float = 80.0,
                 propagation_delay: float = 0.005, name: Optional[str] = None):
        self.src, self.dst = src, dst
        self.capacity_pps = capacity_pps
        self.base_capacity = capacity_pps
        self.propagation_delay = propagation_delay
        self.name = name or f"{src}->{dst}"
        self.is_up: bool = True
        self._carry: float = 0.0
        self._next_tx_start: float = 0.0
        self._in_transit: List[InTransitPacket] = []
        self._lost: List[Packet] = []
        self._sent_this_step: int = 0
        self.packets_sent: int = 0
        self.packets_lost: int = 0
        self.bytes_sent: float = 0.0
        self.utilization_ema: float = 0.0  # smoothed fraction of capacity actually used

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        if self.is_up:
            self._carry = min(self._carry + self.capacity_pps * dt,
                              max(1.0, self.capacity_pps * dt * 2))
            # measured use: packets sent last step over what the capacity allows per step
            used = min(1.0, self._sent_this_step / max(1e-9, self.capacity_pps * dt))
            alpha = min(1.0, dt * 2.0)
            self.utilization_ema = (1 - alpha) * self.utilization_ema + alpha * used
        else:
            self._carry = 0.0
            self.utilization_ema = 0.0
        self._sent_this_step = 0

    def can_accept(self) -> bool:
        return self.is_up and self._carry >= 1.0 and len(self._in_transit) < self.MAX_IN_TRANSIT

    def residual_capacity_ratio(self) -> float:
        """Instantaneous token availability (used by the DQN state encoder)."""
        if not self.is_up or self.capacity_pps <= 0:
            return 0.0
        return min(1.0, self._carry / max(1.0, self.capacity_pps * 0.05))

    def available_capacity_pps(self) -> float:
        """Capacity not currently used, in packets/s, from the smoothed utilization."""
        if not self.is_up:
            return 0.0
        return self.capacity_pps * (1.0 - min(1.0, self.utilization_ema))

    def backlog_delay(self, now: float) -> float:
        """Seconds until the transmitter finishes everything already scheduled."""
        return max(0.0, self._next_tx_start - now)

    def send(self, packet: Packet, now: float) -> bool:
        if not self.can_accept():
            return False
        self._carry -= 1.0
        start = max(now, self._next_tx_start)
        end = start + 1.0 / max(1e-6, self.capacity_pps)
        self._next_tx_start = end
        self._in_transit.append(InTransitPacket(packet, now, end + self.propagation_delay))
        self._sent_this_step += 1
        self.packets_sent += 1
        self.bytes_sent += packet.size
        return True

    def deliver_ready(self, now: float) -> List[InTransitPacket]:
        ready = [it for it in self._in_transit if it.arrival_time <= now]
        if ready:
            self._in_transit = [it for it in self._in_transit if it.arrival_time > now]
        return ready

    def set_status(self, up: bool) -> List[Packet]:
        """Change link state. Going down returns (and records) the packets that were in flight."""
        self.is_up = up
        if up:
            return []
        self._carry = 0.0
        self._next_tx_start = 0.0
        self.utilization_ema = 0.0
        self._sent_this_step = 0
        lost = [it.packet for it in self._in_transit]
        self._in_transit.clear()
        self._lost.extend(lost)
        self.packets_lost += len(lost)
        return lost

    def pop_lost(self) -> List[Packet]:
        lost, self._lost = self._lost, []
        return lost

    def set_capacity(self, capacity: float) -> None:
        self.capacity_pps = max(1.0, capacity)

    def reset_capacity(self) -> None:
        self.capacity_pps = self.base_capacity

    @property
    def in_transit_count(self) -> int:
        return len(self._in_transit)

    @property
    def utilization(self) -> float:
        return self.utilization_ema

    @property
    def load(self) -> float:
        return len(self._in_transit) / self.MAX_IN_TRANSIT