"""Packet model with QoS class, path history, and RL bookkeeping."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Any
import itertools

_packet_id_gen = itertools.count(1)


class PacketClass(IntEnum):
    BEST_EFFORT = 0
    PRIORITY = 1


@dataclass
class Packet:
    src: str
    dst: str
    creation_time: float
    ttl: float
    packet_id: int = field(default_factory=lambda: next(_packet_id_gen))
    pclass: PacketClass = PacketClass.BEST_EFFORT
    size: float = 1.0                 # normalized size units
    current_router: str = ""
    path: List[str] = field(default_factory=list)
    # RL bookkeeping (set by agent on decision)
    pending_state: Optional[Any] = None
    pending_action: Optional[int] = None
    pending_router: Optional[str] = None
    # Diagnostics
    hop_count: int = 0
    queue_wait_time: float = 0.0

    def __post_init__(self):
        if not self.current_router:
            self.current_router = self.src

    @property
    def expiry_time(self) -> float:
        return self.creation_time + self.ttl

    @property
    def is_priority(self) -> bool:
        return self.pclass == PacketClass.PRIORITY

    def age(self, now: float) -> float:
        return now - self.creation_time

    def __repr__(self) -> str:
        cls = "P" if self.is_priority else "B"
        return f"Pkt#{self.packet_id}[{cls}]({self.src}->{self.dst})"
