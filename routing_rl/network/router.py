"""Router with priority-aware queue and rich local observations."""
from __future__ import annotations
from collections import deque
from typing import Deque, List, Optional, Dict, Any, Tuple
from .packet import Packet, PacketClass


class Router:
    def __init__(
        self,
        router_id: str,
        max_queue_size: int = 150,
        pos: Optional[Tuple[float, float]] = None,
    ):
        self.id = router_id
        self.max_queue_size = max_queue_size
        self.pos = pos
        # Two logical queues for simple priority (priority first)
        self._q_priority: Deque[Packet] = deque()
        self._q_best: Deque[Packet] = deque()
        self.neighbors: List[str] = []
        self.policy = None
        # Stats
        self.packets_dropped: int = 0
        self.packets_received: int = 0
        self.packets_forwarded: int = 0
        self.total_queue_time: float = 0.0

    # ------------------------------------------------------------------
    @property
    def queue(self) -> Deque[Packet]:
        """Compatibility view — priority packets appear first."""
        return deque(list(self._q_priority) + list(self._q_best))

    def __len__(self) -> int:
        return len(self._q_priority) + len(self._q_best)

    def enqueue(self, packet: Packet) -> bool:
        if len(self) >= self.max_queue_size:
            self.packets_dropped += 1
            return False
        packet.current_router = self.id
        if packet.pclass == PacketClass.PRIORITY:
            self._q_priority.append(packet)
        else:
            self._q_best.append(packet)
        self.packets_received += 1
        return True

    def has_packets(self) -> bool:
        return len(self) > 0

    def peek(self) -> Optional[Packet]:
        if self._q_priority:
            return self._q_priority[0]
        if self._q_best:
            return self._q_best[0]
        return None

    def pop(self) -> Optional[Packet]:
        if self._q_priority:
            pkt = self._q_priority.popleft()
            self.packets_forwarded += 1
            return pkt
        if self._q_best:
            pkt = self._q_best.popleft()
            self.packets_forwarded += 1
            return pkt
        return None

    def queue_level(self, bins: int = 3) -> int:
        """Discrete congestion level 0..bins-1."""
        frac = len(self) / max(1, self.max_queue_size)
        for i in range(bins):
            if frac < (i + 1) / bins:
                return i
        return bins - 1

    def queue_fraction(self) -> float:
        return len(self) / max(1, self.max_queue_size)

    def observation(self) -> Dict[str, Any]:
        """Local state features useful for RL."""
        return {
            "id": self.id,
            "queue_len": len(self),
            "queue_frac": self.queue_fraction(),
            "queue_level": self.queue_level(),
            "n_neighbors": len(self.neighbors),
            "priority_pending": len(self._q_priority),
        }

    def clear(self) -> None:
        self._q_priority.clear()
        self._q_best.clear()
