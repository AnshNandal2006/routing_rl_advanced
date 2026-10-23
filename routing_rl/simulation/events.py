"""Scheduled network events: link failures, capacity degradations, recoveries.

The effective state of a link is recomputed from *all* events active at time t, so overlapping
events on the same link behave correctly:
  * any active outage  -> link is down
  * else any active degradation -> link is up at the lowest degraded capacity
  * else -> link is up at its original capacity
A link is restored only when the last event affecting it has ended.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

Key = Tuple[str, str]
State = Tuple[bool, Optional[float]]  # (up, degraded capacity or None)
_NORMAL: State = (True, None)


def _key(link) -> Key:
    a, b = link
    return (a, b) if a <= b else (b, a)


@dataclass
class LinkFailureEvent:
    t_start: float
    t_end: float
    link: Tuple[str, str]                    # both directions are affected
    capacity_during: Optional[float] = None  # None -> fully down


@dataclass
class EventScheduler:
    events: List[LinkFailureEvent] = field(default_factory=list)
    _applied: Dict[Key, State] = field(default_factory=dict)
    last_changes: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, event: LinkFailureEvent) -> None:
        if event.t_end <= event.t_start:
            raise ValueError(f"Event on {event.link} must end after it starts")
        event.link = tuple(event.link)
        self.events.append(event)

    def from_config(self, schedule: List[Dict[str, Any]]) -> None:
        for e in schedule:
            self.add(LinkFailureEvent(float(e["t_start"]), float(e["t_end"]), tuple(e["link"]),
                                      e.get("capacity_during")))

    def active_events(self, t: float) -> List[LinkFailureEvent]:
        return [e for e in self.events if e.t_start <= t < e.t_end]

    def _effective(self, key: Key, t: float) -> State:
        active = [e for e in self.events if _key(e.link) == key and e.t_start <= t < e.t_end]
        if not active:
            return _NORMAL
        if any(e.capacity_during is None for e in active):
            return (False, None)
        return (True, min(float(e.capacity_during) for e in active))

    def apply(self, t: float, links: Dict, policy=None) -> List[str]:
        """Bring every affected link to its effective state. Returns human-readable messages.

        Structured details of what changed are left in ``self.last_changes``.
        """
        messages: List[str] = []
        self.last_changes = []
        for key in sorted({_key(e.link) for e in self.events}):
            state, prev = self._effective(key, t), self._applied.get(key, _NORMAL)
            if state == prev:
                continue
            self._applied[key] = state
            self._set_link(links, key, *state)
            up, cap = state
            what = "FAILED" if not up else (f"capacity→{cap:g}" if cap is not None else "RECOVERED")
            messages.append(f"t={t:.1f}: Link {key[0]}↔{key[1]} {what}")
            self.last_changes.append({"t": t, "link": key, "up": up, "capacity": cap,
                                      "prev_normal": prev == _NORMAL, "normal": state == _NORMAL})
        if self.last_changes and policy is not None:
            policy.on_link_change()
        return messages

    @staticmethod
    def _set_link(links, key: Key, up: bool, capacity: Optional[float]) -> None:
        a, b = key
        for pair in ((a, b), (b, a)):
            link = links.get(pair)
            if link is None:
                continue
            if not up:
                link.set_status(False)
                continue
            if not link.is_up:
                link.set_status(True)
            if capacity is None:
                link.reset_capacity()
            else:
                link.set_capacity(capacity)