"""Traffic generators: constant, Poisson, bursty and multi-flow.

Every (src, dst) flow owns its own fractional-arrival accumulator, so flows that share a source
never borrow packets from each other. In ``multi_flow`` mode each flow may set its own arrival
process with ``"type": "poisson" | "constant" | "bursty"`` (default poisson), plus optional
``burst_on``, ``burst_off`` and ``burst_multiplier``. All randomness comes from the generator
passed in, so a seed fully determines the traffic trace.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from routing_rl.network.packet import Packet, PacketClass

KINDS = ("poisson", "constant", "bursty")


class TrafficGenerator:
    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._accum: Dict[Tuple[str, str], float] = {}
        self.emitted = 0
        self.rate_multiplier = 1.0   # live demo: scale offered load (1.0 = unchanged behaviour)
        self.emitted_by_flow: Dict[Tuple[str, str], int] = {}
        self._flows = self._build_flows()

    def _build_flows(self) -> List[Dict[str, Any]]:
        c = self.cfg
        base = dict(burst_on=c.burst_on, burst_off=c.burst_off, burst_mult=c.burst_multiplier)
        flows = []
        if c.type == "multi_flow" and c.flows:
            default_kind = getattr(c, "flow_type", None) or "poisson"
            for f in c.flows:
                flows.append({**base, "src": f["src"], "dst": f["dst"],
                              "rate": f.get("rate", c.rate_pps),
                              "kind": f.get("type", default_kind),
                              "burst_on": f.get("burst_on", c.burst_on),
                              "burst_off": f.get("burst_off", c.burst_off),
                              "burst_mult": f.get("burst_multiplier", c.burst_multiplier)})
        else:
            for src in c.sources:
                for dst in c.destinations:
                    if src != dst:
                        flows.append({**base, "src": src, "dst": dst, "rate": c.rate_pps, "kind": c.type})
        for fl in flows:
            if fl["kind"] not in KINDS:
                raise ValueError(f"Unknown traffic type {fl['kind']!r}; expected one of {KINDS}")
        return flows

    def _rate(self, fl: Dict[str, Any], t: float) -> float:
        rate = fl["rate"] * self.rate_multiplier
        if fl["kind"] == "bursty":
            period = fl["burst_on"] + fl["burst_off"]
            if period > 0 and (t % period) < fl["burst_on"]:
                rate *= fl["burst_mult"]
        return rate

    def generate(self, t: float, dt: float, ttl: float) -> List[Tuple[str, str, Packet]]:
        out: List[Tuple[str, str, Packet]] = []
        for fl in self._flows:
            out.extend(self._emit(fl, t, dt, ttl))
        return out

    def _emit(self, fl: Dict[str, Any], t: float, dt: float, ttl: float):
        src, dst, key = fl["src"], fl["dst"], (fl["src"], fl["dst"])
        rate = self._rate(fl, t)
        if fl["kind"] == "poisson":
            n = int(self.rng.poisson(rate * dt))
        else:  # constant / bursty: deterministic, per-flow fractional accumulator
            acc = self._accum.get(key, 0.0) + rate * dt
            n = int(acc + 1e-9)  # epsilon guards against 0.1*10 == 0.9999999
            self._accum[key] = max(0.0, acc - n)
        out = []
        for _ in range(n):
            pclass = (PacketClass.PRIORITY if self.rng.random() < self.cfg.priority_ratio
                      else PacketClass.BEST_EFFORT)
            out.append((src, dst, Packet(src=src, dst=dst, creation_time=t, ttl=ttl, pclass=pclass)))
        self.emitted += n
        self.emitted_by_flow[key] = self.emitted_by_flow.get(key, 0) + n
        return out