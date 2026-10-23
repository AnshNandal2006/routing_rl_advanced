"""Central configuration system with rich defaults and YAML loading."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import json


@dataclass
class SimulationConfig:
    duration: float = 120.0
    dt: float = 0.01
    warmup: float = 10.0
    seed: int = 42
    sample_every: int = 5          # steps between metric snapshots
    verbose: bool = False


@dataclass
class TrafficConfig:
    type: str = "poisson"          # constant | poisson | bursty | multi_flow
    sources: List[str] = field(default_factory=lambda: ["S0", "S1"])
    destinations: List[str] = field(default_factory=lambda: ["D0", "D1"])
    rate_pps: float = 25.0
    burst_on: float = 2.0
    burst_off: float = 3.0
    burst_multiplier: float = 4.0
    # Multi-flow: list of (src, dst, rate) triples
    flows: List[Dict[str, Any]] = field(default_factory=list)
    priority_ratio: float = 0.15   # fraction of high-priority packets


@dataclass
class NetworkConfig:
    topology: str = "geometric"    # geometric | diamond | mesh | scale_free | hierarchical | grid | ring
    default_queue_size: int = 150
    default_ttl: float = 8.0
    default_capacity: float = 80.0
    propagation_delay: float = 0.005
    n_nodes_min: int = 10
    n_nodes_max: int = 30
    # If set, geometric layouts are reproducible; None → a fresh random topology.
    topology_seed: Optional[int] = None
    # Dynamic events
    enable_failures: bool = True
    failure_schedule: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"t_start": 30.0, "t_end": 45.0, "link": ["S0", "R1"]}]


@dataclass
class DQNConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    learning_rate: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_freq: int = 200
    train_every: int = 4
    min_buffer: int = 500
    double_dqn: bool = True
    dueling: bool = True
    device: str = "cpu"


@dataclass
class RoutingConfig:
    method: str = "dqn"            # shortest_path | congestion_aware | tabular_q | dqn
    alpha: float = 1.5             # congestion weight for heuristic
    # Shared reward design
    hop_penalty: float = -1.0
    queue_penalty_scale: float = -2.5
    delivery_bonus: float = 40.0
    drop_penalty: float = -60.0
    latency_scale: float = -0.5
    dqn: DQNConfig = field(default_factory=DQNConfig)


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        sim = SimulationConfig(**raw.get("simulation", {}))
        traf = TrafficConfig(**raw.get("traffic", {}))
        net = NetworkConfig(**raw.get("network", {}))
        rout_raw = raw.get("routing", {})
        dqn_raw = rout_raw.pop("dqn", {}) if isinstance(rout_raw, dict) else {}
        dqn = DQNConfig(**dqn_raw)
        rout = RoutingConfig(**{**rout_raw, "dqn": dqn})
        return cls(simulation=sim, traffic=traf, network=net, routing=rout)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
