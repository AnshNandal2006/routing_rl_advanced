# Adaptive Network Routing with Deep Reinforcement Learning

**Version 2.0 — Research-grade simulation framework**

A complete rewrite of classical vs. learning-based routing under dynamic network conditions.  
Compares **Shortest-Path**, **Congestion-Aware heuristics**, **Tabular Q-Learning**, and **Double Dueling DQN**.

---

## Why this is advanced

| Capability | Details |
|------------|---------|
| **Double Dueling DQN** | Value + advantage streams, Double-DQN targets, experience replay, ε-decay |
| **Rich observation** | Router/dest one-hots · local + neighbor queues · residual link capacity · hop count · packet priority |
| **Shaped multi-objective reward** | Hop cost + congestion penalty + delivery bonus + latency shaping + priority boost |
| **Dynamic topology events** | Scheduled link failures / capacity drops mid-episode |
| **QoS classes** | Priority vs best-effort traffic with differentiated treatment |
| **Multiple topologies** | Diamond (asymmetric), Mesh, Scale-free (BA), Hierarchical, Grid, Ring |
| **Interactive Command Center** | Streamlit + Plotly time-machine, live heat-map topology, training diagnostics |

---

## Project layout

```
routing_rl_advanced/
├── config/default.yaml          # Full experiment configuration
├── routing_rl/
│   ├── network/                 # Packet, Link, Router, Topology builders
│   ├── routing/
│   │   ├── shortest_path.py
│   │   ├── congestion_aware.py
│   │   ├── tabular_q.py
│   │   └── dqn/                 # Neural net, replay buffer, agent
│   ├── simulation/              # Traffic, event scheduler, discrete-time sim
│   └── metrics/                 # Time-series + summary collector
├── dashboard/app.py             # Advanced Streamlit UI
│   └── neurorouter_app.py        # Refined animated NeuroRouter dashboard
├── scripts/run_comparison.py    # Side-by-side benchmark + plots
├── main.py                      # Single-run CLI
└── requirements.txt
```

---

## Quick start

```bash
cd routing_rl_advanced
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Single experiment (uses config/default.yaml)
python main.py

# Full comparison of all 4 methods
python scripts/run_comparison.py

# Interactive Streamlit dashboard
streamlit run dashboard/app.py

# Refined NeuroRouter desktop/browser dashboard (new)
python dashboard/neurorouter_app.py
# Optional native desktop window: pip install pywebview
```

---

## DQN design (summary)

**State** (continuous vector):
- One-hot current router + one-hot destination
- Local queue fraction & discrete level
- Neighbor queue fractions (padded to max degree)
- Neighbor residual capacities
- Normalized hop count so far
- Priority flag

**Action**: index into the ordered neighbor list (invalid actions masked).

**Reward**:
- `−1` per hop
- `−2.5 × queue_level` of next router
- `+45` (scaled) on successful delivery
- `−55` on drop / expiry
- Mild latency shaping + priority multiplier

**Algorithm**: Dueling architecture + Double DQN target + experience replay + ε-greedy with exponential decay.

---

## Example results (diamond topology, Poisson 28 pps, link failure 30–45 s)

| Method                 | Delivery | Avg Delay | P95 Delay | Loss  |
|------------------------|----------|-----------|-----------|-------|
| Shortest Path          | ~0.30    | high      | high      | high  |
| Congestion-Aware       | ~0.99    | low       | low       | ~0    |
| Tabular Q-Learning     | ~0.97    | low-med   | low       | low   |
| **Double Dueling DQN** | **~0.98+** | **competitive** | low | low |

*(Exact numbers depend on seed and training length; DQN continues improving with longer runs.)*

---

## Extending the system

- Add a new topology in `network/topology.py`
- Add a new policy implementing `RoutingPolicy`
- Change reward coefficients in `config/default.yaml` under `routing:`
- Inject custom failure schedules via the `failure_schedule` list

---

## Citation / academic use

This framework is designed for coursework and research prototyping.  
Feel free to extend the state features, try multi-agent independent learners, or replace the tabular/DQN agents with PPO / SAC.
