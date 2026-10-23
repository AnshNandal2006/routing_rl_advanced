# Project Code Review — Adaptive Network Routing with Reinforcement Learning

## Scope and validation

Reviewed the Python source in `routing_rl/`, configuration, CLI/comparison scripts, the existing Streamlit dashboard, and the added `dashboard/neurorouter_app.py` refinement. `python -m compileall -q .` passes. Short smoke simulations ran successfully for `shortest_path`, `congestion_aware`, `tabular_q`, and `dqn` with the installed dependencies. This is a source-level and smoke-test review, not proof of production readiness or a full experiment reproduction.

## Dashboard integration

- Kept the existing Streamlit/Plotly dashboard at `dashboard/app.py`.
- Added the supplied animated NeuroRouter dashboard as `dashboard/neurorouter_app.py` so the two interfaces can coexist.
- Updated README launch instructions. Run the existing dashboard with `streamlit run dashboard/app.py`; run the refinement with `python dashboard/neurorouter_app.py`. `pywebview` is optional; without it, the app opens in a browser.
- Added explicit 404 handling for unknown POST API routes in the new dashboard.

## Architecture walkthrough

- `config.py` and `config/default.yaml`: typed experiment configuration and defaults for topology, traffic, simulation, routing, and DQN.
- `network/packet.py`: packet identity, source/destination, TTL/expiry, priority class, path and routing-transition fields.
- `network/router.py`: router queues and queue-level/queue-fraction observations.
- `network/link.py`: directional links with token-bucket-like send capacity, propagation delay, in-flight packets, utilization estimate, and failure behavior.
- `network/topology.py`: constructs named graph topologies and layout positions used by the dashboards.
- `simulation/traffic.py`: Poisson, constant, bursty and multi-flow packet generation.
- `simulation/events.py`: scheduled link failures or capacity changes, applied to both directions when matching links exist.
- `simulation/simulator.py`: discrete-time loop; events, link capacity, traffic generation, packet delivery, forwarding and TTL expiry; captures history for the time-machine dashboard.
- `routing/base.py`: shared policy interface and callbacks.
- `routing/shortest_path.py`: non-learning baseline.
- `routing/congestion_aware.py`: queue/congestion-informed heuristic baseline.
- `routing/tabular_q.py`: tabular Q-learning with router/destination/queue-level states and epsilon-greedy action choice.
- `routing/dqn/network.py`, `replay.py`, `agent.py`: dueling neural Q-network, replay memory, epsilon-greedy action selection, masked invalid actions, Double-DQN target option, Huber loss and target-network updates.
- `metrics/collector.py`: counters, packet delay/hop statistics, sliding-window time-series and summary metrics.
- `scripts/run_comparison.py`: benchmark workflow for the four routing methods; `main.py` runs a configured experiment.
- `dashboard/app.py`: Streamlit controls, time-machine topology graph, Plotly metrics, DQN diagnostics and event log.
- `dashboard/neurorouter_app.py`: standalone local HTTP API and HTML/canvas animation, KPI cards, telemetry charts, queue heatmap, alerts, policy statistics and four-policy battle mode.

## What is genuinely implemented

- Four selectable policies: Double-Dueling DQN, tabular Q-learning, congestion-aware heuristic, and shortest-path baseline.
- Single shared policy object per simulation. The DQN uses a neural network and replay buffer; it does not create an independent learner per router.
- Epsilon-greedy exploration is implemented for DQN and tabular Q-learning.
- The DQN state includes current-router and destination one-hots, local queue features, neighbor queue features, residual link-capacity features, hop count, and priority.
- Invalid/down-link actions are masked in DQN selection; loop avoidance is attempted in both learning policies.
- Failure/capacity events can change links during a simulation; links are directional objects and event application attempts both directions.
- Priority packets, configurable traffic patterns, multiple topologies, time-series metrics, and side-by-side policy comparisons are present.
- Dashboard charts and animation are backed by simulation history/metrics, but the animated comets in the canvas view are decorative link-flow effects, not individual packet traces from the simulator.

## Findings and caveats to address before claiming research-grade results

1. **Metrics can be misleading for short runs / warm-up.** The collector excludes pre-warmup packets from delay samples but total generated/delivered counters include them. With a short smoke run, there may be no post-warmup delay samples, so `avg_delay`, percentiles, and `avg_hops` show zero even though packets were delivered. Test using durations longer than the configured warm-up and clarify whether summary ratios should include warm-up traffic.
2. **Instantaneous delivery ratio is not a strict per-packet success probability.** It divides deliveries in a five-second sliding window by generated packets in that window, with a denominator floor of one. Deliveries can correspond to packets generated earlier; values can be unintuitive and should be labeled as a windowed rate/ratio or redesigned with cohorts.
3. **`loss_ratio` definition should be checked.** It uses `(dropped + expired) / generated`, while some drop pathways may affect overlapping counters depending on how callbacks are recorded. Validate against packet-level accounting and explicitly define whether loops are included.
4. **Constant/bursty multi-destination generation shares an accumulator by source.** In `_emit`, the accumulator key is only `src`; for one source with multiple destinations, fractional arrivals are shared across flows rather than independently tracked per source-destination pair. Use `(src, dst)` keys if independent flow rates are intended.
5. **Multi-flow currently always uses Poisson sampling.** `_emit`'s condition explicitly routes `multi_flow` through Poisson generation regardless of any intended per-flow pattern. Document this behavior or implement a per-flow pattern field.
6. **Overlapping scheduled events on the same link can interfere.** Active events are keyed only by link pair; when one overlapping event ends, it may restore the link while another event should remain active. Merge schedules or track active events per link with precedence rules.
7. **DQN terminal credit assignment is simplified.** The hop callback stores a non-terminal transition and the later delivery/drop callback also rewards the last pending action. This can be workable as shaping, but the transition semantics should be documented/tested to avoid unintended duplicate credit.
8. **The simulation is a simplified model, not a real network stack.** It models packet movement and link capacity in discrete time; it does not implement real routing protocols, sockets, TCP/IP stack behavior, or deployment to routers.
9. **The new dashboard is local-only.** It binds to `127.0.0.1` and runs simulations synchronously inside HTTP request handlers. Long simulations can block that request and concurrent requests can run expensive simulations at once. Suitable for a local demo; not a multi-user server.
10. **Battle mode trains DQN during its run.** `run_sim()` enables training for DQN; comparison scores can depend on seed and training length. For fair evaluation, separate training from evaluation, use fixed checkpoints or identical train/eval protocols, and repeat across multiple seeds.
11. **Reproducibility is partial.** NumPy's generator is seeded, but Python's `random` and PyTorch randomness should also be seeded for fully repeatable Q-learning/DQN behavior. Report mean and variation over several seeds, not one run.
12. **Score is a custom composite metric.** The dashboard's `score_of()` combines delivery, normalized delay, loss, and priority delivery with hand-chosen weights. It is useful for ranking within the app, but not a standard benchmark score; display its formula and weights in the UI/report.
13. **Topology/source-destination assumptions need validation.** Generic topology selection uses sorted router IDs for source and destination; this is deterministic but may not represent meaningful traffic endpoints for every topology. Validate that each configured failure link exists in each topology.
14. **Failure clears in-flight packets.** `Link.set_status(False)` deletes packets already in transit without directly notifying the metrics collector. Depending on desired semantics, these packets may disappear from packet accounting rather than be recorded as drops.
15. **Dashboard metric labels should be precise.** The new UI shows throughput as the delivered-window count divided by a fixed five-second window, and the Streamlit dashboard describes some telemetry as real-time even though it is generated after the full simulation and then scrubbed through recorded history.

## Smoke-test result

A two-second smoke run completed without exceptions for all four methods. Its metrics were not suitable for performance conclusions because the configured warm-up left no delay/hop samples in that short interval. Run longer simulations and inspect warm-up configuration before interpreting scores.

## Recommended next steps

1. Run the refined dashboard on the target machine and verify all topology choices, failure schedules, and battle mode.
2. Fix/define metric accounting and warm-up behavior before using numerical claims in a report.
3. Add automated tests for packet conservation, link failure/recovery, action masking, loop prevention, and event overlap.
4. Seed Python `random`, NumPy, and PyTorch consistently; separate training and evaluation for fair comparisons.
5. Document that the system is a simulator/prototype rather than a deployed network-routing implementation.
