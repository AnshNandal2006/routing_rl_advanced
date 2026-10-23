"""Tests for topology, simulator correctness, policies and the experiment tooling.

Run from the project root:   python -m unittest discover -s tests -v
DQN tests are skipped (not failed) when PyTorch is not installed.
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from routing_rl.config import Config
from routing_rl.experiment_utils import METHODS, apply_overrides, describe, paired, resolve_cfg, set_path
from routing_rl.metrics.collector import MetricsCollector
from routing_rl.network.link import Link
from routing_rl.network.packet import Packet
from routing_rl.network.topology import TOPOLOGY_CATALOG, build_topology, get_layout_positions, traffic_defaults
from routing_rl.simulation.events import EventScheduler, LinkFailureEvent
from routing_rl.simulation.simulator import Simulator
from routing_rl.simulation.traffic import TrafficGenerator

HAS_TORCH = importlib.util.find_spec("torch") is not None
DEFAULT_YAML = ROOT / "config" / "default.yaml"


def make_cfg(method="shortest_path", duration=12.0, warmup=1.0, fail=False, ttl=None, seed=7):
    cfg = Config.from_yaml(str(DEFAULT_YAML))
    cfg.routing.method = method
    cfg.simulation.duration, cfg.simulation.warmup, cfg.simulation.seed = duration, warmup, seed
    cfg.network.topology_seed = 314
    cfg.network.enable_failures = fail
    cfg.network.failure_schedule = [{"t_start": 4.0, "t_end": 8.0, "link": "auto"}] if fail else []
    if ttl is not None:
        cfg.network.default_ttl = ttl
    return resolve_cfg(cfg)


# ======================================================================== topology (kept)
class GeometricTopologyTests(unittest.TestCase):
    def test_seeded_topology_is_reproducible(self):
        a = build_topology("geometric", seed=1234, n_nodes_min=10, n_nodes_max=30)
        b = build_topology("geometric", seed=1234, n_nodes_min=10, n_nodes_max=30)
        self.assertEqual(list(a[0]), list(b[0]))
        self.assertEqual(set(a[1]), set(b[1]))
        self.assertEqual({k: v.pos for k, v in a[0].items()}, {k: v.pos for k, v in b[0].items()})
        self.assertEqual({k: v.capacity_pps for k, v in a[1].items()}, {k: v.capacity_pps for k, v in b[1].items()})

    def test_node_count_connectivity_labels_positions_and_capacities(self):
        for seed in range(20):
            routers, links, graph = build_topology("geometric", seed=seed, n_nodes_min=10, n_nodes_max=30)
            self.assertTrue(10 <= len(routers) <= 30)
            self.assertTrue(nx.is_weakly_connected(graph))
            self.assertEqual(set(routers), set(graph.nodes))
            self.assertEqual(sum(node == "S" for node in routers), 1)
            self.assertEqual(sum(node == "D" for node in routers), 1)
            self.assertEqual(len(routers) - 2, sum(node.startswith("R") for node in routers))
            self.assertTrue(all(router.pos is not None for router in routers.values()))
            self.assertTrue(all("pos" in graph.nodes[node] for node in routers))
            self.assertTrue(all(20 <= link.capacity_pps <= 100 for link in links.values()))
            self.assertEqual(len(links) % 2, 0)
            for u, v in links:
                self.assertIn((v, u), links)
            sources, destinations, failure = traffic_defaults("geometric", routers)
            self.assertEqual(sources, ["S"])
            self.assertEqual(destinations, ["D"])
            self.assertTrue(tuple(failure) in links or tuple(reversed(failure)) in links)
            self.assertEqual(set(get_layout_positions("geometric", routers)), set(routers))

    def test_custom_node_count_is_honored(self):
        routers, _, _ = build_topology("geometric", seed=7, n_nodes=17)
        self.assertEqual(len(routers), 17)

    def test_legacy_topologies_still_build(self):
        for name in TOPOLOGY_CATALOG:
            with self.subTest(topology=name):
                routers, links, graph = build_topology(name, seed=42)
                self.assertTrue(routers)
                self.assertTrue(links)
                self.assertEqual(set(routers), set(graph.nodes))
                self.assertTrue(nx.is_weakly_connected(graph))

    def test_all_routing_policies_initialize_and_run(self):
        base = Config.from_yaml(str(DEFAULT_YAML))
        base.simulation.duration, base.simulation.dt, base.simulation.warmup = 0.5, 0.05, 0.0
        base.network.topology_seed = 314
        base.network.enable_failures = True
        base.network.failure_schedule = [{"t_start": 0.1, "t_end": 0.2, "link": ["S", "R0"]}]
        for method in METHODS:
            with self.subTest(method=method):
                if method == "dqn" and not HAS_TORCH:
                    self.skipTest("PyTorch not installed")
                cfg = Config.from_dict(base.to_dict())
                cfg.routing.method = method
                sim = Simulator(cfg, train_rl=False)
                sim.run(progress=False)
                self.assertTrue(10 <= sim.n_nodes <= 30)
                self.assertEqual(sim.n_nodes, len(sim.routers))


# ======================================================================== optional PyTorch
class OptionalTorchTests(unittest.TestCase):
    def test_non_dqn_policies_work_without_torch_and_dqn_fails_clearly(self):
        code = ("import sys; sys.modules['torch']=None\n"
                "import routing_rl.routing as r; r.ShortestPathPolicy; r.CongestionAwarePolicy; r.TabularQPolicy\n"
                "try:\n    r.DQNPolicy\nexcept ImportError as e:\n    assert 'PyTorch' in str(e); print('ok')")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("ok", out.stdout, out.stderr)

    def test_simulator_runs_shortest_path_without_torch(self):
        code = ("import sys; sys.modules['torch']=None\n"
                "from routing_rl.config import Config\nfrom routing_rl.simulation.simulator import Simulator\n"
                "c=Config.from_yaml('config/default.yaml'); c.routing.method='shortest_path'; c.simulation.duration=1.0\n"
                "c.network.enable_failures=False; Simulator(c).run(progress=False); print('ran')")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("ran", out.stdout, out.stderr)


# ======================================================================== links, events, traffic
class LinkAndEventTests(unittest.TestCase):
    def test_failure_returns_in_flight_packets(self):
        link = Link("A", "B", 100.0)
        link.step(0.1)
        pkt = Packet(src="A", dst="B", creation_time=0.0, ttl=5.0)
        self.assertTrue(link.send(pkt, 0.0))
        self.assertEqual(link.set_status(False), [pkt])
        self.assertEqual(link.pop_lost(), [pkt])
        self.assertEqual(link.in_transit_count, 0)

    def test_overlapping_outages_restore_only_after_last_ends(self):
        links = {("A", "B"): Link("A", "B", 100.0), ("B", "A"): Link("B", "A", 100.0)}
        s = EventScheduler()
        s.add(LinkFailureEvent(10, 20, ("A", "B")))
        s.add(LinkFailureEvent(15, 30, ("B", "A")))  # reversed key, overlaps
        for t, up in [(5, True), (12, False), (25, False), (31, True)]:
            s.apply(t, links)
            self.assertEqual(links[("A", "B")].is_up, up, f"t={t}")
            self.assertEqual(links[("B", "A")].is_up, up, f"t={t}")

    def test_degradation_and_outage_stack_and_capacity_is_restored(self):
        links = {("A", "B"): Link("A", "B", 100.0), ("B", "A"): Link("B", "A", 100.0)}
        s = EventScheduler()
        s.add(LinkFailureEvent(0, 10, ("A", "B"), capacity_during=20.0))
        s.add(LinkFailureEvent(5, 8, ("A", "B")))
        l = links[("A", "B")]
        s.apply(1, links);  self.assertTrue(l.is_up);  self.assertEqual(l.capacity_pps, 20.0)
        s.apply(6, links);  self.assertFalse(l.is_up)
        s.apply(9, links);  self.assertTrue(l.is_up);  self.assertEqual(l.capacity_pps, 20.0)
        s.apply(11, links); self.assertTrue(l.is_up);  self.assertEqual(l.capacity_pps, 100.0)

    def test_event_must_end_after_it_starts(self):
        with self.assertRaises(ValueError):
            EventScheduler().add(LinkFailureEvent(5, 5, ("A", "B")))


class TrafficTests(unittest.TestCase):
    def _cfg(self, flows):
        return SimpleNamespace(sources=[], destinations=[], flows=flows, type="multi_flow", rate_pps=10.0,
                               priority_ratio=0.0, burst_on=1.0, burst_off=1.0, burst_multiplier=3.0)

    def test_multi_flow_constant_uses_independent_accumulators(self):
        flows = [{"src": "A", "dst": "B", "rate": 10, "type": "constant"},
                 {"src": "A", "dst": "C", "rate": 10, "type": "constant"}]
        gen = TrafficGenerator(self._cfg(flows), np.random.default_rng(1))
        c = Counter()
        for i in range(100):
            for s, d, _ in gen.generate(i * 0.01, 0.01, 5.0):
                c[(s, d)] += 1
        self.assertEqual(c[("A", "B")], 10)
        self.assertEqual(c[("A", "C")], 10)

    def test_same_seed_same_trace_and_different_seed_differs(self):
        flows = [{"src": "A", "dst": "B", "rate": 40}]
        def trace(seed):
            gen = TrafficGenerator(self._cfg(flows), np.random.default_rng(seed))
            return [len(gen.generate(i * 0.01, 0.01, 5.0)) for i in range(300)]
        self.assertEqual(trace(5), trace(5))
        self.assertNotEqual(trace(5), trace(6))

    def test_unknown_traffic_type_is_rejected(self):
        with self.assertRaises(ValueError):
            TrafficGenerator(self._cfg([{"src": "A", "dst": "B", "type": "weird"}]), np.random.default_rng(0))


# ======================================================================== metrics
class MetricsTests(unittest.TestCase):
    def test_exactly_once_and_warmup_cohort(self):
        m = MetricsCollector(warmup=1.0)
        ps = [Packet("A", "B", t, 5.0) for t in (0.5, 1.5, 2.0, 2.5)]
        for p in ps:
            m.record_generation(p, p.creation_time)
        m.record_delivery(ps[1], 2.0)
        m.record_drop(ps[2], "queue_full", 2.1)
        m.record_drop(ps[2], "queue_full", 2.2)   # duplicate outcome must be ignored
        m.record_expiry(ps[0], 3.0)               # created in warm-up: excluded from the cohort
        s = m.summary()
        self.assertEqual((s["generated"], s["delivered"], s["dropped"], s["expired"]), (3, 1, 1, 0))
        self.assertEqual(s["duplicate_terminal"], 1)
        self.assertEqual(m.raw, {"generated": 4, "delivered": 1, "dropped": 1, "expired": 1})
        self.assertAlmostEqual(s["delivery_ratio"] + s["loss_ratio"] + s["unresolved_ratio"], 1.0)

    def test_route_changes_count_only_real_changes_after_warmup(self):
        m = MetricsCollector(warmup=5.0)
        for t, hop in [(1, "X"), (2, "Y"), (6, "Y"), (7, "X"), (8, "X"), (9, "Z")]:
            m.record_route_choice("S", "D", hop, t)
        self.assertEqual(m.route_changes, 2)  # X->Y at t=2 is in warm-up; Y->X and X->Z count
        m.record_route_choice("T", "D", "A", 10)
        self.assertEqual(m.route_changes, 2)  # first choice of another router is not a change

    def _series(self, bad_from, bad_to):
        m = MetricsCollector(warmup=0.0)
        for t in np.arange(0.5, 60.01, 0.5):
            m.time_series["t"].append(float(t))
            m.time_series["delivery_ratio_inst"].append(0.2 if bad_from <= t < bad_to else 1.0)
            m.time_series["avg_delay"].append(0.05)
        return m

    def test_adaptation_time_measured_from_event(self):
        m = self._series(20, 30)
        m.mark_event(20.0, "disruption", "A-B")
        m.mark_event(30.0, "recovery", "A-B")
        rep = m.adaptation_report()
        self.assertAlmostEqual(rep[0]["adaptation_time"], 10.0, delta=0.6)
        self.assertAlmostEqual(rep[1]["adaptation_time"], 0.0, delta=0.6)
        self.assertGreater(rep[0]["delivery_drop"], 0.7)

    def test_never_recovering_run_reports_not_recovered(self):
        m = self._series(20, 1000)
        m.mark_event(20.0, "disruption", "A-B")
        rep = m.adaptation_report()
        self.assertFalse(rep[0]["recovered"])
        self.assertIsNone(rep[0]["adaptation_time"])


# ======================================================================== simulator
class SimulationTests(unittest.TestCase):
    def _run(self, method, **kw):
        sim = Simulator(make_cfg(method, **kw), strict_accounting=True)
        return sim, sim.run(progress=False).summary()

    def _methods(self):
        return [m for m in METHODS if m != "dqn" or HAS_TORCH]

    def test_accounting_is_exact_and_bounded_with_a_failure(self):
        for method in self._methods():
            with self.subTest(method=method):
                sim, s = self._run(method, fail=True)
                self.assertTrue(sim.accounting_report["balanced"], sim.accounting_report)
                self.assertEqual(s["duplicate_terminal"], 0)
                self.assertLessEqual(s["loss_ratio"], 1.0)
                self.assertAlmostEqual(s["delivery_ratio"] + s["loss_ratio"] + s["unresolved_ratio"], 1.0, places=9)
                self.assertGreater(s["delivered"], 0)
                self.assertEqual([a["kind"] for a in s["adaptation"]], ["disruption", "recovery"])

    def test_same_seed_is_reproducible(self):
        for method in self._methods():
            with self.subTest(method=method):
                a, b = self._run(method, fail=True)[1], self._run(method, fail=True)[1]
                for k in ("generated", "delivered", "dropped", "expired", "route_changes"):
                    self.assertEqual(a[k], b[k], k)

    def test_all_methods_see_identical_traffic(self):
        gens = {m: self._run(m)[1]["generated"] for m in ("shortest_path", "congestion_aware")}
        self.assertEqual(len(set(gens.values())), 1, gens)

    def test_ttl_is_enforced_for_packets_in_flight(self):
        sim, s = self._run("shortest_path", ttl=0.001)  # shorter than a single propagation delay
        self.assertEqual(s["delivered"], 0)
        self.assertGreater(s["expired"], 0)
        self.assertTrue(sim.accounting_report["balanced"])

    def test_disconnected_destination_is_handled_safely(self):
        for method in self._methods():
            with self.subTest(method=method):
                sim = Simulator(make_cfg(method, ttl=2.0), strict_accounting=True)
                for l in sim.links.values():
                    if "D" in (l.src, l.dst):
                        l.set_status(False)
                sim.policy.on_link_change()
                s = sim.run(progress=False).summary()
                self.assertEqual(s["delivered"], 0)
                self.assertGreater(s["expired"] + s["dropped"], 0)
                self.assertTrue(sim.accounting_report["balanced"], sim.accounting_report)


class PolicyTests(unittest.TestCase):
    def test_shortest_path_reports_unreachable_destination(self):
        sim = Simulator(make_cfg("shortest_path"))
        for l in sim.links.values():
            if "D" in (l.src, l.dst):
                l.set_status(False)
        sim.policy.on_link_change()
        pkt = Packet(src="S", dst="D", creation_time=0.0, ttl=50.0)
        self.assertFalse(sim.policy.is_reachable("S", "D"))
        self.assertIsNone(sim.policy.select_next_hop(pkt, sim.routers["S"], 0.0))

    def test_shortest_path_reroutes_when_a_link_fails_and_returns(self):
        sim = Simulator(make_cfg("shortest_path"))
        pkt = Packet(src="S", dst="D", creation_time=0.0, ttl=50.0)
        first = sim.policy.select_next_hop(pkt, sim.routers["S"], 0.0)
        sim.links[("S", first)].set_status(False)
        sim.policy.on_link_change()
        second = sim.policy.select_next_hop(pkt, sim.routers["S"], 0.0)
        self.assertNotEqual(first, second)  # None (bridge) or another neighbour, never the dead link
        sim.links[("S", first)].set_status(True)
        sim.policy.on_link_change()
        self.assertEqual(sim.policy.select_next_hop(pkt, sim.routers["S"], 0.0), first)

    def test_congestion_aware_moves_away_from_a_congested_path(self):
        sim = Simulator(make_cfg("congestion_aware"))
        pol, router = sim.policy, sim.routers["S"]
        pol.alpha = 100.0
        pkt = Packet(src="S", dst="D", creation_time=0.0, ttl=500.0)
        before = pol.candidate_costs(pkt, router)
        if len(before) < 2:
            self.skipTest("source has a single usable next hop in this topology")
        choice = min(before, key=lambda n: before[n]["cost"])
        if choice == "D":
            self.skipTest("destination is a direct neighbour")
        nb = sim.routers[choice]
        for _ in range(nb.max_queue_size):
            nb.enqueue(Packet(src=choice, dst="D", creation_time=0.0, ttl=500.0))
        self.assertNotEqual(pol.select_next_hop(pkt, router, 0.0), choice)

    def test_congestion_aware_penalises_a_saturated_link(self):
        sim = Simulator(make_cfg("congestion_aware"))
        pol, router = sim.policy, sim.routers["S"]
        pkt = Packet(src="S", dst="D", creation_time=0.0, ttl=500.0)
        before = pol.candidate_costs(pkt, router)
        n = next(iter(before))
        sim.links[("S", n)].utilization_ema = 0.99
        self.assertGreater(pol.candidate_costs(pkt, router)[n]["cost"], before[n]["cost"])


# ======================================================================== experiment tooling
class ExperimentToolingTests(unittest.TestCase):
    def test_overrides_reject_typos(self):
        cfg = make_cfg()
        apply_overrides(cfg, {"traffic.rate_pps": 12.5})
        self.assertEqual(cfg.traffic.rate_pps, 12.5)
        with self.assertRaises(ValueError):
            set_path(cfg, "traffic.rate_ppss", 1)
        with self.assertRaises(ValueError):
            set_path(cfg, "nonsense.field", 1)

    def test_statistics_helpers(self):
        d = describe([1.0, 2.0, 3.0, None, float("nan")])
        self.assertEqual((d["n"], d["mean"]), (3, 2.0))
        self.assertGreater(d["ci95"], 0)
        p = paired({1: 0.9, 2: 0.8}, {1: 0.7, 2: 0.7}, "higher")
        self.assertEqual((p["n"], p["better_in"]), (2, 2))

    def test_failed_runs_are_reported_not_hidden(self):
        import run_comparison as rc
        experiments = {"baseline": "shortest_path", "scenarios": {"tiny": {"overrides": {
            "network.enable_failures": False, "simulation.duration": 6.0, "simulation.warmup": 1.0}}}}
        with tempfile.TemporaryDirectory() as tmp:
            out = rc.run_experiments(make_cfg(), experiments, methods=["shortest_path", "congestion_aware", "bogus"],
                                     seeds=[1, 2], out_dir=Path(tmp), plots=False, topology_seed=314)
            self.assertEqual(len(out["failures"]), 2)
            self.assertEqual(sum(r["status"] == "failed" for r in out["rows"]), 2)
            self.assertEqual(sum(r["status"] == "ok" for r in out["rows"]), 4)
            for name in ("raw_runs.csv", "comparison_summaries.json", "failed_runs.json", "experiment_manifest.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            m = out["aggregate"]["tiny"]["methods"]
            self.assertEqual((m["bogus"]["runs_ok"], m["bogus"]["runs_failed"]), (0, 2))
            self.assertEqual(m["shortest_path"]["runs_ok"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)