"""Correctness tests for the simulator core. Run from the project root:
    python -m unittest discover -s tests -v
"""
import subprocess, sys, unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_rl.config import Config
from routing_rl.network.link import Link
from routing_rl.network.packet import Packet
from routing_rl.simulation.events import EventScheduler, LinkFailureEvent
from routing_rl.simulation.simulator import Simulator
from routing_rl.simulation.traffic import TrafficGenerator
from routing_rl.metrics.collector import MetricsCollector


def make_cfg(method="shortest_path", duration=20.0, fail=True, ttl=None):
    cfg = Config.from_yaml(ROOT / "config" / "default.yaml")
    cfg.routing.method = method
    cfg.network.topology = "diamond"
    cfg.traffic.sources, cfg.traffic.destinations = ["S0"], ["D0"]
    cfg.traffic.type, cfg.traffic.rate_pps = "poisson", 30.0
    cfg.simulation.duration, cfg.simulation.warmup, cfg.simulation.seed = duration, 2.0, 7
    cfg.network.enable_failures = fail
    cfg.network.failure_schedule = [{"t_start": 8.0, "t_end": 14.0, "link": ["S0", "B"]}] if fail else []
    if ttl is not None:
        cfg.network.default_ttl = ttl
    return cfg


class TestImports(unittest.TestCase):
    def test_policies_import_without_torch(self):
        code = ("import sys; sys.modules['torch']=None; import routing_rl.routing as r; "
                "r.ShortestPathPolicy; r.CongestionAwarePolicy; r.TabularQPolicy\n"
                "try:\n    r.DQNPolicy\nexcept ImportError as e:\n    assert 'PyTorch' in str(e); print('ok')")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("ok", out.stdout, out.stderr)


class TestLinkAndEvents(unittest.TestCase):
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


class TestTraffic(unittest.TestCase):
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

    def test_same_seed_same_trace(self):
        flows = [{"src": "A", "dst": "B", "rate": 40}]
        runs = []
        for _ in range(2):
            gen = TrafficGenerator(self._cfg(flows), np.random.default_rng(5))
            runs.append([len(gen.generate(i * 0.01, 0.01, 5.0)) for i in range(300)])
        self.assertEqual(runs[0], runs[1])


class TestMetrics(unittest.TestCase):
    def test_adaptation_time_measured_from_event(self):
        m = MetricsCollector(warmup=0.0)
        for t in np.arange(0.5, 60.01, 0.5):
            bad = 20 <= t < 30
            m.time_series["t"].append(float(t))
            m.time_series["delivery_ratio_inst"].append(0.2 if bad else 1.0)
            m.time_series["avg_delay"].append(0.05)
        m.mark_event(20.0, "disruption", "A-B")
        m.mark_event(30.0, "recovery", "A-B")
        rep = m.adaptation_report()
        self.assertAlmostEqual(rep[0]["adaptation_time"], 10.0, delta=0.6)
        self.assertAlmostEqual(rep[1]["adaptation_time"], 0.0, delta=0.6)
        self.assertGreater(rep[0]["delivery_drop"], 0.7)


class TestSimulation(unittest.TestCase):
    def _run(self, method, **kw):
        sim = Simulator(make_cfg(method, **kw), strict_accounting=True)
        return sim, sim.run(progress=False).summary()

    def test_accounting_is_exact_and_bounded(self):
        for method in ("shortest_path", "congestion_aware"):
            sim, s = self._run(method)
            self.assertTrue(sim.accounting_report["balanced"], sim.accounting_report)
            self.assertEqual(s["duplicate_terminal"], 0)
            self.assertLessEqual(s["loss_ratio"], 1.0)
            self.assertAlmostEqual(s["delivery_ratio"] + s["loss_ratio"] + s["unresolved_ratio"], 1.0, places=9)
            self.assertGreater(s["delivered"], 0)
            self.assertEqual(len(s["adaptation"]), 2)  # one disruption, one recovery

    def test_same_seed_is_reproducible(self):
        a = self._run("congestion_aware")[1]
        b = self._run("congestion_aware")[1]
        self.assertEqual(a["delivered"], b["delivered"])
        self.assertEqual(a["route_changes"], b["route_changes"])

    def test_ttl_is_enforced_for_packets_in_flight(self):
        sim, s = self._run("shortest_path", fail=False, ttl=0.001)  # shorter than one propagation delay
        self.assertEqual(s["delivered"], 0)
        self.assertGreater(s["expired"], 0)


class TestPolicies(unittest.TestCase):
    def test_shortest_path_reports_unreachable_destination(self):
        sim = Simulator(make_cfg("shortest_path", fail=False))
        for l in sim.links.values():
            if "D0" in (l.src, l.dst):
                l.set_status(False)
        sim.policy.on_link_change()
        pkt = Packet(src="S0", dst="D0", creation_time=0.0, ttl=50.0)
        self.assertFalse(sim.policy.is_reachable("S0", "D0"))
        self.assertIsNone(sim.policy.select_next_hop(pkt, sim.routers["S0"], 0.0))

    def test_congestion_aware_moves_away_from_a_congested_path(self):
        sim = Simulator(make_cfg("congestion_aware", fail=False))
        pol, router = sim.policy, sim.routers["S0"]
        pol.alpha = 100.0
        pkt = Packet(src="S0", dst="D0", creation_time=0.0, ttl=500.0)
        before = pol.candidate_costs(pkt, router)
        choice = min(before, key=lambda n: before[n]["cost"])
        if len(before) < 2 or choice == "D0":
            self.skipTest("topology offers no alternative next hop from S0")
        nb = sim.routers[choice]
        for _ in range(nb.max_queue_size):
            nb.enqueue(Packet(src=choice, dst="D0", creation_time=0.0, ttl=500.0))
        self.assertNotEqual(pol.select_next_hop(pkt, router, 0.0), choice)

    def test_congestion_aware_penalises_a_saturated_link(self):
        sim = Simulator(make_cfg("congestion_aware", fail=False))
        pol, router = sim.policy, sim.routers["S0"]
        pkt = Packet(src="S0", dst="D0", creation_time=0.0, ttl=500.0)
        before = pol.candidate_costs(pkt, router)
        n = next(iter(before))
        sim.links[("S0", n)].utilization_ema = 0.99
        self.assertGreater(pol.candidate_costs(pkt, router)[n]["cost"], before[n]["cost"])


if __name__ == "__main__":
    unittest.main()
