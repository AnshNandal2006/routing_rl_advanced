"""Tests for the LIVE Q-LEARNING BRAIN (routing_rl/simulation/live.py, tabular_q hooks, NeuroRouter API).

Run from the project root:
    python -m unittest tests.test_live_brain -v        (or: python -m unittest discover -s tests -v)

None of these tests use fabricated performance numbers: they compare what the live layer reports against
the real agent / simulator state, or against independent recomputations.
"""
import copy
import json
import shutil
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from routing_rl.config import Config
from routing_rl.network.packet import Packet
from routing_rl.network.topology import TOPOLOGY_CATALOG, build_topology, traffic_defaults
from routing_rl.routing.tabular_q import TabularQPolicy
from routing_rl.simulation.live import (CONTROL_ACTIONS, ControlError, LiveSession, SPEEDS)
from routing_rl.simulation.simulator import Simulator


def cfg_for(topology="diamond", duration=30.0, seed=7, rate=30.0, ttype="poisson", method="tabular_q"):
    cfg = Config.from_yaml(ROOT / "config" / "default.yaml")
    cfg.routing.method = method
    cfg.network.topology = topology
    cfg.network.topology_seed = seed
    cfg.simulation.seed, cfg.simulation.duration = seed, duration
    cfg.traffic.type, cfg.traffic.rate_pps = ttype, rate
    cfg.network.failure_schedule = []
    routers, _, _ = build_topology(topology, queue_size=cfg.network.default_queue_size,
                                   capacity=cfg.network.default_capacity, prop_delay=cfg.network.propagation_delay,
                                   seed=seed, n_nodes_min=cfg.network.n_nodes_min, n_nodes_max=cfg.network.n_nodes_max)
    cfg.traffic.sources, cfg.traffic.destinations, _ = traffic_defaults(topology, routers)
    return cfg


def session(**kw):
    history = kw.pop("history", 1000)
    return LiveSession(cfg_for(**kw), history=history)


def run(s, seconds):
    s.sim.advance(int(round(seconds / s.sim.dt)))


class TestQTableMatchesAgent(unittest.TestCase):
    def test_focus_rows_and_full_table_equal_the_agents_table(self):
        s = session()
        run(s, 4)
        snap = s.snapshot()
        d = snap["focus"]["decision"]
        state = tuple(d["state"])
        for row in snap["focus"]["rows"]:
            in_table = row["hop"] in s.policy.q_table.get(state, {})
            self.assertEqual(row["visited"], in_table)
            if in_table:
                self.assertEqual(row["q"], s.policy.q_table[state][row["hop"]])      # exact, read from the agent
            else:
                self.assertIsNone(row["q"])                                           # never invented
                self.assertEqual(row["q_effective"], 0.0)                             # the agent's initial value
        page = s.qtable_page(limit=200)
        flat = {(r["router"], r["dst"], r["level"], r["hop"]): r["q"] for r in s.qtable_page(limit=200)["rows"]}
        expected = {(st[0], st[1], st[2], a): q for st, acts in s.policy.q_table.items() for a, q in acts.items()}
        if page["total"] <= 200:
            self.assertEqual(flat, expected)
        self.assertEqual(page["total"], len(expected))
        self.assertEqual(snap["kpis"]["q_entries"], s.policy.get_stats()["q_entries"])

    def test_displayed_value_follows_the_agent_after_learning(self):
        s = session()
        run(s, 3)
        snap = s.snapshot()
        d = snap["focus"]["decision"]
        state, hop = tuple(d["state"]), d["selected"]
        before = s.policy.q_table[state][hop]
        changed = False
        for _ in range(400):                                   # keep running until that very entry is updated again
            s.sim.advance(5)
            if s.policy.q_table[state][hop] != before:
                changed = True
                break
        self.assertTrue(changed, "expected the Q-value to change through learning")
        now = s.snapshot(focus=d["seq"])
        row = next(r for r in now["focus"]["rows"] if r["hop"] == hop)
        self.assertEqual(row["q"], s.policy.q_table[state][hop])
        self.assertNotEqual(row["q"], before)
        self.assertEqual(row["q_at_decision"], d["candidates"][[c["hop"] for c in d["candidates"]].index(hop)]["q"])

    def test_unvisited_pairs_are_labelled_not_filled_in(self):
        s = session()
        s.control("step")
        rows = s.snapshot()["focus"]["rows"]
        self.assertTrue(any(not r["visited"] for r in rows) or all(r["q"] is not None for r in rows))
        for r in rows:
            if not r["visited"]:
                self.assertIsNone(r["q"])


class TestLearningMath(unittest.TestCase):
    def test_update_function_on_hand_computed_case(self):
        sim = Simulator(cfg_for(), train_rl=False)
        pol = sim.policy
        events = []
        pol.tracer = events.append
        pol.alpha, pol.gamma = 0.5, 0.9
        s, ns = ("S0", "D0", 0), ("A", "D0", 0)
        pol.q_table[s] = {"A": 2.0}
        pol.q_table[ns] = {"x": 3.0, "y": 1.0}
        pol._update_q(s, "A", -1.0, ns, ["x", "y"], now=1.0, trigger="hop_forward")
        # 2 + 0.5 * (-1 + 0.9 * 3 - 2) = 1.85
        self.assertAlmostEqual(pol.q_table[s]["A"], 1.85, places=12)
        u = events[-1]
        self.assertEqual((u["q_old"], u["alpha"], u["gamma"], u["reward"], u["max_next"]), (2.0, 0.5, 0.9, -1.0, 3.0))
        self.assertAlmostEqual(u["target"], 1.7, places=12)
        self.assertAlmostEqual(u["q_new"], pol.q_table[s]["A"], places=12)

    def test_terminal_update_uses_zero_next_value(self):
        sim = Simulator(cfg_for(), train_rl=False)
        pol = sim.policy
        ev = []
        pol.tracer = ev.append
        pol._update_q(("A", "D0", 0), "D0", 40.0, None, [])
        self.assertEqual(ev[-1]["max_next"], 0.0)
        self.assertAlmostEqual(ev[-1]["q_new"], 0.0 + pol.alpha * 40.0, places=12)

    def test_every_recorded_update_reproduces_the_agents_value(self):
        s = session()
        run(s, 8)
        self.assertGreater(len(s.updates_log), 200)
        for u in s.updates_log:
            recomputed = u["q_old"] + u["alpha"] * (u["reward"] + u["gamma"] * u["max_next"] - u["q_old"])
            self.assertAlmostEqual(recomputed, u["q_new"], places=9)
            self.assertTrue(u["verified"])
            self.assertAlmostEqual(u["target"], u["reward"] + u["gamma"] * u["max_next"], places=9)
        self.assertTrue(any(u["changed"] for u in s.updates_log))
        snap_math = s.snapshot()["math"]
        self.assertTrue(snap_math["verified"])
        self.assertIn("formula", snap_math)

    def test_reward_decomposition_matches_agent_reward(self):
        s = session()
        run(s, 5)
        for u in s.updates_log:
            if u["trigger"] == "hop_forward":
                rp = u["reward_parts"]
                self.assertAlmostEqual(u["reward"], (rp["hop"] + rp["queue"]) * rp["priority_multiplier"], places=9)

    def test_params_change_applies_to_later_updates(self):
        s = session()
        run(s, 2)
        s.control("params", alpha=0.4, gamma=0.5)
        mark = s.update_seq
        run(s, 1)
        later = [u for u in s.updates_log if u["seq"] > mark]
        self.assertTrue(later)
        self.assertTrue(all(u["alpha"] == 0.4 and u["gamma"] == 0.5 for u in later))


class TestSelectionRule(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(cfg_for(), train_rl=False)
        self.pol = self.sim.policy
        self.pol.tracer = lambda e: None
        self.router = self.sim.routers["S0"]
        self.nbrs = list(self.router.neighbors)

    def _pkt(self, path=()):
        p = Packet(src="S0", dst="D0", creation_time=0.0, ttl=9.0)
        p.path = list(path)
        return p

    def test_greedy_picks_the_highest_q(self):
        self.pol.epsilon = 0.0
        state = ("S0", "D0", self.router.queue_level())
        self.pol.q_table[state] = {self.nbrs[0]: 1.0, self.nbrs[1]: 5.0}
        for _ in range(20):
            p = self._pkt()
            self.assertEqual(self.pol.select_next_hop(p, self.router, 0.0), self.nbrs[1])
            self.assertEqual(self.pol._last_sel[p.packet_id]["mode"], "greedy")

    def test_ties_are_labelled_and_stay_inside_the_tied_set(self):
        self.pol.epsilon = 0.0
        p = self._pkt()
        hop = self.pol.select_next_hop(p, self.router, 0.0)
        sel = self.pol._last_sel[p.packet_id]
        self.assertEqual(sel["mode"], "greedy_tie")
        self.assertIn(hop, sel["greedy_set"])

    def test_exploration_is_labelled_and_counted(self):
        self.pol.epsilon = 1.0
        before = self.pol.explorations
        modes = set()
        for _ in range(30):
            p = self._pkt()
            self.pol.select_next_hop(p, self.router, 0.0)
            modes.add(self.pol._last_sel[p.packet_id]["mode"])
        self.assertEqual(modes, {"explore"})
        self.assertEqual(self.pol.explorations, before + 30)

    def test_exploration_switch_off_means_greedy_even_with_epsilon_one(self):
        self.pol.epsilon = 1.0
        self.pol.explore_enabled = False
        state = ("S0", "D0", self.router.queue_level())
        self.pol.q_table[state] = {self.nbrs[0]: 9.0, self.nbrs[1]: 1.0}
        for _ in range(30):
            p = self._pkt()
            self.assertEqual(self.pol.select_next_hop(p, self.router, 0.0), self.nbrs[0])
            self.assertNotEqual(self.pol._last_sel[p.packet_id]["mode"], "explore")

    def test_loop_avoidance_and_down_links_are_excluded_and_reported(self):
        self.pol.epsilon = 1.0
        a, b = self.nbrs[0], self.nbrs[1]
        for _ in range(20):                                     # a is on the path -> never chosen, reported
            p = self._pkt(path=[a])
            self.assertEqual(self.pol.select_next_hop(p, self.router, 0.0), b)
            self.assertEqual(self.pol._last_sel[p.packet_id]["loop_excluded"], [a])
        self.sim.links[("S0", a)].set_status(False)             # a is down -> never chosen, reported as down
        for _ in range(20):
            p = self._pkt()
            self.assertEqual(self.pol.select_next_hop(p, self.router, 0.0), b)
            self.assertEqual(self.pol._last_sel[p.packet_id]["down"], [a])
        self.sim.links[("S0", b)].set_status(False)
        self.assertIsNone(self.pol.select_next_hop(self._pkt(), self.router, 0.0))   # nothing valid -> None


class TestLiveDecisions(unittest.TestCase):
    def test_recorded_decisions_obey_the_real_rule_and_are_labelled_correctly(self):
        s = session(duration=20)
        run(s, 10)
        ds = list(s.decisions)
        self.assertGreater(len(ds), 300)
        modes = {d["mode"] for d in ds}
        self.assertIn("explore", modes)
        self.assertTrue(modes & {"greedy", "greedy_tie"})
        for d in ds:
            self.assertTrue(d["selection_matches_forward"])
            hops = [c["hop"] for c in d["candidates"]]
            self.assertIn(d["selected"], hops)
            top = max(c["q"] for c in d["candidates"])
            text = " ".join(e["text"] for e in d["explanation"])
            if d["mode"] == "explore":
                self.assertIn("EXPLORATION", text)
                self.assertIn(f"{d['epsilon']:.3f}", text)
                self.assertNotIn("has the highest Q-value", text)      # never claim greedy for an exploration choice
            else:
                self.assertEqual(next(c["q"] for c in d["candidates"] if c["hop"] == d["selected"]), top)
                self.assertNotIn("EXPLORATION (", text)
                self.assertTrue(d["selected"] in d["greedy_set"])

    def test_every_forward_is_exactly_one_decision(self):
        s = session()
        run(s, 8)
        self.assertEqual(s.decision_seq, sum(l.packets_sent for l in s.sim.links.values()))

    def test_exploration_toggle_stops_exploration(self):
        s = session()
        run(s, 3)
        s.control("explore", enabled=False)
        mark = s.decision_seq
        run(s, 3)
        new = [d for d in s.decisions if d["seq"] > mark]
        self.assertTrue(new)
        self.assertTrue(all(d["mode"] != "explore" for d in new))
        s.control("explore", enabled=True)
        mark = s.decision_seq
        run(s, 6)
        self.assertTrue(any(d["mode"] == "explore" for d in s.decisions if d["seq"] > mark))

    def test_step_reveals_exactly_one_decision_at_a_time(self):
        s = session()
        seen = []
        for _ in range(60):
            s.control("step")
            seen.append(s.snapshot()["focus"]["decision"]["seq"])
        self.assertEqual(seen, list(range(1, 61)))
        self.assertTrue(all(d["seq"] <= 60 for d in s.snapshot()["decisions"]))   # later ones stay hidden
        s.control("follow")
        self.assertGreaterEqual(s.snapshot()["focus"]["decision"]["seq"], 60)


class TestTimelineAuthenticity(unittest.TestCase):
    def test_empty_session_does_not_fabricate_anything(self):
        s = session()
        snap = s.snapshot()
        self.assertIsNone(snap["focus"])
        self.assertIsNone(snap["math"])
        self.assertEqual((snap["decisions"], snap["updates"], snap["routes"]), ([], [], []))
        self.assertEqual(snap["kpis"]["route_changes"], 0)
        self.assertEqual(snap["kpis"]["updates"], 0)
        self.assertIsNone(snap["kpis"]["best_valid_q"])
        self.assertEqual(s.qtable_page()["total"], 0)
        json.dumps(snap, allow_nan=False)

    def test_route_changes_equal_an_independent_recount(self):
        s = session(duration=20)
        run(s, 10)
        last, changes = {}, 0
        for d in s.decisions:                                   # chronological
            key = (d["router"], d["dst"])
            self.assertEqual(d["previous_hop"], last.get(key))
            if key in last and last[key] != d["selected"]:
                changes += 1
                self.assertTrue(d["changed"])
            else:
                self.assertFalse(d["changed"])
            last[key] = d["selected"]
        self.assertEqual(changes, s.route_changes_traced)
        self.assertEqual(changes, s.snapshot()["kpis"]["route_changes"])

    def test_decisions_are_real_topology_hops_in_time_order(self):
        s = session()
        run(s, 5)
        prev_t = -1
        for d in s.decisions:
            self.assertGreaterEqual(d["t"], prev_t)
            prev_t = d["t"]
            self.assertIn(d["selected"], s.sim.routers[d["router"]].neighbors)
            self.assertTrue(set(d["path"]) <= set(s.sim.routers))

    def test_history_is_bounded(self):
        s = session(history=10)
        run(s, 5)
        self.assertEqual(len(s.decisions), 10)
        self.assertEqual(len(s.updates_log), 10)
        self.assertGreater(s.decision_seq, 10)
        self.assertEqual(len(s.snapshot(limit=200)["decisions"]), 10)

    def test_route_history_is_made_of_real_delivered_paths(self):
        s = session()
        run(s, 8)
        snap = s.snapshot()
        self.assertTrue(snap["routes"])
        r = snap["routes"][0]
        self.assertEqual(r["current"]["path"][0], s.cfg.traffic.sources[0])
        self.assertEqual(r["current"]["path"][-1], s.cfg.traffic.destinations[0])
        for a, b in zip(r["current"]["path"], r["current"]["path"][1:]):
            self.assertIn(b, s.sim.routers[a].neighbors)


class TestLinkFailure(unittest.TestCase):
    def test_failed_link_is_never_selected_and_is_reported_down(self):
        s = session(duration=40)
        run(s, 4)
        s.control("fail_link", a="S0", b="B", duration=8)
        mark = s.decision_seq
        run(s, 6)                                              # entirely inside the outage
        self.assertEqual(s.snapshot()["down_links"], [["S0", "B"]])
        during = [d for d in s.decisions if d["seq"] > mark]
        self.assertTrue(during)
        for d in during:
            self.assertNotIn((d["router"], d["selected"]), {("S0", "B"), ("B", "S0")})
        at_source = [d for d in during if d["router"] == "S0"]
        self.assertTrue(at_source)
        self.assertTrue(all("B" in d["down"] for d in at_source))
        snap = s.snapshot(focus=at_source[-1]["seq"])
        row = next(r for r in snap["focus"]["rows"] if r["hop"] == "B")
        self.assertEqual(row["status"], "down")
        self.assertFalse(row["link_up"])
        self.assertTrue(any(c["kind"] == "fail" for c in snap["controls"]))
        self.assertTrue(any("FAILED" in e["text"] for e in snap["events"]))      # real event-system log line

    def test_link_returns_when_the_outage_ends_and_event_is_logged(self):
        s = session(duration=40)
        s.control("fail_link", a="S0", b="B", duration=3)
        run(s, 5)
        self.assertEqual(s.snapshot()["down_links"], [])
        log = " ".join(s.sim.event_log)
        self.assertIn("FAILED", log)
        self.assertIn("RECOVERED", log)

    def test_restore_ends_an_outage_early_and_rejects_when_none(self):
        s = session(duration=40)
        with self.assertRaises(ControlError):
            s.control("restore_link", a="S0", b="B")
        s.control("fail_link", a="S0", b="B", duration=30)
        run(s, 1)
        self.assertEqual(len(s.snapshot()["down_links"]), 1)
        s.control("restore_link", a="S0", b="B")
        run(s, 1)
        self.assertEqual(s.snapshot()["down_links"], [])
        self.assertIn("RECOVERED", " ".join(s.sim.event_log))

    def test_route_change_explanation_names_the_downed_link(self):
        s = session(duration=60)
        run(s, 6)
        used = [d["selected"] for d in s.decisions if d["router"] == "S0"][-1]     # the hop the agent used last at S0
        s.control("fail_link", a="S0", b=used, duration=20)
        mark = s.decision_seq
        run(s, 6)
        after = [d for d in s.decisions if d["seq"] > mark and d["router"] == "S0"]
        first = after[0]
        self.assertEqual(first["previous_hop"], used)
        self.assertNotEqual(first["selected"], used)
        self.assertTrue(first["changed"])
        self.assertIn(used, first["down"])
        texts = [e["text"] for e in first["explanation"]]
        self.assertTrue(any(f"Link S0→{used} is down" in t for t in texts), texts)
        self.assertTrue(any(e["kind"] == "observed" and "is down" in e["text"] for e in first["explanation"]))
        self.assertTrue(all(used not in d["selected"] for d in after))


class TestResetBehaviour(unittest.TestCase):
    def test_reset_keep_preserves_the_q_table(self):
        s = session()
        run(s, 6)
        table, eps, upd = copy.deepcopy(s.policy.q_table), s.policy.epsilon, s.policy.updates
        s.control("reset", keep_learning=True)
        self.assertEqual(s.policy.q_table, table)
        self.assertEqual((s.policy.epsilon, s.policy.updates), (eps, upd))
        snap = s.snapshot()
        self.assertEqual(snap["reset_info"]["q_table"], "preserved")
        self.assertEqual(snap["reset_info"]["q_entries"], sum(len(v) for v in table.values()))
        self.assertEqual(snap["episode"], 1)
        self.assertEqual(s.sim.t, 0.0)
        self.assertEqual((snap["decisions"], snap["updates"], snap["routes"]), ([], [], []))
        self.assertEqual(s.route_changes_traced, 0)
        self.assertIsNot(s.policy.q_table, table)

    def test_reset_wipe_empties_the_q_table(self):
        s = session()
        run(s, 6)
        self.assertGreater(len(s.policy.q_table), 0)
        s.control("reset", keep_learning=False)
        self.assertEqual(s.policy.q_table, {})
        self.assertEqual(s.policy.updates, 0)
        self.assertEqual(s.policy.epsilon, s.policy.initial_epsilon)
        self.assertEqual(s.snapshot()["reset_info"]["q_table"], "reset")

    def test_fresh_session_reports_initial(self):
        self.assertEqual(session().snapshot()["reset_info"]["q_table"], "initial")

    def test_reset_pauses_and_clears_controls_effects(self):
        s = session()
        s.control("traffic", multiplier=3)
        s.control("reset", keep_learning=True)
        self.assertEqual(s.sim.traffic.rate_multiplier, 1.0)
        self.assertEqual(s.status(), "paused")


class TestReproducibility(unittest.TestCase):
    def test_live_session_matches_a_plain_run_with_the_same_seed(self):
        cfg = cfg_for(duration=15)
        sim = Simulator(copy.deepcopy(cfg), train_rl=False)
        summary = sim.run(progress=False).summary()
        s = LiveSession(cfg, history=50)
        s.sim.advance(10 ** 9)
        self.assertEqual(json.dumps(s.sim.metrics.summary(), sort_keys=True, default=float),
                         json.dumps(summary, sort_keys=True, default=float))
        self.assertEqual(s.policy.get_stats(), sim.policy.get_stats())
        self.assertEqual(s.policy.q_table, sim.policy.q_table)

    def test_other_simulations_in_the_process_do_not_disturb_a_live_session(self):
        baseline = session(duration=20)
        run(baseline, 12)
        s = session(duration=20)
        run(s, 5)
        other = Simulator(cfg_for(seed=99, duration=6), train_rl=False)    # reseeds the global RNGs and uses them
        other.run(progress=False)
        run(s, 7)
        self.assertEqual(s.policy.q_table, baseline.policy.q_table)

    def test_tracer_does_not_change_agent_behaviour(self):
        cfg = cfg_for(duration=10)
        a = Simulator(copy.deepcopy(cfg), train_rl=False)
        a.run(progress=False)
        b = LiveSession(copy.deepcopy(cfg), history=50)
        b.sim.advance(10 ** 9)
        self.assertEqual(a.policy.q_table, b.policy.q_table)


class TestControlsValidation(unittest.TestCase):
    def setUp(self):
        self.s = session()

    def test_rejects_bad_values(self):
        bad = [("traffic", dict(multiplier=0)), ("traffic", dict(multiplier=99)), ("traffic", dict(multiplier="x")),
               ("traffic", dict(multiplier=float("nan"))), ("params", dict(alpha=0, gamma=0.5)), ("params", dict(alpha=1.5, gamma=0.5)),
               ("params", dict(alpha=0.5, gamma=1.0)), ("params", dict(alpha=0.5, gamma=-0.1)), ("speed", dict(value=7)),
               ("speed", dict(value=None)), ("reset", dict(keep_learning="yes")), ("reset", dict()), ("explore", dict(enabled=1)),
               ("fail_link", dict(a="S0", b="B", duration=0)), ("fail_link", dict(a="S0", b="B", duration=1e9)),
               ("fail_link", dict(a="S0", b="nope", duration=5)), ("fail_link", dict(a=1, b=2, duration=5)),
               ("exec", dict()), ("__init__", dict())]
        for action, kw in bad:
            with self.assertRaises(ControlError, msg=f"{action} {kw}"):
                self.s.control(action, **kw)

    def test_accepts_every_allowed_speed(self):
        for sp in SPEEDS:
            self.s.control("speed", value=sp)
            self.assertEqual(self.s.speed, sp)

    def test_traffic_multiplier_really_scales_the_offered_load(self):
        a, b = session(ttype="constant"), session(ttype="constant")
        b.control("traffic", multiplier=2)
        run(a, 4)
        run(b, 4)
        self.assertAlmostEqual(b.sim.traffic.emitted / a.sim.traffic.emitted, 2.0, delta=0.05)

    def test_control_actions_list_is_complete(self):
        self.assertEqual(set(CONTROL_ACTIONS), {"play", "pause", "follow", "speed", "step", "reset", "traffic",
                                                "fail_link", "restore_link", "explore", "params"})

    def test_background_runner_advances_and_pauses(self):
        import time
        s = session(duration=60)
        s.control("speed", value=25.0)
        s.control("play")
        time.sleep(1.0)
        t1 = s.sim.t
        self.assertGreater(t1, 3.0)
        s.control("pause")
        time.sleep(0.3)
        t2 = s.sim.t
        time.sleep(0.4)
        self.assertEqual(s.sim.t, t2)
        self.assertEqual(s.status(), "paused")
        s.close()

    def test_finished_episode_refuses_play_until_reset(self):
        s = session(duration=2)
        s.sim.advance(10 ** 6)
        self.assertEqual(s.status(), "finished")
        with self.assertRaises(ControlError):
            s.control("play")
        s.control("reset", keep_learning=True)
        s.control("play")
        s.close()


class TestAllPoliciesAndTopologies(unittest.TestCase):
    def test_every_non_dqn_policy_runs_on_every_topology(self):
        for topo in TOPOLOGY_CATALOG:
            for method in ("shortest_path", "congestion_aware", "tabular_q"):
                with self.subTest(topology=topo, method=method):
                    sim = Simulator(cfg_for(topo, duration=5, method=method), train_rl=False)
                    summary = sim.run(progress=False).summary()
                    self.assertGreaterEqual(summary["delivery_ratio"], 0.0)
                    self.assertTrue(sim.accounting_report["balanced"])

    def test_live_session_works_on_every_topology(self):
        for topo in TOPOLOGY_CATALOG:
            with self.subTest(topology=topo):
                s = session(topology=topo, duration=10)
                run(s, 3)
                snap = s.snapshot()
                json.dumps(snap, allow_nan=False)
                self.assertGreater(snap["kpis"]["decisions"], 0)
                self.assertTrue(snap["graph"]["edges"])
                s.close()

    def test_non_tabular_methods_are_refused_by_the_live_brain(self):
        with self.assertRaises(ControlError):
            LiveSession(cfg_for(method="shortest_path"))

    def test_dqn_still_constructs_when_torch_is_installed(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch not installed")
        sim = Simulator(cfg_for(method="dqn", duration=3), train_rl=True)
        sim.run(progress=False)


class TestNoTorchNeeded(unittest.TestCase):
    def test_live_brain_runs_with_torch_blocked(self):
        code = (
            "import sys; sys.modules['torch']=None\n"
            f"sys.path.insert(0, r'{ROOT}')\n"
            "from routing_rl.config import Config\n"
            "from routing_rl.simulation.live import LiveSession\n"
            f"cfg=Config.from_yaml(r'{ROOT / 'config' / 'default.yaml'}'); cfg.routing.method='tabular_q'; cfg.network.topology='diamond'\n"
            "cfg.traffic.sources, cfg.traffic.destinations=['S0'],['D0']; cfg.simulation.duration=5; cfg.network.failure_schedule=[]\n"
            "s=LiveSession(cfg); s.sim.advance(300); snap=s.snapshot()\n"
            "assert snap['kpis']['decisions']>0 and 'torch' not in [m for m in sys.modules if sys.modules[m] is not None and m=='torch']\n"
            "print('OK', snap['kpis']['decisions'])\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertIn("OK", out.stdout, out.stderr)


class TestHttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import neurorouter_app as app
        cls.app = app
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.sc = dict(topology="diamond", rate=28, traffic_type="poisson", priority=0.12, duration=40, seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.app._close_live()
        cls.srv.shutdown()

    def setUp(self):
        self.app._close_live()

    def call(self, path, body=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST")
        try:
            r = urllib.request.urlopen(req)
            code, text = r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            code, text = e.code, e.read().decode()
        def no_nan(x):
            raise ValueError(f"non-JSON constant {x}")
        return code, json.loads(text, parse_constant=no_nan)

    def test_state_and_controls_before_start(self):
        self.assertEqual(self.call("/api/live/state"), (200, {"active": False}))
        self.assertEqual(self.call("/api/live/control", {"action": "play"})[0], 400)
        self.assertEqual(self.call("/api/live/qtable")[0], 400)

    def test_start_step_state_qtable_roundtrip(self):
        code, snap = self.call("/api/live/start", self.sc)
        self.assertEqual(code, 200)
        self.assertEqual(snap["status"], "paused")
        for _ in range(25):
            self.assertEqual(self.call("/api/live/control", {"action": "step"})[0], 200)
        code, st = self.call("/api/live/state?limit=10")
        self.assertEqual(code, 200)
        self.assertEqual(st["kpis"]["decisions"], 25)
        self.assertEqual(st["focus"]["decision"]["seq"], 25)
        self.assertLessEqual(len(st["decisions"]), 10)
        code, qt = self.call("/api/live/qtable?limit=5")
        self.assertEqual((code, len(qt["rows"]) <= 5), (200, True))
        live = self.app.LIVE["session"]
        for r in qt["rows"]:
            self.assertEqual(r["q"], live.policy.q_table[(r["router"], r["dst"], r["level"])][r["hop"]])

    def test_live_start_forces_tabular_q_and_no_scheduled_failure(self):
        code, snap = self.call("/api/live/start", dict(self.sc, method="dqn", enable_fail=True, fail_start=5, fail_dur=5))
        self.assertEqual(code, 200)
        live = self.app.LIVE["session"]
        self.assertIsInstance(live.policy, TabularQPolicy)
        self.assertEqual(live.cfg.network.failure_schedule, [])

    def test_invalid_input_is_rejected_with_400(self):
        self.call("/api/live/start", self.sc)
        bad = [{"action": "traffic", "multiplier": 99}, {"action": "params", "alpha": 0, "gamma": 0.5},
               {"action": "exec", "code": "__import__('os')"}, {"action": "play", "x": 1},
               {"action": "fail_link", "a": "nope", "b": "S0", "duration": 5}, {"action": "speed", "value": 7},
               {"action": "reset", "keep_learning": "yes"}, {"action": 5}, {}]
        for b in bad:
            code, body = self.call("/api/live/control", b)
            self.assertEqual(code, 400, b)
            self.assertIn("error", body)
        self.assertEqual(self.call("/api/live/state?focus=abc")[0], 400)
        self.assertEqual(self.call("/api/live/state?limit=99999")[0], 400)
        self.assertEqual(self.call("/api/live/qtable?router=../../x")[0], 400)
        self.assertEqual(self.call("/api/live/start", dict(self.sc, topology="x"))[0], 400)
        self.assertEqual(self.call("/api/live/start", dict(self.sc, rate=1e9))[0], 400)
        self.assertEqual(self.call("/api/live/start", dict(self.sc, history=5))[0], 400)
        self.assertEqual(self.call("/api/live/control", raw=b"{not json")[0], 400)
        self.assertEqual(self.call("/api/live/control", raw=b"x" * 20000)[0], 413)
        self.assertEqual(self.call("/api/live/other", {"a": 1})[0], 404)

    def test_failed_link_via_http_is_visible_in_state(self):
        self.call("/api/live/start", self.sc)
        self.call("/api/live/control", {"action": "step"})
        self.assertEqual(self.call("/api/live/control", {"action": "fail_link", "a": "S0", "b": "B", "duration": 10})[0], 200)
        for _ in range(3):
            self.call("/api/live/control", {"action": "step"})
        # the outage is applied by the event system on the next simulator tick; step until it shows
        for _ in range(40):
            if self.call("/api/live/state")[1]["down_links"]:
                break
            self.call("/api/live/control", {"action": "step"})
        self.assertEqual(self.call("/api/live/state")[1]["down_links"], [["S0", "B"]])

    def test_page_contains_panel_and_existing_features(self):
        req = urllib.request.urlopen(self.base + "/")
        page = req.read().decode()
        for marker in ("LIVE Q-LEARNING BRAIN", "WHY DID THE AI CHOOSE THIS PATH?", "LEARNING UPDATE", "ROUTE DECISION TIMELINE",
                       "Battle mode", "Launch simulation", "Router load over time"):
            self.assertIn(marker, page)
        self.assertNotIn("__LIMITS__", page)
        self.assertEqual(self.call("/api/battle", None, raw=b"{}")[0], 400)   # existing endpoint still validates


class TestBrowserUi(unittest.TestCase):
    """Runs the real page in jsdom against a real server (skipped unless node and jsdom are available)."""

    def test_ui_script(self):
        script = ROOT / "tests" / "ui_live_check.js"
        node = shutil.which("node")
        if not node or not script.exists():
            self.skipTest("node or tests/ui_live_check.js not available")
        probe = subprocess.run([node, "-e", "require('jsdom')"], cwd=script.parent, capture_output=True, text=True)
        if probe.returncode != 0:
            self.skipTest("jsdom not installed (cd tests && npm install jsdom)")
        import neurorouter_app as app
        srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = subprocess.run([node, str(script), f"http://127.0.0.1:{srv.server_port}"], cwd=script.parent,
                                 capture_output=True, text=True, timeout=240)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        finally:
            app._close_live()
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
