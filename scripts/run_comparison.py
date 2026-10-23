#!/usr/bin/env python3
"""Run every routing method on identical scenarios and seeds, save raw and aggregate results, plot.

For each (scenario, seed) the SAME topology, traffic seed, parameters and failure events are given
to every method; only routing.method changes. Nothing is dropped silently: a run that raises is
recorded in failed_runs.json, shown in the console summary, and makes the script exit with code 2
(unless --allow-failures).

Learning policies (tabular_q, dqn) learn ONLINE during the measured run, so their numbers include
the cost of exploration. Warm-up traffic is excluded from the summary.

Outputs (default results/):
  raw_runs.csv               one row per (scenario, method, seed) with every scalar metric
  runs/<scenario>/*.json     per-run summary, time series, policy stats, accounting, config
  comparison_summaries.json  mean / std / 95% CI per scenario, method and metric + paired differences
  failed_runs.json           runs that raised, with tracebacks
  experiment_manifest.json   how the results were produced
  plots/*.png                bar charts, time series, load sweep

Examples
  python scripts/run_comparison.py --scenarios failure_recovery --seeds 11 12 13
  python scripts/run_comparison.py --methods shortest_path congestion_aware --set simulation.duration=60
  python scripts/run_comparison.py --resume          # continue an interrupted experiment
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from routing_rl.config import Config
from routing_rl.experiment_utils import (COLORS, LABELS, METHODS, METRICS, apply_overrides, clone_cfg,
                                         collect_metadata, describe, json_default, paired, parse_override,
                                         resolve_cfg, run_one)

COUNT_KEYS = ["generated", "delivered", "dropped", "expired", "loops", "unresolved", "route_changes", "duplicate_terminal"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


def expand_scenarios(experiments: Dict[str, Any], names: Optional[List[str]]) -> List[Dict[str, Any]]:
    spec_all = experiments.get("scenarios", {})
    unknown = [n for n in (names or []) if n not in spec_all]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown}; available: {list(spec_all)}")
    out = []
    for name, spec in spec_all.items():
        if names and name not in names:
            continue
        base = dict(spec.get("overrides") or {})
        sweep = spec.get("sweep")
        if sweep:
            for v in sweep["values"]:
                out.append({"label": f"{name}[{sweep['parameter']}={v}]", "base": name, "description": spec.get("description", ""),
                            "overrides": {**base, sweep["parameter"]: v},
                            "sweep_parameter": sweep["parameter"], "sweep_value": v})
        else:
            out.append({"label": name, "base": name, "description": spec.get("description", ""), "overrides": base,
                        "sweep_parameter": None, "sweep_value": None})
    return out


def make_row(scn, method, seed, result=None, error="") -> Dict[str, Any]:
    row = {"scenario": scn["label"], "base_scenario": scn["base"], "sweep_parameter": scn["sweep_parameter"],
           "sweep_value": scn["sweep_value"], "method": method, "seed": seed,
           "status": "ok" if result else "failed", "error": error}
    if result:
        s = result["summary"]
        for key, _, _ in METRICS:
            row[key] = s.get(key)
        for k in COUNT_KEYS:
            row[k] = s.get(k)
        for reason, cnt in (s.get("drop_reasons") or {}).items():
            row[f"drop_{reason}"] = cnt
        row["accounting_balanced"] = result["accounting"].get("balanced")
        row["runtime_s"] = result["runtime_s"]
    return row


# ---------------------------------------------------------------- aggregation
def aggregate(rows: List[Dict[str, Any]], methods: List[str], baseline: str) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for label in dict.fromkeys(r["scenario"] for r in rows):
        lr = [r for r in rows if r["scenario"] == label]
        entry = {"base_scenario": lr[0]["base_scenario"], "sweep_parameter": lr[0]["sweep_parameter"],
                 "sweep_value": lr[0]["sweep_value"], "methods": {}, f"paired_vs_{baseline}": {}}
        for m in methods:
            mr = [r for r in lr if r["method"] == m]
            ok = [r for r in mr if r["status"] == "ok"]
            entry["methods"][m] = {"runs_total": len(mr), "runs_ok": len(ok), "runs_failed": len(mr) - len(ok),
                                   "metrics": {k: describe([r.get(k) for r in ok]) for k, _, _ in METRICS}}
        for m in methods:
            if m == baseline:
                continue
            res = {}
            for key, _, direction in METRICS:
                mv = {r["seed"]: r.get(key) for r in lr if r["method"] == m and r["status"] == "ok"}
                bv = {r["seed"]: r.get(key) for r in lr if r["method"] == baseline and r["status"] == "ok"}
                res[key] = paired(mv, bv, direction)
            entry[f"paired_vs_{baseline}"][m] = res
        agg[label] = entry
    return agg


# ---------------------------------------------------------------- plotting (lazy import: backend set first)
def make_plots(agg, ts_store, scenario_meta, methods, out_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg", force=True)          # must happen before pyplot is imported
    import matplotlib.pyplot as plt
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    saved = []

    def stat(entry, m, key):
        d = entry["methods"][m]["metrics"][key]
        return d["mean"], (d["ci95"] or 0.0), d["n"]

    for label, entry in agg.items():                                   # bar charts per scenario
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (key, title, direction) in zip(axes.ravel(), METRICS[:8]):
            for i, m in enumerate(methods):
                mean, ci, n = stat(entry, m, key)
                ax.bar(i, mean or 0.0, yerr=ci, color=COLORS.get(m, "gray"), capsize=4,
                       hatch=None if n else "//", edgecolor="black" if not n else None)
                if not n:
                    ax.text(i, 0, "n/a", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([LABELS.get(m, m).replace(" ", "\n") for m in methods], fontsize=7)
            ax.set_title(f"{title} ({'higher' if direction == 'higher' else 'lower'} is better)", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        n_runs = max(entry["methods"][m]["runs_ok"] for m in methods)
        fig.suptitle(f"{label}: mean with 95% CI over {n_runs} seeds", fontweight="bold")
        fig.tight_layout()
        p = pdir / f"bars_{_slug(label)}.png"
        fig.savefig(p, dpi=160); plt.close(fig); saved.append(str(p))

    for label in agg:                                                  # time series per scenario
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        drew = False
        for key, ax, title in (("avg_delay", axes[0], "Average delay (s)"),
                               ("delivery_ratio_inst", axes[1], "Delivery share (rolling 5 s)"),
                               ("queue_total", axes[2], "Total queued packets")):
            for m in methods:
                runs = [t for t in ts_store.get((label, m), []) if t and len(t.get("t", [])) > 1]
                if not runs:
                    continue
                grid = np.asarray(runs[0]["t"], float)
                arr = np.vstack([np.interp(grid, r["t"], r[key]) for r in runs])
                mu, sd = arr.mean(0), arr.std(0)
                ax.plot(grid, mu, color=COLORS.get(m), label=LABELS.get(m, m), lw=1.8)
                ax.fill_between(grid, mu - sd, mu + sd, color=COLORS.get(m), alpha=0.15)
                drew = True
            for a, b in scenario_meta.get(label, {}).get("failure_windows", []):
                ax.axvspan(a, b, color="red", alpha=0.08)
            ax.set_ylabel(title); ax.grid(True, alpha=0.3)
        if drew:
            axes[0].legend(fontsize=8); axes[2].set_xlabel("Time (s)  (red band = link outage)")
            fig.suptitle(f"{label}: mean ± 1 std across seeds", fontweight="bold")
            fig.tight_layout()
            p = pdir / f"timeseries_{_slug(label)}.png"
            fig.savefig(p, dpi=160); saved.append(str(p))
        plt.close(fig)

    sweeps: Dict[str, list] = {}                                       # load sweep across scenario points
    for label, entry in agg.items():
        if entry["sweep_parameter"]:
            sweeps.setdefault(entry["base_scenario"], []).append(entry)
    for base, entries in sweeps.items():
        entries = sorted(entries, key=lambda e: e["sweep_value"])
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (key, title, _) in zip(axes.ravel(), [m for m in METRICS if m[0] in
                                                      ("delivery_ratio", "avg_delay", "p95_delay", "loss_ratio")]):
            for m in methods:
                xs, ys, es = [], [], []
                for e in entries:
                    mean, ci, n = stat(e, m, key)
                    if n:
                        xs.append(e["sweep_value"]); ys.append(mean); es.append(ci)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=COLORS.get(m), label=LABELS.get(m, m), marker="o", capsize=3)
            ax.set_xlabel(entries[0]["sweep_parameter"]); ax.set_ylabel(title); ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{base}: performance vs offered load (95% CI)", fontweight="bold")
        fig.tight_layout()
        p = pdir / f"load_sweep_{_slug(base)}.png"
        fig.savefig(p, dpi=160); plt.close(fig); saved.append(str(p))
    return saved


# ---------------------------------------------------------------- driver
def run_experiments(base_cfg: Config, experiments: Dict[str, Any], *, methods: List[str], seeds: List[int],
                    out_dir: Path, scenarios: Optional[List[str]] = None, overrides: Optional[Dict[str, Any]] = None,
                    topology_seed: Optional[int] = None, resume: bool = False, keep_timeseries: bool = True,
                    plots: bool = True, save_checkpoints: bool = False, baseline: Optional[str] = None,
                    console: Optional[Console] = None) -> Dict[str, Any]:
    console = console or Console()
    baseline = baseline or experiments.get("baseline") or methods[0]
    scn_list = expand_scenarios(experiments, scenarios)
    if not scn_list:
        raise ValueError("No scenarios selected")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    rows, failures, ts_store, scenario_meta = [], [], {}, {}
    total, done, t_start = len(scn_list) * len(seeds) * len(methods), 0, time.time()
    try:
        for scn in scn_list:
            for seed in seeds:
                cfg = clone_cfg(base_cfg)                      # identical for all methods of this (scenario, seed)
                apply_overrides(cfg, scn["overrides"])
                apply_overrides(cfg, overrides or {})
                cfg.simulation.seed = int(seed)
                if topology_seed is not None:
                    cfg.network.topology_seed = int(topology_seed)
                resolve_cfg(cfg)
                scenario_meta.setdefault(scn["label"], {"description": scn["description"], "failure_windows": [
                    (e["t_start"], e["t_end"]) for e in cfg.network.failure_schedule] if cfg.network.enable_failures else []})
                for method in methods:
                    done += 1
                    run_file = out_dir / "runs" / _slug(scn["label"]) / f"{method}_seed{seed}.json"
                    tag = f"[{done}/{total}] {scn['label']}  {method}  seed={seed}"
                    if resume and run_file.exists():
                        cached = json.loads(run_file.read_text())
                        rows.append(cached["row"]); ts_store.setdefault((scn["label"], method), []).append(cached.get("time_series"))
                        console.print(f"{tag}  [dim]cached[/dim]")
                        continue
                    try:
                        ckpt = out_dir / "checkpoints" / f"{_slug(scn['label'])}_{method}_seed{seed}.pt" \
                            if (save_checkpoints and method == "dqn") else None
                        res = run_one(cfg, method, keep_timeseries=True, checkpoint_path=ckpt)
                        row = make_row(scn, method, seed, res)
                        if res["accounting"].get("balanced") is False:
                            console.print(f"{tag}  [yellow]warning: packet accounting did not balance[/yellow]")
                        run_file.parent.mkdir(parents=True, exist_ok=True)
                        run_file.write_text(json.dumps({"row": row, **{k: res[k] for k in (
                            "summary", "policy", "training", "accounting", "event_log", "runtime_s", "config", "checkpoint")},
                            "time_series": res["time_series"] if keep_timeseries else None}, default=json_default))
                        rows.append(row); ts_store.setdefault((scn["label"], method), []).append(res["time_series"])
                        s = res["summary"]
                        console.print(f"{tag}  delivery={s['delivery_ratio']:.3f} delay={s['avg_delay']:.3f}s "
                                      f"loss={s['loss_ratio']:.3f}  ({res['runtime_s']}s)")
                    except Exception as exc:                    # report, never hide
                        rows.append(make_row(scn, method, seed, None, f"{type(exc).__name__}: {exc}"))
                        failures.append({"scenario": scn["label"], "method": method, "seed": seed,
                                         "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                        console.print(f"{tag}  [bold red]FAILED[/bold red] {type(exc).__name__}: {exc}")
    finally:                                                    # partial results survive Ctrl+C / crashes
        agg = aggregate(rows, methods, baseline) if rows else {}
        cols = list(dict.fromkeys(k for r in rows for k in r))
        if rows:
            with open(out_dir / "raw_runs.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, restval="")
                w.writeheader(); w.writerows(rows)
        (out_dir / "comparison_summaries.json").write_text(json.dumps(
            {"baseline": baseline, "methods": methods, "seeds": seeds, "scenarios": agg}, indent=2, default=json_default))
        (out_dir / "failed_runs.json").write_text(json.dumps(failures, indent=2))
        (out_dir / "experiment_manifest.json").write_text(json.dumps(collect_metadata(
            methods=methods, seeds=seeds, scenarios=[s["label"] for s in scn_list], topology_seed=topology_seed,
            overrides=overrides or {}, planned_runs=total, completed_runs=done, failed_runs=len(failures),
            elapsed_s=round(time.time() - t_start, 1), base_config=base_cfg.to_dict()), indent=2, default=json_default))
    saved = []
    if plots and agg:
        try:
            saved = make_plots(agg, ts_store, scenario_meta, methods, out_dir)
        except Exception as exc:
            console.print(f"[yellow]Plotting failed (data is saved): {type(exc).__name__}: {exc}[/yellow]")
    return {"rows": rows, "failures": failures, "aggregate": agg, "plots": saved, "baseline": baseline}


def print_report(console: Console, out: Dict[str, Any], methods: List[str]) -> None:
    for label, entry in out["aggregate"].items():
        t = Table(title=label, header_style="bold magenta")
        for c in ("Method", "ok/total", "Delivery", "Loss", "Avg delay (s)", "P95 (s)", "Throughput", "Hops", "Route chg/pkt", "Adapt (s)"):
            t.add_column(c, justify="right" if c != "Method" else "left")
        for m in methods:
            e = entry["methods"][m]; mt = e["metrics"]
            def f(k, p=3):
                d = mt[k]; return "n/a" if d["n"] == 0 else f"{d['mean']:.{p}f}" + (f" ±{d['ci95']:.{p}f}" if d["ci95"] is not None else "")
            t.add_row(LABELS.get(m, m), f"{e['runs_ok']}/{e['runs_total']}", f("delivery_ratio"), f("loss_ratio"),
                      f("avg_delay"), f("p95_delay"), f("throughput_pps", 2), f("avg_hops", 2),
                      f("route_changes_per_delivered"), f("adaptation_time_mean", 1))
        console.print(t)
    if out["failures"]:
        t = Table(title=f"{len(out['failures'])} FAILED RUNS (not included in any average)", header_style="bold red")
        for c in ("Scenario", "Method", "Seed", "Error"):
            t.add_column(c)
        for fl in out["failures"]:
            t.add_row(fl["scenario"], fl["method"], str(fl["seed"]), fl["error"][:90])
        console.print(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--experiments", default=str(ROOT / "config" / "experiments.yaml"))
    ap.add_argument("--scenarios", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--methods", nargs="*", choices=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--topology-seed", type=int)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="reuse run files that already exist")
    ap.add_argument("--no-timeseries", action="store_true", help="do not store time series in run files")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true", help="save a DQN checkpoint per DQN run")
    ap.add_argument("--allow-failures", action="store_true", help="exit 0 even if some runs failed")
    args = ap.parse_args(argv)
    console = Console()
    try:
        experiments = yaml.safe_load(Path(args.experiments).read_text())
        base = Config.from_yaml(args.config)
        methods = args.methods or experiments.get("methods") or list(METHODS)
        seeds = args.seeds or experiments.get("seeds") or [base.simulation.seed]
        topo = args.topology_seed if args.topology_seed is not None else experiments.get("topology_seed")
        out = run_experiments(base, experiments, methods=methods, seeds=seeds, out_dir=Path(args.out),
                              scenarios=args.scenarios, overrides=dict(parse_override(s) for s in args.sets),
                              topology_seed=topo, resume=args.resume, keep_timeseries=not args.no_timeseries,
                              plots=not args.no_plots, save_checkpoints=args.save_checkpoints, console=console)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    print_report(console, out, methods)
    console.print(f"\n[bold]Results written to {args.out}[/bold]  ({len(out['rows'])} runs, {len(out['failures'])} failed, {len(out['plots'])} plots)")
    return 2 if out["failures"] and not args.allow_failures else 0


if __name__ == "__main__":
    sys.exit(main())