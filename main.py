#!/usr/bin/env python3
"""Run ONE routing method from the command line and save its summary plus experiment metadata.

Examples
  python main.py                                   # config/default.yaml
  python main.py --method congestion_aware --seed 7
  python main.py --method dqn --save-checkpoint
  python main.py config/default.yaml --set traffic.rate_pps=40 --set simulation.duration=60
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from routing_rl.config import Config
from routing_rl.experiment_utils import (METHODS, apply_overrides, collect_metadata, json_default,
                                         parse_override, resolve_cfg, run_one)


def print_summary(console: Console, method: str, summary: dict, policy_stats: dict) -> None:
    table = Table(title=f"Results: {method}", header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Value", justify="right", style="green")
    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            continue
        table.add_row(k, "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)
    if summary.get("drop_reasons"):
        dr = Table(title="Drop reasons", box=box.SIMPLE)
        dr.add_column("Reason"); dr.add_column("Count", justify="right")
        for reason, cnt in summary["drop_reasons"].items():
            dr.add_row(reason, str(cnt))
        console.print(dr)
    if summary.get("adaptation"):
        ad = Table(title="Adaptation after network events", box=box.SIMPLE)
        for col in ("t (s)", "event", "link", "baseline delivery", "adaptation time (s)", "recovered"):
            ad.add_column(col)
        for a in summary["adaptation"]:
            ad.add_row(f"{a['t']:.1f}", a["kind"], a["detail"],
                       "n/a" if a["baseline_delivery"] is None else f"{a['baseline_delivery']:.3f}",
                       "n/a" if a["adaptation_time"] is None else f"{a['adaptation_time']:.1f}", str(a["recovered"]))
        console.print(ad)
    if policy_stats:
        ps = Table(title="Policy internals", box=box.SIMPLE)
        ps.add_column("Key"); ps.add_column("Value", justify="right")
        for k, v in policy_stats.items():
            ps.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(ps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one routing method and save results.")
    p.add_argument("config", nargs="?", default=str(ROOT / "config" / "default.yaml"), help="YAML config file")
    p.add_argument("--method", choices=METHODS, help="routing method (default: from config)")
    p.add_argument("--seed", type=int, help="traffic / learning seed")
    p.add_argument("--topology-seed", type=int, help="seed of the random topology (default: same as --seed)")
    p.add_argument("--duration", type=float, help="simulated seconds")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. traffic.rate_pps=40 (repeatable)")
    p.add_argument("--out", default="results", help="output directory (default: results)")
    p.add_argument("--tag", default="", help="suffix for the output file names")
    p.add_argument("--save-checkpoint", action="store_true", help="save the DQN model after the run")
    p.add_argument("--strict-accounting", action="store_true", help="fail if packets do not add up")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--debug", action="store_true", help="show full tracebacks")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        cfg_path = Path(args.config)
        if not cfg_path.exists() and (ROOT / args.config).exists():
            cfg_path = ROOT / args.config
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        cfg = Config.from_yaml(str(cfg_path))
        over = dict(parse_override(s) for s in args.sets)
        if args.method: over["routing.method"] = args.method
        if args.seed is not None: over["simulation.seed"] = args.seed
        if args.topology_seed is not None: over["network.topology_seed"] = args.topology_seed
        if args.duration is not None: over["simulation.duration"] = args.duration
        apply_overrides(cfg, over)
        resolve_cfg(cfg)
        method = cfg.routing.method
        if method not in METHODS:
            raise ValueError(f"Unknown routing method {method!r}; choose one of {', '.join(METHODS)}")

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(Panel.fit(
            f"[bold cyan]Adaptive Routing[/bold cyan]  method=[yellow]{method}[/yellow]  topology=[yellow]{cfg.network.topology}[/yellow]  "
            f"seed=[yellow]{cfg.simulation.seed}[/yellow]\ntraffic=[yellow]{cfg.traffic.type} @ {cfg.traffic.rate_pps} pps[/yellow]  "
            f"failures=[yellow]{cfg.network.enable_failures}[/yellow]  duration=[yellow]{cfg.simulation.duration}s[/yellow]",
            border_style="cyan"))

        tag = f"_{args.tag}" if args.tag else ""
        ckpt = out_dir / f"dqn_model{tag}.pt" if (args.save_checkpoint and method == "dqn") else None
        res = run_one(cfg, method, progress=not args.no_progress, strict_accounting=args.strict_accounting,
                      keep_timeseries=False, checkpoint_path=ckpt)
        summary = res["summary"]
        print_summary(console, method, summary, res["policy"])
        if not res["accounting"].get("balanced", True):
            console.print(f"[bold red]Warning:[/bold red] packet accounting did not balance: {res['accounting']}")

        out_path = out_dir / f"results_{method}{tag}.json"
        payload = {"summary": summary, "policy": res["policy"], "accounting": res["accounting"],
                   "training": res["training"], "config": res["config"], "runtime_s": res["runtime_s"],
                   "metadata": collect_metadata(config_file=str(cfg_path), overrides=over)}
        out_path.write_text(json.dumps(payload, indent=2, default=json_default))
        console.print(f"[dim]Saved {out_path}[/dim]")
        if res["checkpoint"]:
            console.print(f"[dim]Saved DQN checkpoint: {res['checkpoint']}[/dim]")
        return 0
    except (FileNotFoundError, ValueError, ImportError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        if args.debug: traceback.print_exc()
        return 2
    except Exception as exc:  # unexpected: still report clearly
        console.print(f"[bold red]Run failed:[/bold red] {type(exc).__name__}: {exc}")
        if args.debug: traceback.print_exc()
        else: console.print("[dim]Re-run with --debug for the full traceback.[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())