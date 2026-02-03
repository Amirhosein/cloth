#!/usr/bin/env python3
"""
Compare adversary simulation curves across multiple result folders.

Each simulation folder is expected to contain:
  - adversary_control_btc.csv      (budget_btc, unique_paths_controlled)
  - adversary_control_percent.csv  (budget_percent, unique_paths_controlled)
  - an outpayments *node-route* CSV to infer the denominator:
      Prefer:
        - outpayments_output1Mil_nodes.csv
        - outpayments_output_nodes.csv
      Otherwise: first match of outpayments*_nodes.csv in the folder.

We convert unique_paths_controlled into:
  covered_pct = 100 * unique_paths_controlled / total_loaded_payments

Where total_loaded_payments matches adversary_simulation.py's loader rule:
  is_success == 1 AND route != '-1' AND route is non-empty.

Outputs a single PNG with two subplots (BTC and % budgets), overlaying all simulations.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Curve:
    label: str
    x: List[float]
    y_pct: List[float]


def _read_two_col_csv(path: Path, x_col: str, y_col: str) -> Tuple[List[float], List[int]]:
    xs: List[float] = []
    ys: List[int] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(int(float(row[y_col])))
    return xs, ys


def _find_payments_nodes_csv(sim_dir: Path) -> Optional[Path]:
    preferred = [
        sim_dir / "outpayments_output1Mil_nodes.csv",
        sim_dir / "outpayments_output_nodes.csv",
        sim_dir / "outpayments_output100k_nodes.csv",
    ]
    for p in preferred:
        if p.exists():
            return p

    matches = sorted(sim_dir.glob("outpayments*_nodes.csv"))
    if matches:
        return matches[0]
    return None


def count_loaded_payments(payments_csv: Path) -> int:
    """
    Count payments that would be loaded by adversary_simulation.py:
      is_success == 1 AND route != '-1' AND route is non-empty.
    """
    n = 0
    with open(payments_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                is_success = int(row["is_success"])
            except Exception:
                continue
            route = (row.get("route") or "").strip()
            if is_success == 1 and route and route != "-1":
                n += 1
    return n


def build_curve(sim_dir: Path, which: str, denom: int) -> Curve:
    if which == "btc":
        csv_path = sim_dir / "adversary_control_btc.csv"
        x_col = "budget_btc"
    elif which == "percent":
        csv_path = sim_dir / "adversary_control_percent.csv"
        x_col = "budget_percent"
    else:
        raise ValueError("which must be 'btc' or 'percent'")

    x, y_abs = _read_two_col_csv(csv_path, x_col=x_col, y_col="unique_paths_controlled")
    y_pct = [100.0 * v / denom if denom > 0 else 0.0 for v in y_abs]

    label = f"{sim_dir.name} (N={denom:,})"
    return Curve(label=label, x=x, y_pct=y_pct)


def main() -> None:
    base_dir = Path(__file__).parent

    p = argparse.ArgumentParser(description="Overlay-compare adversary simulation results.")
    p.add_argument(
        "--sim-dirs",
        nargs="+",
        type=Path,
        default=[
            base_dir / "results/simulation1_0.001-0.02",
            base_dir / "results/simulation2_1$-5$",
        ],
        help="Simulation result directories to compare.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=base_dir / "results/compare_adversary_control_overlay.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Adversary control comparison",
        help="Figure title.",
    )
    args = p.parse_args()

    # Prevent hard crashes from GUI backends (common on macOS/headless runs):
    # - Force a non-interactive backend
    # - Force a writable matplotlib config/cache directory inside the workspace
    mpl_dir = base_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to plot comparisons") from e

    # Resolve sim dirs robustly (support passing relative paths from any cwd)
    sim_dirs: List[Path] = []
    for d in args.sim_dirs:
        cand = d
        if not cand.exists() and not cand.is_absolute():
            cand = base_dir / cand
        if cand.exists():
            sim_dirs.append(cand)

    if not sim_dirs:
        raise FileNotFoundError("No simulation directories found.")

    curves_btc: List[Curve] = []
    curves_pct: List[Curve] = []

    for sim_dir in sim_dirs:
        payments_csv = _find_payments_nodes_csv(sim_dir)
        if payments_csv is None:
            raise FileNotFoundError(
                f"Could not find outpayments*_nodes.csv in {sim_dir}. "
                "Need it to compute denominator for % covered."
            )
        denom = count_loaded_payments(payments_csv)
        if denom <= 0:
            raise ValueError(f"Denominator is 0 for {sim_dir} using {payments_csv}")

        curves_btc.append(build_curve(sim_dir, "btc", denom))
        curves_pct.append(build_curve(sim_dir, "percent", denom))

    # Styling
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    # BTC subplot (log x)
    ax1.set_xscale("log")
    ax1.set_xlabel("Budget (BTC)")
    ax1.set_ylabel("Successful payments covered (%)")
    ax1.grid(True, alpha=0.25)

    # Percent subplot (linear x)
    ax2.set_xlabel("Budget (% of network balance)")
    ax2.set_ylabel("Successful payments covered (%)")
    ax2.grid(True, alpha=0.25)

    for i, (c_btc, c_pct) in enumerate(zip(curves_btc, curves_pct)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Fewer markers for readability on dense curves
        me_btc = max(1, len(c_btc.x) // 12)
        me_pct = max(1, len(c_pct.x) // 12)

        ax1.plot(
            c_btc.x,
            c_btc.y_pct,
            label=c_btc.label,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=5,
            markevery=me_btc,
            alpha=0.9,
        )
        ax2.plot(
            c_pct.x,
            c_pct.y_pct,
            label=c_pct.label,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=5,
            markevery=me_pct,
            alpha=0.9,
        )

    # Align y-limits for fair visual comparison
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    # One shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    # Resolve output path relative to current working directory
    # (so passing "Cloth/results/..." from repo root works)
    out_path = args.out
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote comparison plot to: {out_path}")


if __name__ == "__main__":
    main()

