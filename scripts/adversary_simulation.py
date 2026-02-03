#!/usr/bin/env python3
"""
Adversary Simulation: Greedy node purchase to maximize network control.

This script simulates an adversary with a budget trying to buy nodes
to maximize control (betweenness) over the payment network.

Usage:
  python adversary_simulation.py --payments <payments.csv> --channels <channels_ln.csv> --nodes <nodes_ln.csv> [--output-dir <dir>]
"""

import argparse
import csv
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Constants
MILLISAT_TO_BTC = 1e-11  # 1 millisat = 1e-11 BTC

# =========================
# Budget configuration
# =========================
# BTC budget grid (log scale)
BTC_BUDGET_MIN = 0.01
BTC_BUDGET_MAX = 20.0
BTC_BUDGET_POINTS = 50

# Percent-of-network budget grid (linear scale)
PCT_BUDGET_MIN = 0.01
PCT_BUDGET_MAX = 4.0
PCT_BUDGET_POINTS = 50


def parse_payment_routes(payments_file: Path, exclude_nodes: Set[int] = None) -> Dict[int, int]:
    """
    Parse payment routes and count node occurrences in paths.
    Excludes paths containing any node in exclude_nodes.
    
    Args:
        payments_file: Path to payments CSV file
        exclude_nodes: Set of node IDs to exclude from paths
        
    Returns:
        dict: {node_id: count} - number of times each node appears in payment paths
    """
    if exclude_nodes is None:
        exclude_nodes = set()
    
    node_betweenness = defaultdict(int)
    
    with open(payments_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            is_success = int(row['is_success'])
            route = row['route'].strip()
            sender_id = int(row['sender_id'])
            receiver_id = int(row['receiver_id'])
            
            # Only process successful payments with valid routes
            if is_success == 1 and route != '-1' and route:
                # Skip paths containing any excluded node
                if sender_id in exclude_nodes or receiver_id in exclude_nodes:
                    continue
                
                # Check intermediate nodes
                route_nodes = []
                if route:
                    route_nodes = [int(node_id) for node_id in route.split('-') if node_id.strip()]
                    if any(node_id in exclude_nodes for node_id in route_nodes):
                        continue
                
                # Count sender and receiver
                node_betweenness[sender_id] += 1
                node_betweenness[receiver_id] += 1
                
                # Count intermediate nodes
                for node_id in route_nodes:
                    node_betweenness[node_id] += 1
    
    return node_betweenness


def calculate_node_balances(channels_file: Path) -> Dict[int, float]:
    """
    Calculate initial node balances from channel capacities.
    
    Returns:
        dict: {node_id: balance} - balance in millisatoshis
    """
    node_balances = defaultdict(float)
    
    with open(channels_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            node1_id = int(row['node1_id'])
            node2_id = int(row['node2_id'])
            capacity = float(row['capacity(millisat)'])
            
            # Each node gets half the channel capacity
            half_capacity = capacity / 2.0
            node_balances[node1_id] += half_capacity
            node_balances[node2_id] += half_capacity
    
    return node_balances


def get_total_network_balance(node_balances: Dict[int, float]) -> float:
    """Calculate total network balance in millisatoshis."""
    return sum(node_balances.values())


def load_successful_payments(
    payments_file: Path,
) -> List[Tuple[int, int, List[int]]]:
    """
    Load all successful payment paths into memory.

    Each element is a tuple: (sender_id, receiver_id, [intermediate_node_ids]).
    Failed or invalid routes (-1 or empty) are skipped.

    NOTE: This function expects `route` to contain intermediate *node ids* (NOT edge ids).
    Use `convert_edges.py` to generate the node-route CSV from the edge-route CSV.
    """
    payment_paths: List[Tuple[int, int, List[int]]] = []

    with open(payments_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_success = int(row["is_success"])
            route = row["route"].strip()
            if not (is_success == 1 and route and route != "-1"):
                continue

            sender_id = int(row["sender_id"])
            receiver_id = int(row["receiver_id"])

            intermediates: List[int] = [
                int(node_id) for node_id in route.split("-") if node_id.strip()
            ]

            payment_paths.append((sender_id, receiver_id, intermediates))

    return payment_paths


def build_payments_of_node(
    payment_paths: List[Tuple[int, int, List[int]]],
) -> Dict[int, List[int]]:
    """
    Build incidence: for each node, list of payment indices where it appears.

    IMPORTANT: We only include **intermediate nodes** (the `route` field) and
    intentionally exclude the sender/receiver endpoints. This prevents the
    adversary from getting “credit” for buying leaf endpoint nodes.
    """
    payments_of_node: Dict[int, List[int]] = defaultdict(list)

    for pid, (sender_id, receiver_id, intermediates) in enumerate(payment_paths):
        # Only use the intermediate nodes from the route field.
        # (Sender/receiver are excluded by design.)
        for node_id in set(intermediates):
            payments_of_node[node_id].append(pid)

    return payments_of_node


def compute_betweenness_with_exclusions(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    exclude_nodes: Set[int]
) -> List[Dict]:
    """
    Compute betweenness analysis excluding certain nodes.
    
    Returns:
        list: List of dicts with node_id, betweenness_count, balance(millisat)
    """
    node_betweenness = parse_payment_routes(payments_file, exclude_nodes)
    
    results = []
    for node_id in sorted(all_nodes):
        if node_id in exclude_nodes:
            continue  # Skip already bought nodes
        
        betweenness_count = node_betweenness.get(node_id, 0)
        balance = node_balances.get(node_id, 0.0)
        results.append({
            'node_id': node_id,
            'betweenness_count': betweenness_count,
            'balance(millisat)': int(balance) if balance > 0 else 0
        })
    
    # Sort by betweenness count (descending)
    results.sort(key=lambda x: (x['betweenness_count'], x['node_id']), reverse=True)
    
    return results


def celf_greedy_selection(
    payment_paths: List[Tuple[int, int, List[int]]],
    payments_of_node: Dict[int, List[int]],
    node_costs: Dict[int, float],
    budget_millisat: float,
    budget_threshold: float = 0.95,
    verbose: bool = False,
) -> Tuple[List[int], int, float]:
    """
    CELF-style greedy selection to maximize covered payments under a budget.

    - Universe: payment indices [0..n_payments-1]
    - Each node v is associated with payments_of_node[v] ⊆ {0..n_payments-1}
    - Cost of node v is node_costs[v] (millisat)
    - Objective: maximize # of covered payments under total cost ≤ budget_millisat
    """
    n_payments = len(payment_paths)
    covered = [False] * n_payments

    bought_nodes: List[int] = []
    remaining_budget = float(budget_millisat)
    initial_budget = float(budget_millisat)

    # Max-heap implemented as min-heap with negative ratio
    heap: List[Tuple[float, int, int]] = []  # (-ratio, node_id, last_gain)

    # Initial marginal gains: all payments uncovered
    for node_id, pids in payments_of_node.items():
        cost = float(node_costs.get(node_id, 0.0))
        if cost <= 0 or cost > remaining_budget:
            continue
        gain = len(pids)
        if gain <= 0:
            continue
        ratio = gain / cost
        heapq.heappush(heap, (-ratio, node_id, gain))

    if verbose:
        print(
            f"[CELF] Initialized {len(heap)} candidate nodes "
            f"for budget {budget_millisat:,.0f} millisat"
        )

    iterations = 0

    while heap and remaining_budget > 0:
        iterations += 1
        budget_spent_ratio = (
            (initial_budget - remaining_budget) / initial_budget if initial_budget > 0 else 0.0
        )
        if budget_spent_ratio >= budget_threshold:
            if verbose:
                print(
                    f"[CELF] Threshold reached: {budget_spent_ratio*100:.1f}% of budget spent. Stopping."
                )
            break

        neg_ratio, node_id, old_gain = heapq.heappop(heap)
        cost = float(node_costs.get(node_id, 0.0))
        if cost <= 0 or cost > remaining_budget:
            continue

        # Recompute marginal gain with current covered[]
        new_gain = 0
        for pid in payments_of_node.get(node_id, []):
            if not covered[pid]:
                new_gain += 1

        if new_gain <= 0:
            # No additional coverage
            continue

        new_ratio = new_gain / cost if cost > 0 else float("inf")

        # Lazy evaluation: check if still best
        if heap:
            next_neg_ratio, _, _ = heap[0]
            if -next_neg_ratio > new_ratio:
                # Someone else looks better; reinsert with updated ratio
                heapq.heappush(heap, (-new_ratio, node_id, new_gain))
                continue

        # Accept this node
        newly_covered = 0
        for pid in payments_of_node.get(node_id, []):
            if not covered[pid]:
                covered[pid] = True
                newly_covered += 1

        bought_nodes.append(node_id)
        remaining_budget -= cost

        if verbose:
            print(
                f"[CELF] Iteration {iterations}: bought node {node_id} "
                f"(gain={newly_covered}, cost={cost:,.0f}, ratio={new_ratio:.6f}), "
                f"remaining budget={remaining_budget:,.0f}"
            )

    total_covered = sum(1 for c in covered if c)

    if verbose:
        spent_ratio = (
            (initial_budget - remaining_budget) / initial_budget if initial_budget > 0 else 0.0
        )
        print(
            f"[CELF] Completed selection: bought {len(bought_nodes)} nodes, "
            f"covered {total_covered:,} / {n_payments:,} payments, "
            f"budget spent ≈ {spent_ratio*100:.2f}%"
        )

    return bought_nodes, total_covered, remaining_budget


def run_simulation_btc_budgets(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    btc_budgets: List[float],
    original_betweenness: Dict[int, int],
    payment_paths: List[Tuple[int, int, List[int]]],
    payments_of_node: Dict[int, List[int]],
    node_costs: Dict[int, float],
) -> Tuple[List[float], List[int]]:
    """
    Run simulation with BTC-based budgets sequentially.
    
    Returns:
        Tuple of (budgets_btc, controls) - budgets in BTC and corresponding control values
    """
    budgets_millisat = [budget / MILLISAT_TO_BTC for budget in btc_budgets]
    
    print(f"\nRunning simulation with BTC budgets (sequential)...")
    print(f"Testing {len(btc_budgets)} budget levels")
    
    controls: List[int] = []
    
    for i, (btc_budget, budget_millisat) in enumerate(zip(btc_budgets, budgets_millisat), start=1):
        print(f"\n[BTC {i}/{len(btc_budgets)}] Starting budget {btc_budget:.6f} BTC "
              f"(≈ {budget_millisat:,.0f} millisat)")
        
        bought_nodes, total_covered, remaining_budget = celf_greedy_selection(
            payment_paths,
            payments_of_node,
            node_costs,
            budget_millisat,
            budget_threshold=0.95,
            verbose=False,
        )
        
        controls.append(total_covered)
        
        spent_ratio = 0.0
        if budget_millisat > 0:
            spent_ratio = (budget_millisat - remaining_budget) / budget_millisat
        
        print(f"[BTC {i}/{len(btc_budgets)}] Finished budget {btc_budget:.6f} BTC: "
              f"paths controlled = {total_covered:,}, nodes bought = {len(bought_nodes)}, "
              f"budget used ≈ {spent_ratio*100:.2f}%")
    
    return btc_budgets, controls


def run_simulation_percentage_budgets(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    total_balance: float,
    percentage_budgets: List[float],
    original_betweenness: Dict[int, int],
    payment_paths: List[Tuple[int, int, List[int]]],
    payments_of_node: Dict[int, List[int]],
    node_costs: Dict[int, float],
) -> Tuple[List[float], List[int]]:
    """
    Run simulation with percentage-based budgets sequentially.
    
    Returns:
        Tuple of (budgets_percent, controls) - budgets as percentages and corresponding control values
    """
    budgets_millisat = [total_balance * p / 100.0 for p in percentage_budgets]
    
    print(f"\nRunning simulation with percentage budgets (sequential)...")
    print(f"Testing {len(percentage_budgets)} budget levels")
    
    controls: List[int] = []
    
    for i, (percent_budget, budget_millisat) in enumerate(
        zip(percentage_budgets, budgets_millisat), start=1
    ):
        print(f"\n[PCT {i}/{len(percentage_budgets)}] Starting budget {percent_budget:.2f}% "
              f"(≈ {budget_millisat:,.0f} millisat)")
        
        bought_nodes, total_covered, remaining_budget = celf_greedy_selection(
            payment_paths,
            payments_of_node,
            node_costs,
            budget_millisat,
            budget_threshold=0.95,
            verbose=False,
        )
        
        controls.append(total_covered)
        
        spent_ratio = 0.0
        if budget_millisat > 0:
            spent_ratio = (budget_millisat - remaining_budget) / budget_millisat
        
        print(f"[PCT {i}/{len(percentage_budgets)}] Finished budget {percent_budget:.2f}%: "
              f"paths controlled = {total_covered:,}, nodes bought = {len(bought_nodes)}, "
              f"budget used ≈ {spent_ratio*100:.2f}%")
    
    return percentage_budgets, controls


def plot_results(
    btc_budgets: List[float],
    btc_controls: List[int],
    percent_budgets: List[float],
    percent_controls: List[int],
    total_payments: int,
    output_dir: Path,
):
    """Generate and save visualization charts.

    Plots y-axis as percentage of successful payments covered.
    """

    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping chart generation")
        return

    if total_payments <= 0:
        print("Warning: total_payments is non-positive; skipping chart generation")
        return

    # Convert absolute counts to percentages of covered payments
    btc_controls_pct = [100.0 * c / total_payments for c in btc_controls]
    percent_controls_pct = [100.0 * c / total_payments for c in percent_controls]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: BTC-based budgets
    ax1.plot(btc_budgets, btc_controls_pct, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Budget (BTC)", fontsize=12)
    ax1.set_ylabel("Successful Payments Covered (%)", fontsize=12)
    ax1.set_title("Adversary Control vs Budget (BTC)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="plain", axis="x", scilimits=(0, 0))

    # Plot 2: Percentage-based budgets
    ax2.plot(
        percent_budgets, percent_controls_pct, "r-", linewidth=2, marker="s", markersize=4
    )
    ax2.set_xlabel("Budget (% of Network Balance)", fontsize=12)
    ax2.set_ylabel("Successful Payments Covered (%)", fontsize=12)
    ax2.set_title("Adversary Control vs Budget (% of Network)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "adversary_control_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved chart to {output_file}")

    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversary simulation: greedy node purchase to maximize control over payment paths.",
    )
    p.add_argument(
        "--payments",
        type=str,
        required=True,
        help="Path to payments CSV (with columns: sender_id, receiver_id, route, is_success).",
    )
    p.add_argument(
        "--channels",
        type=str,
        required=True,
        help="Path to channels_ln.csv (node1_id, node2_id, capacity(millisat)).",
    )
    p.add_argument(
        "--nodes",
        type=str,
        required=True,
        help="Path to nodes_ln.csv (id column).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output CSVs and plot (default: results).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    payments_file = Path(args.payments)
    channels_file = Path(args.channels)
    nodes_file = Path(args.nodes)
    results_dir = Path(args.output_dir)

    # Verify input files exist
    for file_path in [payments_file, channels_file, nodes_file]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Adversary Simulation: Greedy Node Purchase")
    print("=" * 60)
    print("\nNote: This simulation may take a long time as it processes")
    print("1 million payments multiple times. Please be patient.")
    print("=" * 60)
    
    # Load network data
    print("\nLoading network data...")
    node_balances = calculate_node_balances(channels_file)
    total_balance = get_total_network_balance(node_balances)
    
    all_nodes = set()
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_nodes.add(int(row['id']))
    
    print(f"Total network balance: {total_balance:,.0f} millisat ({total_balance * MILLISAT_TO_BTC:.6f} BTC)")
    print(f"Total nodes: {len(all_nodes):,}")
    
    # Compute original betweenness (for control calculation)
    print("\nComputing original network betweenness...")
    original_betweenness = parse_payment_routes(payments_file, exclude_nodes=set())
    total_node_appearances = sum(original_betweenness.values())
    
    # Count successful transactions for context
    successful_txns = 0
    with open(payments_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['is_success']) == 1 and row['route'].strip() != '-1' and row['route'].strip():
                successful_txns += 1
    
    print(f"Successful transactions: {successful_txns:,}")
    print(f"Total node appearances in payment paths: {total_node_appearances:,}")
    print(f"(This is the sum of all node occurrences, not the number of transactions)")
    
    # Define budget ranges (configured at top of file)
    if BTC_BUDGET_POINTS <= 0 or PCT_BUDGET_POINTS <= 0:
        raise ValueError("Budget points must be positive.")
    if BTC_BUDGET_MIN <= 0 or BTC_BUDGET_MAX <= 0 or BTC_BUDGET_MIN > BTC_BUDGET_MAX:
        raise ValueError("BTC_BUDGET_MIN/BTC_BUDGET_MAX must be positive and min <= max.")
    if PCT_BUDGET_MIN < 0 or PCT_BUDGET_MAX < 0 or PCT_BUDGET_MIN > PCT_BUDGET_MAX:
        raise ValueError("PCT_BUDGET_MIN/PCT_BUDGET_MAX must satisfy 0 <= min <= max.")

    if HAS_NUMPY:
        # Use logarithmic scale for BTC budgets
        if BTC_BUDGET_POINTS == 1:
            btc_budgets = [float(BTC_BUDGET_MIN)]
        else:
            btc_budgets = np.logspace(
                np.log10(BTC_BUDGET_MIN),
                np.log10(BTC_BUDGET_MAX),
                BTC_BUDGET_POINTS,
            ).tolist()
        btc_budgets = [round(b, 6) for b in btc_budgets]
        
        # Use linear scale for percent budgets
        percent_budgets = np.linspace(PCT_BUDGET_MIN, PCT_BUDGET_MAX, PCT_BUDGET_POINTS).tolist()
        percent_budgets = [round(p, 2) for p in percent_budgets]
    else:
        # Fallback if numpy not available - use manual ranges
        print("Warning: numpy not available, using manual budget ranges")
        # Logarithmic BTC grid
        if BTC_BUDGET_POINTS == 1:
            btc_budgets = [float(BTC_BUDGET_MIN)]
        else:
            btc_budgets = [
                BTC_BUDGET_MIN
                * ((BTC_BUDGET_MAX / BTC_BUDGET_MIN) ** (i / (BTC_BUDGET_POINTS - 1)))
                for i in range(BTC_BUDGET_POINTS)
            ]
        btc_budgets = [round(b, 6) for b in btc_budgets]

        # Linear percent grid
        if PCT_BUDGET_POINTS == 1:
            percent_budgets = [float(PCT_BUDGET_MIN)]
        else:
            percent_budgets = [
                PCT_BUDGET_MIN
                + (PCT_BUDGET_MAX - PCT_BUDGET_MIN) * i / (PCT_BUDGET_POINTS - 1)
                for i in range(PCT_BUDGET_POINTS)
            ]
        percent_budgets = [round(p, 2) for p in percent_budgets]
    
    print(
        f"\nBTC Budgets ({len(btc_budgets)} points, logspace): "
        f"{BTC_BUDGET_MIN} BTC -> {BTC_BUDGET_MAX} BTC"
    )
    print(
        f"Percent Budgets ({len(percent_budgets)} points, linspace): "
        f"{PCT_BUDGET_MIN}% -> {PCT_BUDGET_MAX}%"
    )
    print(
        f"Total BTC budget span: {min(btc_budgets):.6f} BTC -> {max(btc_budgets):.6f} BTC"
    )
    print(
        f"Total percent budget span: {min(percent_budgets):.2f}% -> {max(percent_budgets):.2f}%"
    )

    # Build coverage structures for CELF-based selection
    print("\nLoading successful payments into memory for coverage-based selection...")
    payment_paths = load_successful_payments(payments_file)
    print(f"Loaded {len(payment_paths):,} successful payment paths")

    print("Building node–payment incidence (payments_of_node)...")
    payments_of_node = build_payments_of_node(payment_paths)
    print(f"Found {len(payments_of_node):,} nodes with at least one incident payment")

    # Node costs in millisatoshi (based on initial balances)
    node_costs: Dict[int, float] = {
        node_id: float(balance) for node_id, balance in node_balances.items() if balance > 0
    }
    print(f"Nodes with positive cost (balance): {len(node_costs):,}")
    
    # Run simulations sequentially using CELF-based coverage optimization
    btc_budgets_result, btc_controls = run_simulation_btc_budgets(
        payments_file,
        node_balances,
        all_nodes,
        btc_budgets,
        original_betweenness,
        payment_paths,
        payments_of_node,
        node_costs,
    )
    
    percent_budgets_result, percent_controls = run_simulation_percentage_budgets(
        payments_file,
        node_balances,
        all_nodes,
        total_balance,
        percent_budgets,
        original_betweenness,
        payment_paths,
        payments_of_node,
        node_costs,
    )
    
    # Generate plots (as percentage of successful payments covered)
    print("\nGenerating visualization charts (percentage of successful payments covered)...")
    plot_results(
        btc_budgets_result,
        btc_controls,
        percent_budgets_result,
        percent_controls,
        len(payment_paths),
        results_dir,
    )
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    
    # BTC results
    btc_output_file = results_dir / 'adversary_control_btc.csv'
    with open(btc_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['budget_btc', 'unique_paths_controlled'])
        for budget, control in zip(btc_budgets_result, btc_controls):
            writer.writerow([budget, control])
    print(f"Saved BTC results to {btc_output_file}")
    
    # Percentage results
    percent_output_file = results_dir / 'adversary_control_percent.csv'
    with open(percent_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['budget_percent', 'unique_paths_controlled'])
        for budget, control in zip(percent_budgets_result, percent_controls):
            writer.writerow([budget, control])
    print(f"Saved percentage results to {percent_output_file}")
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
