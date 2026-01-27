#!/usr/bin/env python3
"""
Adversary Simulation: Greedy node purchase to maximize network control.

This script simulates an adversary with a budget trying to buy nodes
to maximize control (betweenness) over the payment network.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple
from multiprocessing import Pool, cpu_count

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


def count_unique_paths_controlled(
    payments_file: Path,
    bought_nodes: Set[int]
) -> int:
    """
    Count unique successful payment paths that contain at least one bought node.
    
    Args:
        payments_file: Path to payments CSV file
        bought_nodes: Set of node IDs that have been bought by adversary
        
    Returns:
        int: Number of unique successful payments that pass through at least one bought node
    """
    controlled_payments = 0
    
    with open(payments_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            is_success = int(row['is_success'])
            route = row['route'].strip()
            sender_id = int(row['sender_id'])
            receiver_id = int(row['receiver_id'])
            
            # Only process successful payments with valid routes
            if is_success == 1 and route != '-1' and route:
                # Check if sender or receiver is bought
                if sender_id in bought_nodes or receiver_id in bought_nodes:
                    controlled_payments += 1
                    continue
                
                # Check intermediate nodes
                if route:
                    route_nodes = [int(node_id) for node_id in route.split('-') if node_id.strip()]
                    if any(node_id in bought_nodes for node_id in route_nodes):
                        controlled_payments += 1
    
    return controlled_payments


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


def greedy_node_selection(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    budget_millisat: float,
    original_betweenness: Dict[int, int] = None,
    budget_threshold: float = 0.95,
    verbose: bool = False
) -> Tuple[List[int], List[int], List[float]]:
    """
    Greedy algorithm to select nodes maximizing control within budget.
    Selects nodes based on betweenness/cost ratio to maximize efficiency.
    
    Args:
        payments_file: Path to payments CSV
        node_balances: Dict of node balances
        all_nodes: Set of all node IDs
        budget_millisat: Budget in millisatoshis
        original_betweenness: Original betweenness counts (not used, kept for compatibility)
        budget_threshold: Stop when remaining budget is less than this fraction (default 0.95 = 95% spent)
        verbose: Print progress if True
        
    Returns:
        Tuple of (bought_nodes, control_history, budget_history)
        - bought_nodes: List of node IDs purchased
        - control_history: List of unique payment paths controlled after each purchase
        - budget_history: List of remaining budget after each purchase
    """
    
    bought_nodes = []
    exclude_nodes = set()
    remaining_budget = budget_millisat
    initial_budget = budget_millisat
    control_history = [0]
    budget_history = [budget_millisat]
    
    iteration = 0
    
    while True:
        iteration += 1
        
        # Check threshold: stop if we've spent >= threshold% of budget
        budget_spent_ratio = (initial_budget - remaining_budget) / initial_budget
        if budget_spent_ratio >= budget_threshold:
            if verbose:
                print(f"Threshold reached: {budget_spent_ratio*100:.1f}% of budget spent. Stopping.")
            break
        
        # Compute current betweenness excluding bought nodes
        current_analysis = compute_betweenness_with_exclusions(
            payments_file, node_balances, all_nodes, exclude_nodes
        )
        
        # Find best affordable node based on betweenness/cost ratio
        best_node = None
        best_ratio = -1.0  # Initialize to -1 to handle zero ratios
        
        for node_data in current_analysis:
            node_id = node_data['node_id']
            betweenness = node_data['betweenness_count']
            balance = node_data['balance(millisat)']
            
            # Can we afford this node?
            if balance <= remaining_budget:
                # Calculate betweenness/cost ratio
                # Handle zero-cost nodes as infinite ratio (highest priority)
                if balance == 0:
                    ratio = float('inf')
                else:
                    ratio = betweenness / balance
                
                # Select node with highest ratio
                if ratio > best_ratio:
                    best_node = node_id
                    best_ratio = ratio
        
        # No affordable node found
        if best_node is None:
            break
        
        # Buy the node
        node_cost = node_balances[best_node]
        bought_nodes.append(best_node)
        exclude_nodes.add(best_node)
        remaining_budget -= node_cost
        
        # Calculate current total control (unique payment paths controlled)
        # This represents how many unique successful payments pass through at least one bought node
        total_control = count_unique_paths_controlled(payments_file, set(bought_nodes))
        
        control_history.append(total_control)
        budget_history.append(remaining_budget)
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Bought node {best_node}, "
                  f"Paths controlled: {total_control:,}, Budget remaining: {remaining_budget:,.0f} millisat "
                  f"({budget_spent_ratio*100:.1f}% spent)")
    
    if verbose:
        print(f"\nCompleted: Bought {len(bought_nodes)} nodes")
        print(f"Unique payment paths controlled: {control_history[-1]:,}")
        print(f"Budget remaining: {remaining_budget:,.0f} millisat")
        print(f"Budget spent: {(initial_budget - remaining_budget) / initial_budget * 100:.1f}%")
    
    return bought_nodes, control_history, budget_history


def _run_single_budget_simulation(args):
    """
    Worker function for parallel budget simulation.
    
    Args:
        args: Tuple of (index, budget_millisat, payments_file_str, node_balances, all_nodes)
    
    Returns:
        Tuple of (index, control_count, nodes_bought)
    """
    index, budget_millisat, payments_file_str, node_balances, all_nodes = args
    
    payments_file = Path(payments_file_str)
    
    bought_nodes, control_history, _ = greedy_node_selection(
        payments_file, node_balances, all_nodes, budget_millisat,
        original_betweenness=None, budget_threshold=0.95, verbose=False
    )
    
    return (index, control_history[-1], len(bought_nodes))


def run_simulation_btc_budgets(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    btc_budgets: List[float],
    original_betweenness: Dict[int, int],
    num_workers: int = 6
) -> Tuple[List[float], List[int]]:
    """
    Run simulation with BTC-based budgets using parallel processing.
    
    Args:
        num_workers: Number of parallel workers (default: 6)
    
    Returns:
        Tuple of (budgets_btc, controls) - budgets in BTC and corresponding control values
    """
    budgets_millisat = [budget / MILLISAT_TO_BTC for budget in btc_budgets]
    
    print(f"\nRunning simulation with BTC budgets (using {num_workers} workers)...")
    print(f"Testing {len(btc_budgets)} budget levels")
    
    # Prepare arguments for parallel processing
    # Convert Path to string for pickling, and all_nodes set to list
    payments_file_str = str(payments_file)
    all_nodes_list = list(all_nodes)
    
    args_list = [
        (i, budget_millisat, payments_file_str, node_balances, set(all_nodes_list))
        for i, (btc_budget, budget_millisat) in enumerate(zip(btc_budgets, budgets_millisat))
    ]
    
    # Run in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(_run_single_budget_simulation, args_list)
    
    # Sort results by index (maintain order)
    results.sort(key=lambda x: x[0])
    controls = []
    for index, control, nodes in results:
        btc_budget = btc_budgets[index]
        controls.append(control)
        print(f"  Budget {btc_budget:.6f} BTC: Paths controlled: {control:,} (bought {nodes} nodes)")
    
    return btc_budgets, controls


def run_simulation_percentage_budgets(
    payments_file: Path,
    node_balances: Dict[int, float],
    all_nodes: Set[int],
    total_balance: float,
    percentage_budgets: List[float],
    original_betweenness: Dict[int, int],
    num_workers: int = 6
) -> Tuple[List[float], List[int]]:
    """
    Run simulation with percentage-based budgets using parallel processing.
    
    Args:
        num_workers: Number of parallel workers (default: 6)
    
    Returns:
        Tuple of (budgets_percent, controls) - budgets as percentages and corresponding control values
    """
    budgets_millisat = [total_balance * p / 100.0 for p in percentage_budgets]
    
    print(f"\nRunning simulation with percentage budgets (using {num_workers} workers)...")
    print(f"Testing {len(percentage_budgets)} budget levels")
    
    # Prepare arguments for parallel processing
    # Convert Path to string for pickling, and all_nodes set to list
    payments_file_str = str(payments_file)
    all_nodes_list = list(all_nodes)
    
    args_list = [
        (i, budget_millisat, payments_file_str, node_balances, set(all_nodes_list))
        for i, (percent_budget, budget_millisat) in enumerate(zip(percentage_budgets, budgets_millisat))
    ]
    
    # Run in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(_run_single_budget_simulation, args_list)
    
    # Sort results by index (maintain order)
    results.sort(key=lambda x: x[0])
    controls = []
    for index, control, nodes in results:
        percent_budget = percentage_budgets[index]
        controls.append(control)
        print(f"  Budget {percent_budget:.2f}%: Paths controlled: {control:,} (bought {nodes} nodes)")
    
    return percentage_budgets, controls


def plot_results(
    btc_budgets: List[float],
    btc_controls: List[int],
    percent_budgets: List[float],
    percent_controls: List[int],
    output_dir: Path
):
    """Generate and save visualization charts."""
    
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping chart generation")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: BTC-based budgets
    ax1.plot(btc_budgets, btc_controls, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Budget (BTC)', fontsize=12)
    ax1.set_ylabel('Unique Payment Paths Controlled', fontsize=12)
    ax1.set_title('Adversary Control vs Budget (BTC)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Plot 2: Percentage-based budgets
    ax2.plot(percent_budgets, percent_controls, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Budget (% of Network Balance)', fontsize=12)
    ax2.set_ylabel('Unique Payment Paths Controlled', fontsize=12)
    ax2.set_title('Adversary Control vs Budget (% of Network)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'adversary_control_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved chart to {output_file}")
    
    plt.close()


def main():
    # File paths
    base_dir = Path(__file__).parent
    payments_file = base_dir / 'results' / 'outpayments_output1Mil.csv'
    channels_file = base_dir / 'data' / 'channels_ln.csv'
    nodes_file = base_dir / 'data' / 'nodes_ln.csv'
    results_dir = base_dir / 'results'
    
    # Verify input files exist
    for file_path in [payments_file, channels_file, nodes_file]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
    
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
    
    # Define budget ranges
    # BTC budgets: from 0.01 to 20 BTC (exactly 10 budgets, logarithmic scale)
    # Always use 0.01 to 20 BTC range regardless of network balance
    max_btc_budget = 20.0
    if HAS_NUMPY:
        # Use logarithmic scale from 0.01 to 20 BTC (exactly 10 values)
        btc_budgets = np.logspace(np.log10(0.01), np.log10(max_btc_budget), 10).tolist()
        btc_budgets = [round(b, 6) for b in btc_budgets]
        
        # Percentage budgets: from 0.01% to 50% of network balance (exactly 10 budgets)
        percent_budgets = np.linspace(0.01, 10.0, 10).tolist()
        percent_budgets = [round(p, 2) for p in percent_budgets]
    else:
        # Fallback if numpy not available - use manual ranges
        print("Warning: numpy not available, using manual budget ranges")
        # Logarithmic scale: 0.01 * (20/0.01)^(i/9) for i in 0..9
        btc_budgets = [0.01 * ((max_btc_budget / 0.01) ** (i / 9.0)) for i in range(10)]
        btc_budgets = [round(b, 6) for b in btc_budgets]
        # Percentage budgets: evenly spaced from 0.01% to 10%
        percent_budgets = [0.01 + (10.0 - 0.01) * i / 9.0 for i in range(10)]
        percent_budgets = [round(p, 2) for p in percent_budgets]
    
    print(f"\nBTC Budgets (10 values from 0.01 to 20 BTC): {btc_budgets}")
    print(f"Percentage Budgets (10 values from 0.01% to 10%): {percent_budgets}")
    
    # Determine number of workers (use 6 or available CPU count, whichever is smaller)
    num_workers = min(6, cpu_count(), len(btc_budgets))
    print(f"\nUsing {num_workers} parallel workers for simulation")
    
    # Run simulations
    btc_budgets_result, btc_controls = run_simulation_btc_budgets(
        payments_file, node_balances, all_nodes, btc_budgets, original_betweenness,
        num_workers=num_workers
    )
    
    percent_budgets_result, percent_controls = run_simulation_percentage_budgets(
        payments_file, node_balances, all_nodes, total_balance, percent_budgets, original_betweenness,
        num_workers=num_workers
    )
    
    # Generate plots
    print("\nGenerating visualization charts...")
    plot_results(
        btc_budgets_result, btc_controls,
        percent_budgets_result, percent_controls,
        results_dir
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
    # Required for multiprocessing on Windows/macOS
    import multiprocessing
    multiprocessing.freeze_support()
    main()
