#!/usr/bin/env python3
"""
Analyze payment routes to calculate node betweenness centrality and initial balances.

This script:
1. Parses successful payment routes from the simulation results
2. Counts how many times each node appears in payment paths (betweenness)
3. Calculates initial node balances from channel capacities
4. Outputs results to CSV
"""

import csv
from collections import defaultdict
from pathlib import Path


def parse_payment_routes(payments_file):
    """
    Parse payment routes and count node occurrences in paths.
    
    Returns:
        dict: {node_id: count} - number of times each node appears in payment paths
    """
    node_betweenness = defaultdict(int)
    
    print(f"Reading payment routes from {payments_file}...")
    
    with open(payments_file, 'r') as f:
        reader = csv.DictReader(f)
        successful_count = 0
        total_count = 0
        
        for row in reader:
            total_count += 1
            is_success = int(row['is_success'])
            route = row['route'].strip()
            sender_id = int(row['sender_id'])
            receiver_id = int(row['receiver_id'])
            
            # Only process successful payments with valid routes
            if is_success == 1 and route != '-1' and route:
                successful_count += 1
                
                # Count sender and receiver
                node_betweenness[sender_id] += 1
                node_betweenness[receiver_id] += 1
                
                # Parse intermediate nodes in route (dash-separated)
                if route:
                    route_nodes = [int(node_id) for node_id in route.split('-') if node_id.strip()]
                    for node_id in route_nodes:
                        node_betweenness[node_id] += 1
        
        print(f"Processed {total_count:,} payments")
        print(f"Found {successful_count:,} successful payments with routes")
        print(f"Found {len(node_betweenness):,} unique nodes in payment paths")
    
    return node_betweenness


def calculate_node_balances(channels_file):
    """
    Calculate initial node balances from channel capacities.
    
    Balance = sum of (channel_capacity / 2) for all channels the node is part of.
    
    Returns:
        dict: {node_id: balance} - balance in millisatoshis
    """
    node_balances = defaultdict(float)
    
    print(f"\nReading channel data from {channels_file}...")
    
    with open(channels_file, 'r') as f:
        reader = csv.DictReader(f)
        channel_count = 0
        
        for row in reader:
            channel_count += 1
            node1_id = int(row['node1_id'])
            node2_id = int(row['node2_id'])
            capacity = float(row['capacity(millisat)'])
            
            # Each node gets half the channel capacity
            half_capacity = capacity / 2.0
            node_balances[node1_id] += half_capacity
            node_balances[node2_id] += half_capacity
        
        print(f"Processed {channel_count:,} channels")
        print(f"Found {len(node_balances):,} unique nodes with channels")
    
    return node_balances


def get_all_nodes(nodes_file):
    """
    Read all node IDs from the nodes file.
    
    Returns:
        set: Set of all node IDs
    """
    all_nodes = set()
    
    print(f"\nReading node list from {nodes_file}...")
    
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['id'])
            all_nodes.add(node_id)
    
    print(f"Found {len(all_nodes):,} total nodes in network")
    
    return all_nodes


def main():
    # File paths
    base_dir = Path(__file__).parent
    payments_file = base_dir / 'results' / 'outpayments_output1Mil.csv'
    channels_file = base_dir / 'data' / 'channels_ln.csv'
    nodes_file = base_dir / 'data' / 'nodes_ln.csv'
    output_file = base_dir / 'results' / 'node_betweenness_analysis.csv'
    
    # Verify input files exist
    for file_path in [payments_file, channels_file, nodes_file]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
    
    # Parse payment routes and calculate betweenness
    node_betweenness = parse_payment_routes(payments_file)
    
    # Calculate node balances
    node_balances = calculate_node_balances(channels_file)
    
    # Get all nodes
    all_nodes = get_all_nodes(nodes_file)
    
    # Combine results
    print(f"\nCombining results...")
    results = []
    
    for node_id in sorted(all_nodes):
        betweenness_count = node_betweenness.get(node_id, 0)
        balance = node_balances.get(node_id, 0.0)
        results.append({
            'node_id': node_id,
            'betweenness_count': betweenness_count,
            'balance(millisat)': int(balance) if balance > 0 else 0
        })
    
    # Sort by betweenness count (descending), then by node_id
    results.sort(key=lambda x: (x['betweenness_count'], x['node_id']), reverse=True)
    
    # Write output CSV
    print(f"\nWriting results to {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['node_id', 'betweenness_count', 'balance(millisat)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Successfully wrote {len(results):,} node records to {output_file}")
    
    # Print summary statistics
    nodes_with_routes = sum(1 for r in results if r['betweenness_count'] > 0)
    nodes_with_balance = sum(1 for r in results if r['balance(millisat)'] > 0)
    max_betweenness = max(r['betweenness_count'] for r in results) if results else 0
    max_balance = max(r['balance(millisat)'] for r in results) if results else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Nodes appearing in payment routes: {nodes_with_routes:,}")
    print(f"  Nodes with channel balances: {nodes_with_balance:,}")
    print(f"  Maximum betweenness count: {max_betweenness:,}")
    print(f"  Maximum balance: {max_balance:,} millisat")


if __name__ == '__main__':
    main()
