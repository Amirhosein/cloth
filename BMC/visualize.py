#!/usr/bin/env python3
"""
Interactive visualization of Lightning Network graph with greedy algorithm selection.

Usage:
    python visualize.py <snapshot.json> <budget_in_satoshis> [--max-nodes N] [--output OUTPUT.html]

This script:
1. Loads the Lightning Network snapshot
2. Builds the graph and computes betweenness centrality
3. Runs the greedy algorithm to select nodes
4. Creates an interactive HTML visualization where:
   - Red nodes = Selected by greedy algorithm
   - Blue nodes = Not selected
   - Node size = Betweenness centrality
   - Hover/click nodes to see details
"""

import sys
import argparse
import time
from typing import Dict, Set, Any
import networkx as nx

# Import functions from bmc.py
from bmc import (
    load_channels,
    build_ln_graph,
    compute_node_costs_from_channels,
    compute_betweenness,
    greedy_budgeted_max_value,
)


def visualize_graph_interactive(
    G: nx.DiGraph,
    bet: Dict[Any, float],
    node_cost: Dict[Any, float],
    chosen_nodes: Set[Any],
    output_file: str = "ln_graph.html",
    max_nodes: int = 1000,
    sample_strategy: str = "top_betweenness",
):
    """
    Create an interactive HTML visualization using pyvis.
    
    Args:
        G: NetworkX graph
        bet: Betweenness centrality dict
        node_cost: Node cost dict
        chosen_nodes: Set of selected nodes
        output_file: Output HTML file path
        max_nodes: Maximum number of nodes to visualize (if graph is larger)
        sample_strategy: How to sample nodes if graph is too large
            - "top_betweenness": Keep top nodes by betweenness
            - "selected_plus_neighbors": Keep selected nodes + their neighbors
            - "random": Random sample
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("ERROR: pyvis not installed.")
        print("Install it with: pip install pyvis")
        print("\nFalling back to matplotlib visualization...")
        visualize_graph_static(G, bet, node_cost, chosen_nodes, output_file.replace('.html', '.png'), max_nodes)
        return
    
    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes. Sampling {max_nodes} nodes using '{sample_strategy}' strategy...")
        
        if sample_strategy == "top_betweenness":
            # Keep top nodes by betweenness
            top_nodes = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_set = set([n[0] for n in top_nodes])
            G_viz = G.subgraph(top_node_set).copy()
            
        elif sample_strategy == "selected_plus_neighbors":
            # Keep selected nodes + their neighbors
            selected_set = set(chosen_nodes)
            neighbors = set()
            for node in chosen_nodes:
                if node in G:
                    neighbors.update(G.predecessors(node))
                    neighbors.update(G.successors(node))
            keep_nodes = selected_set | neighbors
            if len(keep_nodes) > max_nodes:
                # If still too many, prioritize selected + top betweenness neighbors
                neighbor_bet = [(n, bet.get(n, 0)) for n in neighbors if n not in selected_set]
                neighbor_bet.sort(key=lambda x: x[1], reverse=True)
                keep_nodes = selected_set | set([n[0] for n in neighbor_bet[:max_nodes - len(selected_set)]])
            G_viz = G.subgraph(keep_nodes).copy()
            
        else:  # random
            import random
            sample_nodes = random.sample(list(G.nodes()), min(max_nodes, G.number_of_nodes()))
            G_viz = G.subgraph(sample_nodes).copy()
    else:
        G_viz = G
    
    print(f"Visualizing {G_viz.number_of_nodes()} nodes and {G_viz.number_of_edges()} edges")
    
    # Compute static layout using NetworkX (this will be fixed, no animation)
    print("Computing static layout (this may take a moment for large graphs)...")
    # Use spring layout with fewer iterations for faster computation
    # Scale positions to fit in a reasonable viewport
    pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)
    
    # Normalize positions to fit in viewport (scale to 0-1000 range)
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        # Scale to 0-1000 range
        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_x = ((x - min_x) / range_x) * 1000
            scaled_y = ((y - min_y) / range_y) * 1000
            scaled_pos[node] = (scaled_x, scaled_y)
    else:
        scaled_pos = {}
    
    # Normalize betweenness for node sizes
    bet_values = [bet.get(node, 0) for node in G_viz.nodes()]
    max_bet = max(bet_values) if bet_values and max(bet_values) > 0 else 1
    min_bet = min([b for b in bet_values if b > 0]) if bet_values else 1
    
    # Create network
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#1a1a1a",
        font_color="white",
        notebook=False,
    )
    
    # Add nodes with fixed positions
    for node in G_viz.nodes():
        node_bet = bet.get(node, 0)
        node_c = node_cost.get(node, 0)
        is_chosen = node in chosen_nodes
        
        # Node size based on betweenness (log scale for better visualization)
        if node_bet > 0:
            # Use log scale to better visualize differences
            normalized_bet = (node_bet - min_bet) / (max_bet - min_bet) if max_bet > min_bet else 0
            size = 15 + 50 * normalized_bet
        else:
            size = 10
        
        # Color: red if chosen, blue otherwise
        # Make selected nodes more vibrant
        if is_chosen:
            color = "#ff4444"  # Bright red
            border_color = "#ff0000"
            border_width = 3
        else:
            color = "#4a90e2"  # Blue
            border_color = "#2e5c8a"
            border_width = 1
        
        # Title with detailed info
        node_str = str(node)
        title = f"Node: {node_str}\n"
        title += f"Betweenness: {node_bet:.2f}\n"
        title += f"Cost: {node_c:.0f} satoshis ({node_c/1e8:.4f} BTC)\n"
        if node_c > 0 and node_bet > 0:
            title += f"Value/Cost Ratio: {node_bet/node_c:.6f}\n"
        if is_chosen:
            title += "\n[SELECTED BY GREEDY ALGORITHM]"
        
        # Label: truncate long node IDs
        label = node_str[:12] + "..." if len(node_str) > 15 else node_str
        
        # Set fixed position if available
        node_params = {
            "label": label,
            "size": size,
            "color": color,
            "title": title,
            "borderWidth": border_width,
            "borderColor": border_color,
        }
        
        # Add fixed position if layout was computed
        if node in scaled_pos:
            x, y = scaled_pos[node]
            node_params["x"] = x
            node_params["y"] = y
            node_params["fixed"] = True  # Lock position
        
        net.add_node(str(node), **node_params)
    
    # Add edges with reduced opacity
    for u, v in G_viz.edges():
        net.add_edge(str(u), str(v), width=0.3, color={"color": "#888888", "opacity": 0.3})
    
    # Disable physics completely - use static layout only
    # This prevents CPU/RAM usage from animation
    net.set_options("""
    {
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "hideEdgesOnZoom": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # Add legend/info at the top
    info_html = f"""
    <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); 
                 color: white; padding: 15px; border-radius: 5px; font-family: Arial; 
                 font-size: 12px; z-index: 1000; max-width: 300px;">
        <h3 style="margin-top: 0;">Lightning Network Visualization</h3>
        <p><strong>Total Nodes:</strong> {G.number_of_nodes()}</p>
        <p><strong>Visualized Nodes:</strong> {G_viz.number_of_nodes()}</p>
        <p><strong>Selected Nodes:</strong> {len([n for n in G_viz.nodes() if n in chosen_nodes])}</p>
        <hr style="border-color: #555;">
        <p><span style="color: #ff4444;">●</span> Red = Selected by greedy algorithm</p>
        <p><span style="color: #4a90e2;">●</span> Blue = Not selected</p>
        <p>Node size = Betweenness centrality</p>
        <p style="font-size: 10px; color: #aaa;">Hover over nodes for details</p>
    </div>
    """
    
    # Save graph
    net.save_graph(output_file)
    
    # Inject custom HTML
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Insert info box before closing body tag
    content = content.replace('</body>', info_html + '</body>')
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Static graph saved to: {output_file}")
    print(f"  Open it in your browser to explore!")
    print(f"  - Graph is STATIC (no animation) - drag to pan, scroll to zoom")
    print(f"  - Red nodes are selected by the greedy algorithm")
    print(f"  - Node size represents betweenness centrality")
    print(f"  - Hover/click nodes to see detailed information")


def visualize_graph_static(
    G: nx.DiGraph,
    bet: Dict[Any, float],
    node_cost: Dict[Any, float],
    chosen_nodes: Set[Any],
    output_file: str = "ln_graph.png",
    max_nodes: int = 500,
):
    """
    Create a static matplotlib visualization (fallback if pyvis not available).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Cannot create static visualization.")
        return
    
    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes. Sampling {max_nodes} nodes...")
        top_nodes = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_set = set([n[0] for n in top_nodes])
        G_viz = G.subgraph(top_node_set).copy()
    else:
        G_viz = G
    
    # Create layout
    print("Computing layout (this may take a while for large graphs)...")
    pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)
    
    # Prepare node sizes (normalize betweenness)
    bet_values = [bet.get(node, 0) for node in G_viz.nodes()]
    max_bet = max(bet_values) if bet_values else 1
    node_sizes = [100 + 1000 * (bet.get(node, 0) / max_bet) if max_bet > 0 else 100 
                  for node in G_viz.nodes()]
    
    # Node colors: red if chosen, blue otherwise
    node_colors = ['#ff4444' if node in chosen_nodes else '#4a90e2' for node in G_viz.nodes()]
    
    # Draw
    plt.figure(figsize=(20, 20))
    nx.draw_networkx_edges(G_viz, pos, alpha=0.1, arrows=False, width=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    
    # Add labels for chosen nodes
    chosen_labels = {node: str(node)[:8] + "..." if len(str(node)) > 8 else str(node) 
                     for node in G_viz.nodes() if node in chosen_nodes}
    nx.draw_networkx_labels(G_viz, pos, chosen_labels, font_size=8, font_color='darkred', font_weight='bold')
    
    plt.title(f"Lightning Network Graph\n"
              f"Red nodes = Selected ({len([n for n in G_viz.nodes() if n in chosen_nodes])}), "
              f"Blue = Others\n"
              f"Node size = Betweenness centrality",
              fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Static graph saved to {output_file}")
    plt.close()


def print_statistics(
    G: nx.DiGraph,
    bet: Dict[Any, float],
    node_cost: Dict[Any, float],
    chosen_nodes: Set[Any],
    total_value: float,
    total_cost: float,
    total_bet_all: float,
):
    """Print detailed statistics about the graph and selected nodes."""
    import numpy as np
    
    print("\n" + "="*70)
    print("GRAPH STATISTICS")
    print("="*70)
    
    # Basic stats
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    # Betweenness stats
    bet_values = list(bet.values())
    if bet_values:
        print(f"\nBetweenness Centrality:")
        print(f"  Max: {max(bet_values):.2f}")
        print(f"  Mean: {np.mean(bet_values):.2f}")
        print(f"  Median: {np.median(bet_values):.2f}")
        if len(bet_values) >= 10:
            top10_sum = sum(sorted(bet_values, reverse=True)[:10])
            print(f"  Top 10 nodes account for {top10_sum / sum(bet_values) * 100:.1f}% of total")
    
    # Cost stats
    cost_values = [node_cost.get(n, 0) for n in G.nodes()]
    cost_values = [c for c in cost_values if c > 0]
    if cost_values:
        print(f"\nNode Costs (satoshis):")
        print(f"  Max: {max(cost_values):.0f} ({max(cost_values)/1e8:.4f} BTC)")
        print(f"  Mean: {np.mean(cost_values):.0f} ({np.mean(cost_values)/1e8:.4f} BTC)")
        print(f"  Median: {np.median(cost_values):.0f} ({np.median(cost_values)/1e8:.4f} BTC)")
    
    # Selected nodes stats
    if chosen_nodes:
        chosen_bet = [bet.get(n, 0) for n in chosen_nodes]
        chosen_costs = [node_cost.get(n, 0) for n in chosen_nodes]
        print(f"\nSelected Nodes ({len(chosen_nodes)}):")
        print(f"  Total betweenness: {total_value:.2f}")
        print(f"  Total cost: {total_cost:.0f} satoshis ({total_cost/1e8:.4f} BTC)")
        print(f"  Avg betweenness: {np.mean(chosen_bet):.2f}")
        print(f"  Avg cost: {np.mean(chosen_costs):.0f} satoshis ({np.mean(chosen_costs)/1e8:.4f} BTC)")
        print(f"  Fraction of total betweenness: {total_value/total_bet_all*100:.2f}%")
        
        # Show top selected nodes
        selected_with_ratios = [(n, bet.get(n, 0), node_cost.get(n, 0), 
                                bet.get(n, 0) / node_cost.get(n, 0) if node_cost.get(n, 0) > 0 else 0)
                               for n in chosen_nodes]
        selected_with_ratios.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\nTop 10 Selected Nodes (by value/cost ratio):")
        print(f"{'Node':<25} {'Betweenness':<15} {'Cost (BTC)':<15} {'Ratio':<10}")
        print("-" * 65)
        for node, b, c, ratio in selected_with_ratios[:10]:
            node_str = str(node)[:23] + ".." if len(str(node)) > 25 else str(node)
            print(f"{node_str:<25} {b:<15.2f} {c/1e8:<15.4f} {ratio:<10.6f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Lightning Network graph with greedy algorithm selection"
    )
    parser.add_argument("snapshot", help="Path to snapshot JSON file")
    parser.add_argument("budget", type=float, help="Budget in satoshis")
    parser.add_argument("--max-nodes", type=int, default=1000,
                       help="Maximum nodes to visualize (default: 1000)")
    parser.add_argument("--output", "-o", default="ln_graph.html",
                       help="Output file path (default: ln_graph.html)")
    parser.add_argument("--sample-strategy", choices=["top_betweenness", "selected_plus_neighbors", "random"],
                       default="top_betweenness",
                       help="Strategy for sampling nodes if graph is too large (default: top_betweenness)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Skip printing statistics")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Lightning Network Visualization Tool")
    print("="*70)
    
    print("\nLoading channels...")
    channels = load_channels(args.snapshot)
    print(f"✓ Loaded {len(channels)} channels")
    
    print("\nBuilding graph...")
    G = build_ln_graph(channels, use_only_active=True)
    print(f"✓ Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nComputing node costs from channel capacities...")
    node_cost = compute_node_costs_from_channels(channels, use_only_active=True)
    print(f"✓ Computed costs for {len(node_cost)} nodes")
    
    print("\nComputing betweenness centrality (Brandes algorithm)...")
    t0 = time.perf_counter()
    bet = compute_betweenness(G, use_edge_weights=False)
    t1 = time.perf_counter()
    print(f"✓ Betweenness computed in {t1 - t0:.2f} seconds")
    
    total_bet_all = sum(bet.values())
    print(f"  Total betweenness over all nodes: {total_bet_all:.2f}")
    
    print("\nRunning greedy budgeted max value selection...")
    chosen_nodes, total_value, total_cost = greedy_budgeted_max_value(
        bet, node_cost, args.budget
    )
    
    frac = total_value / total_bet_all if total_bet_all > 0 else 0.0
    
    print(f"✓ Selected {len(chosen_nodes)} nodes")
    print(f"  Total cost: {total_cost:.0f} satoshis ({total_cost/1e8:.4f} BTC)")
    print(f"  Total betweenness: {total_value:.2f}")
    print(f"  Fraction captured: {frac:.4f} ({frac * 100:.2f}%)")
    
    if not args.no_stats:
        print_statistics(G, bet, node_cost, chosen_nodes, total_value, total_cost, total_bet_all)
    
    print("\nCreating visualization...")
    visualize_graph_interactive(
        G, bet, node_cost, chosen_nodes,
        output_file=args.output,
        max_nodes=args.max_nodes,
        sample_strategy=args.sample_strategy,
    )
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

