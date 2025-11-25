import json
from typing import Dict, List, Tuple, Iterable, Set, Any
import networkx as nx


def load_channels(path: str) -> List[dict]:
    """
    Load Lightning Network channels from a snapshot file.

    Expected format:

    {
        "channels": [
            { "source": "...", "destination": "...", ... },
            ...
        ]
    }

    If the file is a plain JSON array of channels, that is also supported.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "channels" in data:
        return data["channels"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unsupported snapshot format. Expected {'channels': [...]} or a JSON array."
    )


def build_ln_graph(channels: Iterable[dict], use_only_active: bool = True) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from a list of channel dicts.

    Nodes: node ids (strings).
    Edges: directed from 'source' to 'destination'.
           Edge weight is 1 by default (shortest paths by hop count).

    You can later change edge weights to fees or any custom metric.
    """
    G = nx.DiGraph()
    for ch in channels:
        if use_only_active and not ch.get("active", True):
            continue

        u = ch["source"]
        v = ch["destination"]

        satoshis = ch.get("satoshis")
        amount_msat = ch.get("amount_msat")
        base_fee_msat = ch.get("base_fee_millisatoshi", 0)
        fee_per_millionth = ch.get("fee_per_millionth", 0)
        delay = ch.get("delay")

        G.add_edge(
            u,
            v,
            weight=1.0,  # default cost is hop count
            satoshis=satoshis,
            amount_msat=amount_msat,
            base_fee_msat=base_fee_msat,
            fee_per_millionth=fee_per_millionth,
            delay=delay,
        )

    return G


def compute_node_costs_from_channels(
        channels: Iterable[dict],
        use_only_active: bool = True,
) -> Dict[Any, float]:
    """
    Compute cost for each node as the sum of half the capacity of its incident channels.

    For each channel with 'satoshis' = C between u and v:
      cost[u] += C / 2
      cost[v] += C / 2

    This approximates how much capital the node put into the network.
    """
    cost: Dict[Any, float] = {}

    for ch in channels:
        if use_only_active and not ch.get("active", True):
            continue

        u = ch["source"]
        v = ch["destination"]
        satoshis = ch.get("satoshis")

        if satoshis is None:
            continue

        half = float(satoshis) / 2.0

        cost[u] = cost.get(u, 0.0) + half
        cost[v] = cost.get(v, 0.0) + half

    return cost


def compute_betweenness(G: nx.DiGraph, use_edge_weights: bool = False) -> Dict[Any, float]:
    """
    Compute betweenness centrality for each node using Brandes algorithm.

    If use_edge_weights is False:
      shortest paths are based on hop count.

    If use_edge_weights is True:
      shortest paths use the 'weight' attribute of edges.
    """
    if use_edge_weights:
        return nx.betweenness_centrality(G, weight="weight", normalized=False)
    else:
        # ignore edge weights, use unweighted shortest paths
        return nx.betweenness_centrality(G, weight=None, normalized=False)


def greedy_budgeted_max_value(
        value: Dict[Any, float],
        cost: Dict[Any, float],
        budget: float,
) -> Tuple[Set[Any], float, float]:
    """
    Simple greedy knapsack style selection:

      - value[v]: betweenness of node v
      - cost[v]: cost of node v
      - budget: total budget

    Strategy:
      - sort nodes by value / cost ratio in descending order
      - iterate and pick a node if it fits in remaining budget

    Returns:
      chosen_nodes: set of chosen nodes
      total_value: sum of value[v] for chosen nodes
      total_cost: sum of cost[v] for chosen nodes
    """
    # build list of candidates (skip nodes with zero cost or zero value)
    items = []
    for v in value:
        v_cost = cost.get(v, 0.0)
        v_value = value[v]
        if v_cost <= 0 or v_value <= 0:
            continue
        ratio = v_value / v_cost
        items.append((v, v_value, v_cost, ratio))

    # sort by ratio descending
    items.sort(key=lambda x: x[3], reverse=True)

    chosen: Set[Any] = set()
    remaining_budget = float(budget)
    total_value = 0.0
    total_cost = 0.0

    for v, v_value, v_cost, ratio in items:
        if v_cost <= remaining_budget:
            chosen.add(v)
            remaining_budget -= v_cost
            total_value += v_value
            total_cost += v_cost

    return chosen, total_value, total_cost


if __name__ == "__main__":
    """
    Example usage:

      python ln_budget_attack.py snapshot.json 1e9

    where 1e9 is the adversary budget in satoshis.
    """
    import sys
    import time

    if len(sys.argv) < 3:
        print("Usage: python ln_budget_attack.py <snapshot.json> <budget_in_satoshis>")
        sys.exit(1)

    snapshot_path = sys.argv[1]
    budget = float(sys.argv[2])

    print("Loading channels...")
    channels = load_channels(snapshot_path)
    print(f"Loaded {len(channels)} channels")

    print("Building graph...")
    G = build_ln_graph(channels, use_only_active=True)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print("Computing node costs from channel capacities...")
    node_cost = compute_node_costs_from_channels(channels, use_only_active=True)
    print(f"Computed costs for {len(node_cost)} nodes")

    print("Computing betweenness centrality (Brandes)...")
    t0 = time.perf_counter()
    bet = compute_betweenness(G, use_edge_weights=False)
    t1 = time.perf_counter()
    print(f"Betweenness computed for all nodes in {t1 - t0:.2f} seconds")

    total_bet_all = sum(bet.values())
    print(f"Total betweenness over all nodes: {total_bet_all}")

    print("Running greedy budgeted max value selection...")
    chosen_nodes, total_value, total_cost = greedy_budgeted_max_value(
        bet, node_cost, budget
    )

    frac = total_value / total_bet_all if total_bet_all > 0 else 0.0

    print(f"Chosen {len(chosen_nodes)} nodes under budget {budget}")
    print(f"Total chosen cost: {total_cost}")
    print(f"Total betweenness of chosen nodes: {total_value}")
    print(f"Fraction of total betweenness captured: {frac:.4f} ({frac * 100:.2f}%)")

    print("Adversary should buy these nodes:")
    for v in chosen_nodes:
        print(v)
