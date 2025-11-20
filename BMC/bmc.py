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
           Edge weight is 1 by default (shortest path defined by hop count).

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


def build_paths_universe(
        G: nx.DiGraph,
        use_all_shortest_paths: bool = False,
        max_sources: int = None,
) -> Tuple[List[Dict[str, Any]], Dict[Any, List[int]]]:
    """
    Build the universe of paths and the mapping from node -> list of path indices.

    For each ordered pair (s, t), s != t:
      - If use_all_shortest_paths is False:
          use one shortest path from NetworkX.
      - If use_all_shortest_paths is True:
          use all shortest paths between s and t.

    Returns:
      paths: list of dicts:
        - 'nodes': list of nodes on the path [s, ..., t]
        - 'weight': weight of this path (traffic weight)
      node_to_paths: dict node -> list of indices into paths where this node
                     appears as an internal node (not s, not t).
    """
    nodes = list(G.nodes())
    if max_sources is not None:
        nodes = nodes[:max_sources]

    paths: List[Dict[str, Any]] = []
    node_to_paths: Dict[Any, List[int]] = {v: [] for v in G.nodes()}

    for s in nodes:
        sp_dict = nx.single_source_shortest_path(G, s)
        for t, one_path in sp_dict.items():
            if t == s:
                continue

            if use_all_shortest_paths:
                all_paths = list(nx.all_shortest_paths(G, s, t))
                if not all_paths:
                    continue
                weight_per_path = 1.0 / len(all_paths)
                for p in all_paths:
                    idx = len(paths)
                    paths.append({"nodes": p, "weight": weight_per_path})
                    # only internal nodes
                    for v in p[1:-1]:
                        node_to_paths[v].append(idx)
            else:
                p = one_path
                idx = len(paths)
                paths.append({"nodes": p, "weight": 1.0})
                for v in p[1:-1]:
                    node_to_paths[v].append(idx)

    return paths, node_to_paths


def greedy_budgeted_max_coverage(
        node_to_paths: Dict[Any, List[int]],
        paths: List[Dict[str, Any]],
        cost: Dict[Any, float],
        budget: float,
) -> Tuple[Set[Any], float]:
    """
    Greedy algorithm for Budgeted Maximum Coverage similar to Khuller et al.

    At each step it picks the node with maximum marginal gain per unit cost,
    under a knapsack style budget constraint.

    Returns:
      chosen_nodes: set of nodes selected by the algorithm
      total_coverage: sum of weights of covered paths
    """
    num_paths = len(paths)
    covered = [False] * num_paths
    chosen: Set[Any] = set()
    remaining_budget = float(budget)

    # Track best single node solution
    best_single_node = None
    best_single_gain = 0.0

    for v, p_list in node_to_paths.items():
        if cost.get(v, float("inf")) > budget:
            continue
        gain = 0.0
        for p_idx in p_list:
            gain += paths[p_idx]["weight"]
        if gain > best_single_gain:
            best_single_gain = gain
            best_single_node = v

    # Greedy iterations based on marginal gain / cost
    while True:
        best_v = None
        best_ratio = 0.0
        best_gain = 0.0

        for v, p_list in node_to_paths.items():
            if v in chosen:
                continue
            c = cost.get(v, float("inf"))
            if c > remaining_budget:
                continue

            gain = 0.0
            for p_idx in p_list:
                if not covered[p_idx]:
                    gain += paths[p_idx]["weight"]

            if gain <= 0.0:
                continue

            ratio = gain / c
            if ratio > best_ratio:
                best_ratio = ratio
                best_gain = gain
                best_v = v

        if best_v is None:
            break

        chosen.add(best_v)
        remaining_budget -= cost[best_v]
        for p_idx in node_to_paths[best_v]:
            covered[p_idx] = True

    greedy_value = 0.0
    for p_idx, is_cov in enumerate(covered):
        if is_cov:
            greedy_value += paths[p_idx]["weight"]

    # Compare greedy with best single node
    if best_single_node is not None and best_single_gain > greedy_value:
        return {best_single_node}, best_single_gain
    else:
        return chosen, greedy_value


def evaluate_coverage(
        node_set: Iterable[Any],
        node_to_paths: Dict[Any, List[int]],
        paths: List[Dict[str, Any]],
) -> float:
    """
    Compute total coverage (sum of path weights) for a given set of nodes.
    """
    covered = [False] * len(paths)
    for v in node_set:
        for p_idx in node_to_paths.get(v, []):
            covered[p_idx] = True

    total = 0.0
    for p_idx, is_cov in enumerate(covered):
        if is_cov:
            total += paths[p_idx]["weight"]
    return total


def uniform_node_costs(G: nx.DiGraph, cost_value: float = 1.0) -> Dict[Any, float]:
    """
    Assign the same cost to every node.
    """
    return {v: float(cost_value) for v in G.nodes()}


if __name__ == "__main__":
    """
    Example usage:

      python ln_attack_sim.py snapshot.json 10

    Here 10 is the adversary budget.
    With uniform node costs this means you can buy at most 10 nodes.
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ln_attack_sim.py <snapshot.json> <budget>")
        sys.exit(1)

    snapshot_path = sys.argv[1]
    budget = float(sys.argv[2])

    print("Loading channels...")
    channels = load_channels(snapshot_path)
    print(f"Loaded {len(channels)} channels")

    print("Building graph...")
    G = build_ln_graph(channels, use_only_active=True)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print("Building paths universe (this can be expensive)...")
    paths, node_to_paths = build_paths_universe(
        G,
        use_all_shortest_paths=False,  # set True for exact group betweenness style
        max_sources=None,
    )
    print(f"Constructed {len(paths)} paths in the universe")

    print("Assigning node costs (uniform)...")
    cost = uniform_node_costs(G, cost_value=1.0)

    print("Running greedy budgeted max coverage...")
    chosen_nodes, total_cov = greedy_budgeted_max_coverage(
        node_to_paths, paths, cost, budget
    )

    print(f"Chosen {len(chosen_nodes)} nodes under budget {budget}")
    print(f"Total covered path weight: {total_cov}")

    print("Adversary should buy these nodes:")
    for v in chosen_nodes:
        print(v)
