#!/usr/bin/env python3
"""
Count distinct node pubkeys in an lncli describegraph-style snapshot.

Reports three numbers:
- Nodes from the top-level `nodes` array
- Nodes from channel endpoints (`node1_pub` / `node2_pub` in `edges`)
- The union of both sets

Usage:
  python count_lncli_nodes.py lightning_snapshot_YYYY-MM-DD.json
"""

import argparse
import json
import sys
from typing import Set, Tuple


def load_lncli(path: str):
    """Load lncli snapshot JSON and return (nodes_array, edges_array)."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("lncli snapshot must be a JSON object")
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("lncli snapshot must have 'nodes' array")
    edges = data.get("edges")
    if not isinstance(edges, list):
        raise ValueError("lncli snapshot must have 'edges' array")
    return nodes, edges


def collect_pubkeys(nodes, edges) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (from_nodes, from_edges, union)."""
    from_nodes = set()
    from_edges = set()

    for n in nodes:
        if isinstance(n, dict):
            pk = n.get("pub_key")
            if pk:
                from_nodes.add(pk)

    for e in edges:
        if not isinstance(e, dict):
            continue
        n1 = e.get("node1_pub")
        n2 = e.get("node2_pub")
        if n1:
            from_edges.add(n1)
        if n2:
            from_edges.add(n2)

    union = from_nodes | from_edges
    return from_nodes, from_edges, union


def parse_args():
    p = argparse.ArgumentParser(
        description="Count node pubkeys in lncli snapshot (nodes vs channel endpoints vs union)."
    )
    p.add_argument("snapshot", help="Path to lightning_snapshot_YYYY-MM-DD.json")
    return p.parse_args()


def main():
    args = parse_args()
    nodes, edges = load_lncli(args.snapshot)
    from_nodes, from_edges, union = collect_pubkeys(nodes, edges)

    print(f"Nodes from 'nodes' array      : {len(from_nodes)}")
    print(f"Nodes from channel endpoints : {len(from_edges)}")
    print(f"Union of both                : {len(union)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

