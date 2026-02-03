#!/usr/bin/env python3
"""
Convert payment routes from edge IDs to intermediate node IDs.

Inputs (default):
  - Cloth/results/outpayments_output1Mil.csv   (route = edge_id-edge_id-...)
  - Cloth/results/outedges_output1Mil.csv      (maps edge id -> to_node_id)

Output (default):
  - Cloth/results/outpayments_output1Mil_nodes.csv

Rule:
  For each successful payment with a valid route:
    edge_ids = route.split('-')
    hop_nodes = [to_node_id(edge) for edge in edge_ids]
    If hop_nodes[-1] == receiver_id: drop the last node (receiver) and keep only intermediates.
    Otherwise: keep all hop_nodes as intermediates (mismatch is logged).
  For failed payments or invalid routes: keep route as '-1'.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def build_edge_to_to_node_map(edges_csv: Path) -> Dict[int, int]:
    edge_to_to_node: Dict[int, int] = {}
    with open(edges_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_id = int(row["id"])
            to_node_id = int(row["to_node_id"])
            edge_to_to_node[edge_id] = to_node_id
    return edge_to_to_node


def convert_payments_routes(
    payments_csv: Path,
    edge_to_to_node: Dict[int, int],
    output_csv: Path,
) -> Tuple[int, int, int, int, int]:
    """
    Returns counters:
      (rows_total, rows_success, converted_success, missing_edge_count, receiver_mismatch_count)
    """
    rows_total = 0
    rows_success = 0
    converted_success = 0
    missing_edge_count = 0
    receiver_mismatch_count = 0
    direct_routes_count = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(payments_csv, "r") as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {payments_csv}")

        fieldnames = list(reader.fieldnames)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rows_total += 1
            is_success = int(row["is_success"])
            route = (row.get("route") or "").strip()

            if not (is_success == 1 and route and route != "-1"):
                # Keep as-is for failures / invalid routes
                row["route"] = "-1" if (not route or route == "-1") else route
                writer.writerow(row)
                continue

            rows_success += 1

            sender_id = int(row["sender_id"])
            receiver_id = int(row["receiver_id"])

            edge_ids: List[int] = [int(x) for x in route.split("-") if x.strip()]
            hop_nodes: List[int] = []

            missing = False
            for e in edge_ids:
                to_node = edge_to_to_node.get(e)
                if to_node is None:
                    missing_edge_count += 1
                    missing = True
                    break
                hop_nodes.append(to_node)

            if missing:
                # Can't safely convert; mark route invalid so downstream ignores it.
                row["route"] = "-1"
                writer.writerow(row)
                continue

            # hop_nodes are destinations of each hop; sender is NOT in this list by definition.
            # Usually hop_nodes[-1] should be receiver_id.
            intermediates = hop_nodes
            if hop_nodes:
                if hop_nodes[-1] == receiver_id:
                    intermediates = hop_nodes[:-1]  # drop receiver
                    if len(hop_nodes) == 1:
                        direct_routes_count += 1
                else:
                    receiver_mismatch_count += 1
                    # Keep all hop_nodes as intermediates; receiver isn't the last hop.

            # Extra sanity: intermediates should not include sender; if it does, it's okay but unexpected.
            # We keep it and rely on logging/mismatch counters.
            row["route"] = "-".join(str(n) for n in intermediates)

            converted_success += 1
            writer.writerow(row)

    print("Conversion complete.")
    print(f"  payments file: {payments_csv}")
    print(f"  edges file:    {len(edge_to_to_node):,} edges loaded")
    print(f"  output file:   {output_csv}")
    print(f"  rows_total:            {rows_total:,}")
    print(f"  rows_success:          {rows_success:,}")
    print(f"  converted_success:     {converted_success:,}")
    print(f"  missing_edge_count:    {missing_edge_count:,}")
    print(f"  receiver_mismatches:   {receiver_mismatch_count:,}")
    print(f"  direct_routes_count:   {direct_routes_count:,}")

    return (
        rows_total,
        rows_success,
        converted_success,
        missing_edge_count,
        receiver_mismatch_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert payment routes from edges to nodes.")
    parser.add_argument(
        "--payments",
        type=Path,
        default=Path("results/outpayments_output.csv"),
        help="Input payments CSV (route uses edge ids).",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("results/outedges_output.csv"),
        help="Input edges CSV (contains to_node_id).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/outpayments_output_nodes.csv"),
        help="Output payments CSV (route uses intermediate node ids).",
    )
    args = parser.parse_args()

    edges_csv = args.edges
    payments_csv = args.payments
    output_csv = args.output

    if not edges_csv.exists():
        raise FileNotFoundError(edges_csv)
    if not payments_csv.exists():
        raise FileNotFoundError(payments_csv)

    edge_to_to_node = build_edge_to_to_node_map(edges_csv)
    convert_payments_routes(payments_csv, edge_to_to_node, output_csv)


if __name__ == "__main__":
    main()

