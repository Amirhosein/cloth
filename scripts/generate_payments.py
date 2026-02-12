#!/usr/bin/env python3
"""
generate_payments.py

Create a payments CSV similar to cloth sample:
columns: id, sender_id, receiver_id, {amount_col}, start_time

- id:            0..N-1 (int)
- sender_id:     random int in [sender_min, sender_max]
- receiver_id:   random int in [receiver_min, receiver_max] (optionally distinct from sender_id)
- {amount_col}:  random int in [min_amount, max_amount]
- start_time:    integer timestamp-like value
                 * relative: starts at --start-base and increases by random steps in [--step-min, --step-max]
                 * epoch:    UNIX seconds, starting near "now" and stepping forward

Usage examples:
    python generate_payments.py -n 100 --min-amount 90000 --max-amount 110000
    python generate_payments.py -n 10 --min-amount 1000 --max-amount 2500 --amount-colname "amount(millisat)" --output /tmp/payments.csv
    python generate_payments.py -n 50 --min-amount 50 --max-amount 500 --start-mode epoch --step-min 5 --step-max 45 --seed 123


    python3 generate_payments.py -n 10000 --min-amount 10000000 --max-amount 100000000 --output payments/payments0.csv

    With capacity-proportional sampling (one endpoint uniform, the other proportional to node capacity):
    python3 generate_payments.py -n 10000 --min-amount 1000000 --max-amount 50000000 \\
      --snapshot-dir Cloth/data/T1_snapshot --output payments.csv --amount-colname "amount(millisat)" --distinct-parties --seed 42
"""
import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic payments CSV.")
    p.add_argument("-n", "--num-payments", type=int, required=True, help="Number of payments (rows) to generate.")
    p.add_argument("--min-amount", type=int, required=True, help="Minimum amount for each payment (inclusive).")
    p.add_argument("--max-amount", type=int, required=True, help="Maximum amount for each payment (inclusive).")
    p.add_argument("--output", type=str, default="payments_generated.csv", help="Output CSV path.")
    p.add_argument("--amount-colname", type=str, default="amount", help='Amount column name. Use "amount(millisat)" to mirror your template.')
    p.add_argument("--id-start", type=int, default=0, help="Starting id (defaults to 0).")
    p.add_argument(
        "--nodes-file",
        type=str,
        default=None,
        help=(
            "Optional path to nodes_ln.csv (with header 'id'). If provided, sender_id/receiver_id will be sampled "
            "from the actual node IDs in this file (optionally filtered by --sender-range/--receiver-range)."
        ),
    )
    p.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help=(
            "Optional path to a cloth snapshot directory (nodes_ln.csv + channels_ln.csv). When set, node set and "
            "per-node capacities are loaded from it and sampling uses: 50%% sender uniform / receiver proportional "
            "to capacity, 50%% receiver uniform / sender proportional to capacity. Overrides --nodes-file."
        ),
    )
    p.add_argument("--sender-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=[1, 2000], help="Sender id range [MIN, MAX].")
    p.add_argument("--receiver-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=[1, 2000], help="Receiver id range [MIN, MAX].")
    p.add_argument("--distinct-parties", action="store_true", help="Ensure sender_id != receiver_id.")
    p.add_argument("--start-mode", choices=["relative", "epoch"], default="relative", help="How to generate start_time values.")
    p.add_argument("--start-base", type=int, default=0, help="Base start_time for relative mode (ignored for epoch mode).")
    p.add_argument("--step-min", type=int, default=10, help="Minimum step between successive start_time values (>=1).")
    p.add_argument("--step-max", type=int, default=60, help="Maximum step between successive start_time values (>= step-min).")
    p.add_argument("--epoch-base", type=int, default=None, help="Base UNIX epoch seconds for epoch mode; default ~now - N*avg_step.")
    p.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducibility.")
    return p.parse_args()

def _randint_inclusive(lo: int, hi: int) -> int:
    return random.randint(lo, hi)

def _pick_party_id(rng: Tuple[int, int]) -> int:
    lo, hi = rng
    return _randint_inclusive(lo, hi)

def _pick_from_ids(ids: Sequence[int]) -> int:
    return random.choice(ids)

def _make_distinct_pair(sr: Tuple[int, int], rr: Tuple[int, int]) -> Tuple[int, int]:
    """Return (sender_id, receiver_id) with sender != receiver if possible."""
    attempts = 0
    while True:
        sender = _pick_party_id(sr)
        receiver = _pick_party_id(rr)
        attempts += 1
        if sender != receiver or attempts > 2500:
            return sender, receiver

def _make_distinct_pair_from_ids(sender_ids: Sequence[int], receiver_ids: Sequence[int]) -> Tuple[int, int]:
    """Return (sender_id, receiver_id) with sender != receiver if possible."""
    attempts = 0
    while True:
        sender = _pick_from_ids(sender_ids)
        receiver = _pick_from_ids(receiver_ids)
        attempts += 1
        if sender != receiver or attempts > 2500:
            return sender, receiver

def read_node_ids(nodes_file: str) -> List[int]:
    """Read node ids from nodes_ln.csv (expects a header with column 'id')."""
    with open(nodes_file, "r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "id" not in r.fieldnames:
            raise ValueError(f"nodes-file must have column 'id' (got columns: {r.fieldnames})")
        ids: List[int] = []
        for row in r:
            v = row.get("id")
            if v is None or v == "":
                continue
            ids.append(int(v))
    # de-dup and sort (stable sampling still uniform via random.choice)
    return sorted(set(ids))


def load_snapshot_capacities(
    snapshot_dir: str,
    sender_range: Tuple[int, int],
    receiver_range: Tuple[int, int],
) -> Tuple[List[int], Dict[int, float]]:
    """
    Load node IDs and per-node capacity (half of channel capacity per channel) from a cloth snapshot directory.
    Expects nodes_ln.csv (id) and channels_ln.csv (node1_id, node2_id, capacity(millisat)).
    Returns (eligible_node_ids, node_capacity) with eligible set filtered by sender and receiver ranges (intersection).
    """
    base = Path(snapshot_dir)
    nodes_path = base / "nodes_ln.csv"
    channels_path = base / "channels_ln.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Snapshot nodes file not found: {nodes_path}")
    if not channels_path.exists():
        raise FileNotFoundError(f"Snapshot channels file not found: {channels_path}")

    node_ids_set = set()
    with open(nodes_path, "r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "id" not in (r.fieldnames or []):
            raise ValueError(f"nodes_ln.csv must have column 'id' (got {r.fieldnames})")
        for row in r:
            v = row.get("id")
            if v is not None and v != "":
                node_ids_set.add(int(v))

    node_capacity: Dict[int, float] = {}
    with open(channels_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n1 = int(row["node1_id"])
            n2 = int(row["node2_id"])
            cap = float(row["capacity(millisat)"])
            half = cap / 2.0
            node_capacity[n1] = node_capacity.get(n1, 0.0) + half
            node_capacity[n2] = node_capacity.get(n2, 0.0) + half
            node_ids_set.add(n1)
            node_ids_set.add(n2)

    node_ids = sorted(node_ids_set)
    sr_lo, sr_hi = sender_range
    rr_lo, rr_hi = receiver_range
    eligible = [n for n in node_ids if sr_lo <= n <= sr_hi and rr_lo <= n <= rr_hi]
    return eligible, node_capacity


def _build_weighted_lists(
    node_ids: Sequence[int],
    capacity_by_id: Dict[int, float],
) -> Tuple[List[int], List[float]]:
    """Build (ids, weights) once for nodes with positive capacity. Reuse for rejection sampling."""
    ids_w = [n for n in node_ids if (capacity_by_id.get(n) or 0) > 0]
    weights = [capacity_by_id[n] for n in ids_w]
    if not ids_w:
        raise ValueError("No nodes with positive capacity available for weighted choice")
    return ids_w, weights


def _pick_by_capacity(
    ids_w: List[int],
    weights: List[float],
    exclude: Optional[int],
    max_attempts: int = 2500,
) -> int:
    """Pick one node with probability proportional to weights; reject if chosen == exclude (rejection sampling)."""
    for _ in range(max_attempts):
        chosen = random.choices(ids_w, weights=weights, k=1)[0]
        if chosen != exclude:
            return chosen
    return ids_w[0]

def validate_args(a: argparse.Namespace) -> None:
    if a.num_payments <= 0:
        raise ValueError("num-payments must be > 0")
    if a.min_amount > a.max_amount:
        raise ValueError("min-amount must be <= max-amount")
    if a.step_min < 1 or a.step_max < a.step_min:
        raise ValueError("Require 1 <= step-min <= step-max")
    sr_lo, sr_hi = a.sender_range
    rr_lo, rr_hi = a.receiver_range
    if sr_lo > sr_hi or rr_lo > rr_hi:
        raise ValueError("sender/receiver ranges must be MIN <= MAX")

def _party_id_sources(a: argparse.Namespace) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Returns (sender_ids, receiver_ids) if using nodes-file, else (None, None).
    When nodes-file is used, sender/receiver ranges are applied as filters.
    """
    if not a.nodes_file:
        return None, None
    all_ids = read_node_ids(a.nodes_file)
    if not all_ids:
        raise ValueError("nodes-file contained no node IDs")

    sr_lo, sr_hi = a.sender_range
    rr_lo, rr_hi = a.receiver_range
    sender_ids = [i for i in all_ids if sr_lo <= i <= sr_hi]
    receiver_ids = [i for i in all_ids if rr_lo <= i <= rr_hi]
    if not sender_ids:
        raise ValueError(f"No sender IDs available after filtering nodes-file by sender-range [{sr_lo}, {sr_hi}]")
    if not receiver_ids:
        raise ValueError(f"No receiver IDs available after filtering nodes-file by receiver-range [{rr_lo}, {rr_hi}]")
    return sender_ids, receiver_ids

def build_rows(
    a: argparse.Namespace,
    sender_ids: Optional[Sequence[int]] = None,
    receiver_ids: Optional[Sequence[int]] = None,
    snapshot_node_ids: Optional[Sequence[int]] = None,
    node_capacity: Optional[Dict[int, float]] = None,
) -> List[dict]:
    rows = []
    # configure time stepping
    if a.start_mode == "relative":
        current_t = int(a.start_base)
        def next_time():
            nonlocal current_t
            step = _randint_inclusive(a.step_min, a.step_max)
            current_t += step
            return current_t
    else:
        # epoch mode
        avg_step = (a.step_min + a.step_max) // 2
        if a.epoch_base is None:
            base = int(time.time()) - a.num_payments * avg_step
        else:
            base = int(a.epoch_base)
        current_t = base
        def next_time():
            nonlocal current_t
            step = _randint_inclusive(a.step_min, a.step_max)
            current_t += step
            return current_t

    use_snapshot = snapshot_node_ids is not None and node_capacity is not None and len(snapshot_node_ids) >= 1
    eligible_list = list(snapshot_node_ids) if use_snapshot else None
    cap_map = node_capacity if use_snapshot else None
    # Precompute weighted list once for capacity-proportional sampling (rejection sampling for exclude)
    ids_w: Optional[List[int]] = None
    weights: Optional[List[float]] = None
    if use_snapshot and eligible_list and cap_map is not None:
        ids_w, weights = _build_weighted_lists(eligible_list, cap_map)

    for i in range(a.num_payments):
        if use_snapshot and eligible_list and ids_w is not None and weights is not None:
            # 50% sender uniform / receiver proportional; 50% receiver uniform / sender proportional; always distinct
            if random.random() < 0.5:
                sender = _pick_from_ids(eligible_list)
                receiver = _pick_by_capacity(ids_w, weights, exclude=sender)
            else:
                receiver = _pick_from_ids(eligible_list)
                sender = _pick_by_capacity(ids_w, weights, exclude=receiver)
            if a.distinct_parties and sender == receiver:
                # should not happen if _pick_by_capacity excludes the other; fallback swap one
                other = random.choice([n for n in eligible_list if n != sender])
                if random.random() < 0.5:
                    sender = other
                else:
                    receiver = other
        elif sender_ids is not None and receiver_ids is not None:
            if a.distinct_parties:
                sender, receiver = _make_distinct_pair_from_ids(sender_ids, receiver_ids)
            else:
                sender = _pick_from_ids(sender_ids)
                receiver = _pick_from_ids(receiver_ids)
        else:
            if a.distinct_parties:
                sender, receiver = _make_distinct_pair(tuple(a.sender_range), tuple(a.receiver_range))
            else:
                sender = _pick_party_id(tuple(a.sender_range))
                receiver = _pick_party_id(tuple(a.receiver_range))

        amount = _randint_inclusive(a.min_amount, a.max_amount)
        start_time_val = next_time()
        rows.append({
            "id": a.id_start + i,
            "sender_id": sender,
            "receiver_id": receiver,
            a.amount_colname: amount,
            "start_time": start_time_val,
        })
    return rows

def write_csv(path: str, rows: List[dict], amount_colname: str) -> None:
    fieldnames = ["id", "sender_id", "receiver_id", amount_colname, "start_time"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    a = parse_args()
    if a.seed is not None:
        random.seed(a.seed)
    validate_args(a)

    sender_ids: Optional[List[int]] = None
    receiver_ids: Optional[List[int]] = None
    snapshot_node_ids: Optional[List[int]] = None
    node_capacity: Optional[Dict[int, float]] = None

    if a.snapshot_dir:
        eligible, node_capacity = load_snapshot_capacities(
            a.snapshot_dir,
            tuple(a.sender_range),
            tuple(a.receiver_range),
        )
        if len(eligible) < 2:
            raise ValueError(
                f"Snapshot mode requires at least 2 eligible nodes (got {len(eligible)}). "
                "Check --sender-range and --receiver-range."
            )
        nodes_with_capacity = [n for n in eligible if (node_capacity.get(n) or 0) > 0]
        if len(nodes_with_capacity) < 2:
            raise ValueError(
                "Snapshot mode requires at least 2 nodes with positive capacity for proportional sampling."
            )
        snapshot_node_ids = eligible
    else:
        sender_ids, receiver_ids = _party_id_sources(a)

    rows = build_rows(
        a,
        sender_ids=sender_ids,
        receiver_ids=receiver_ids,
        snapshot_node_ids=snapshot_node_ids,
        node_capacity=node_capacity,
    )
    write_csv(a.output, rows, a.amount_colname)
    print(f"Wrote {len(rows)} rows to: {a.output}")

if __name__ == "__main__":
    main()
