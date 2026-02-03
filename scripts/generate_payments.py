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
"""
import argparse
import csv
import random
import time
from typing import List, Optional, Sequence, Tuple

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

def build_rows(a: argparse.Namespace, sender_ids: Optional[Sequence[int]] = None, receiver_ids: Optional[Sequence[int]] = None) -> List[dict]:
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

    # generate rows
    for i in range(a.num_payments):
        if sender_ids is not None and receiver_ids is not None:
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
    sender_ids, receiver_ids = _party_id_sources(a)
    rows = build_rows(a, sender_ids=sender_ids, receiver_ids=receiver_ids)
    write_csv(a.output, rows, a.amount_colname)
    print(f"Wrote {len(rows)} rows to: {a.output}")

if __name__ == "__main__":
    main()
