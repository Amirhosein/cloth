#!/usr/bin/env python3
"""
Convert lncli snapshot (nodes + edges from lncli describegraph) to
Cloth simulator CSVs: nodes_ln.csv, edges_ln.csv, channels_ln.csv, plus mappings.

Usage:
  python clothify_lncli.py lightning_snapshot_2026-02-24.json data/
Options:
  --allow-half-duplex   Keep single-direction channels (synthetic reverse edge with default balance)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# Defaults compatible with Cloth/include/network.h
DEFAULT_FEE_BASE_MSAT = 1000
DEFAULT_FEE_PPM = 1
DEFAULT_MIN_HTLC_MSAT = 1000
DEFAULT_TIMELOCK = 10
DEFAULT_CAPACITY_MSAT = 10_000_000  # 0.01 BTC when only endpoints given


def msat_to_int(x):
    """Convert string like '1000msat' or int to int millisatoshis."""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        # Try parsing as plain number first
        try:
            return int(x)
        except ValueError:
            pass
        # Try parsing as "1000msat" format
        m = re.match(r"^(\d+)msat$", x)
        return int(m.group(1)) if m else None
    return None


def load_lncli(path):
    """Load lncli snapshot JSON and extract nodes and edges arrays."""
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
    return data, nodes, edges


def get_node_ids_lncli(nodes, edges):
    """Collect all node pubkeys from nodes array and edges (node1_pub, node2_pub)."""
    ids = set()
    # Collect from nodes array
    for n in nodes:
        if isinstance(n, dict):
            pub_key = n.get("pub_key")
            if pub_key:
                ids.add(pub_key)
    # Collect from edges
    for e in edges:
        if not isinstance(e, dict):
            continue
        node1_pub = e.get("node1_pub")
        node2_pub = e.get("node2_pub")
        if node1_pub:
            ids.add(node1_pub)
        if node2_pub:
            ids.add(node2_pub)
    return ids


def build_node_mapping_lncli(node_ids):
    """Return pubkey_to_old, node_old2new, old_to_pubkey (0-based new ids)."""
    sorted_ids = sorted(node_ids)
    pubkey_to_old = {pk: i + 1 for i, pk in enumerate(sorted_ids)}
    old_to_pubkey = {i + 1: pk for i, pk in enumerate(sorted_ids)}
    node_old2new = {old: i for i, old in enumerate(sorted(pubkey_to_old.values()))}
    return pubkey_to_old, node_old2new, old_to_pubkey


def extract_policy_fields(policy, direction_name=""):
    """Extract fee, timelock, min_htlc from a policy object. Return dict with defaults if policy is null."""
    if policy is None:
        return {
            "fee_base_msat": DEFAULT_FEE_BASE_MSAT,
            "fee_ppm": DEFAULT_FEE_PPM,
            "time_lock_delta": DEFAULT_TIMELOCK,
            "min_htlc": DEFAULT_MIN_HTLC_MSAT,
            "max_htlc_msat": None,
        }
    
    fee_base = DEFAULT_FEE_BASE_MSAT
    if policy.get("fee_base_msat"):
        try:
            fee_base = int(policy["fee_base_msat"])
        except (ValueError, TypeError):
            pass
    
    fee_ppm = DEFAULT_FEE_PPM
    if policy.get("fee_rate_milli_msat"):
        try:
            # fee_rate_milli_msat is ppm * 1000, so divide by 1000
            fee_rate_milli = int(policy["fee_rate_milli_msat"])
            fee_ppm = fee_rate_milli // 1000
            if fee_ppm == 0 and fee_rate_milli > 0:
                fee_ppm = 1  # Minimum 1 ppm
        except (ValueError, TypeError):
            pass
    
    time_lock_delta = DEFAULT_TIMELOCK
    if policy.get("time_lock_delta") is not None:
        try:
            time_lock_delta = int(policy["time_lock_delta"])
        except (ValueError, TypeError):
            pass
    
    min_htlc = DEFAULT_MIN_HTLC_MSAT
    if policy.get("min_htlc"):
        min_htlc_val = msat_to_int(policy["min_htlc"])
        if min_htlc_val is not None:
            min_htlc = min_htlc_val
    
    max_htlc_msat = None
    if policy.get("max_htlc_msat"):
        max_htlc_val = msat_to_int(policy["max_htlc_msat"])
        if max_htlc_val is not None:
            max_htlc_msat = max_htlc_val
    
    return {
        "fee_base_msat": fee_base,
        "fee_ppm": fee_ppm,
        "time_lock_delta": time_lock_delta,
        "min_htlc": min_htlc,
        "max_htlc_msat": max_htlc_msat,
    }


def build_candidates_and_meta_lncli(edges, pubkey_to_old, node_old2new):
    """
    Build edge candidates and channel meta from lncli edges.
    Uses max_htlc_msat from policies for channel capacity (since capacity field is typically "0").
    Extracts fees from node1_policy/node2_policy.
    """
    ch_meta = {}  # scid -> {old_channel_id, node1_old, node2_old, capacity_msat}
    scid_to_old = {}  # scid (channel_id string) -> old_channel_id
    cand = {}  # (old_ch, u_old, v_old) -> edge dict
    next_old_ch = 1
    
    for e in edges:
        if not isinstance(e, dict):
            continue
        
        channel_id = e.get("channel_id")
        node1_pub = e.get("node1_pub")
        node2_pub = e.get("node2_pub")
        
        if not channel_id or not node1_pub or not node2_pub:
            continue
        
        if node1_pub == node2_pub:
            continue
        
        u = pubkey_to_old.get(node1_pub)
        v = pubkey_to_old.get(node2_pub)
        if u is None or v is None:
            continue
        
        n1, n2 = (u, v) if u <= v else (v, u)
        scid = str(channel_id)  # Use channel_id as short_channel_id
        
        # Extract policies
        node1_policy = e.get("node1_policy")
        node2_policy = e.get("node2_policy")
        
        # Extract max_htlc_msat from both policies for channel capacity
        max_htlc_1 = None
        max_htlc_2 = None
        if node1_policy and node1_policy.get("max_htlc_msat"):
            max_htlc_1 = msat_to_int(node1_policy["max_htlc_msat"])
        if node2_policy and node2_policy.get("max_htlc_msat"):
            max_htlc_2 = msat_to_int(node2_policy["max_htlc_msat"])
        
        # Determine channel capacity: prefer non-zero capacity field, else use max of max_htlc_msat
        cap_msat = 0
        capacity_str = e.get("capacity", "0")
        try:
            cap_sat = int(capacity_str)
            cap_msat = cap_sat * 1000
        except (ValueError, TypeError):
            pass
        
        # If capacity is zero or missing, use max_htlc_msat
        if cap_msat == 0:
            max_htlc_values = [v for v in [max_htlc_1, max_htlc_2] if v is not None]
            if max_htlc_values:
                cap_msat = max(max_htlc_values)
            else:
                cap_msat = DEFAULT_CAPACITY_MSAT
        
        # Create or update channel metadata (use channel_id as unique key)
        if scid not in scid_to_old:
            old_ch = next_old_ch
            scid_to_old[scid] = old_ch
            ch_meta[scid] = {
                "old_channel_id": old_ch,
                "node1_old": n1,
                "node2_old": n2,
                "capacity_msat": cap_msat,
            }
            next_old_ch += 1
        else:
            # Update capacity if we found a larger max_htlc_msat
            old_ch = scid_to_old[scid]
            existing_cap = ch_meta[scid]["capacity_msat"]
            if cap_msat > existing_cap:
                ch_meta[scid]["capacity_msat"] = cap_msat
        
        old_ch = scid_to_old[scid]
        
        # Process edge node1 -> node2
        if (old_ch, u, v) not in cand:
            policy_fields = extract_policy_fields(node1_policy, "node1")
            # Get capacity from ch_meta using scid
            cap_msat_ch = ch_meta.get(scid, {}).get("capacity_msat", DEFAULT_CAPACITY_MSAT)
            half_msat = cap_msat_ch // 2 if cap_msat_ch else DEFAULT_CAPACITY_MSAT // 2
            
            cand[(old_ch, u, v)] = {
                "old_channel_id": old_ch,
                "from_old": u,
                "to_old": v,
                "balance_msat": int(half_msat),
                "fee_base_msat": policy_fields["fee_base_msat"],
                "fee_ppm": policy_fields["fee_ppm"],
                "min_htlc_msat": policy_fields["min_htlc"],
                "timelock": policy_fields["time_lock_delta"],
            }
        
        # Process edge node2 -> node1 (if not already processed)
        if (old_ch, v, u) not in cand:
            policy_fields = extract_policy_fields(node2_policy, "node2")
            # Get capacity from ch_meta using scid
            cap_msat_ch = ch_meta.get(scid, {}).get("capacity_msat", DEFAULT_CAPACITY_MSAT)
            half_msat = cap_msat_ch // 2 if cap_msat_ch else DEFAULT_CAPACITY_MSAT // 2
            
            cand[(old_ch, v, u)] = {
                "old_channel_id": old_ch,
                "from_old": v,
                "to_old": u,
                "balance_msat": int(half_msat),
                "fee_base_msat": policy_fields["fee_base_msat"],
                "fee_ppm": policy_fields["fee_ppm"],
                "min_htlc_msat": policy_fields["min_htlc"],
                "timelock": policy_fields["time_lock_delta"],
            }
    
    return cand, ch_meta


def synthesize_missing_reverse_edges(cand):
    """For any (ch, u, v) missing (ch, v, u), add it with default balance and copied policy."""
    keys = list(cand.keys())
    for (ch, u, v) in keys:
        rev = (ch, v, u)
        if rev not in cand:
            fwd = cand[(ch, u, v)]
            cand[rev] = {
                "old_channel_id": ch,
                "from_old": v,
                "to_old": u,
                "balance_msat": 100000,
                "fee_base_msat": fwd["fee_base_msat"],
                "fee_ppm": fwd["fee_ppm"],
                "min_htlc_msat": fwd["min_htlc_msat"],
                "timelock": fwd["timelock"],
            }
    return cand


def enforce_bidirectional(cand):
    """Keep only edges that have both directions present for the same channel."""
    by_ch = defaultdict(set)
    for (ch, u, v) in cand:
        by_ch[ch].add((u, v))
    keep = {}
    for k, e in cand.items():
        ch, u, v = k
        if (v, u) in by_ch[ch]:
            keep[k] = e
    return keep


def renumber_all_lncli(cand, ch_meta, node_old2new):
    """Renumber edges and channels to 0-based contiguous IDs. Same structure as renumber_all_t1."""
    keys = sorted(cand.keys())
    key_to_eid = {k: i for i, k in enumerate(keys)}
    edges = []
    for k in keys:
        old_ch, u_old, v_old = k
        e = cand[k]
        eid = key_to_eid[k]
        rev = (old_ch, v_old, u_old)
        counter = key_to_eid.get(rev, -1)
        edges.append({
            "id": eid,
            "old_channel_id": e["old_channel_id"],
            "from_new": node_old2new[e["from_old"]],
            "to_new": node_old2new[e["to_old"]],
            "counter_edge_id": counter,
            "balance_msat": e["balance_msat"],
            "fee_base_msat": e["fee_base_msat"],
            "fee_ppm": e["fee_ppm"],
            "min_htlc_msat": e["min_htlc_msat"],
            "timelock": e["timelock"],
        })
    present_old_ch = sorted({e["old_channel_id"] for e in edges})
    ch_old2new = {old: i for i, old in enumerate(present_old_ch)}
    for e in edges:
        e["channel_id"] = ch_old2new[e["old_channel_id"]]
    old2scid = {m["old_channel_id"]: sc for sc, m in ch_meta.items()}
    channels_meta_new = {}
    for old in present_old_ch:
        sc = old2scid.get(old)
        if sc is None:
            continue
        m = ch_meta[sc]
        cid = ch_old2new[old]
        channels_meta_new[cid] = {
            "id": cid,
            "short_channel_id": sc,
            "node1_id": node_old2new[m["node1_old"]],
            "node2_id": node_old2new[m["node2_old"]],
            "capacity_msat": m["capacity_msat"],
        }
    key_new_to_eid = {(e["channel_id"], e["from_new"], e["to_new"]): e["id"] for e in edges}
    return edges, channels_meta_new, key_new_to_eid, ch_old2new


# Reuse writers and sanity_check from clothify_t1
def write_nodes_ln(outdir, node_old2new, old_to_pubkey):
    with open(os.path.join(outdir, "nodes_ln.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for old in sorted(old_to_pubkey.keys()):
            w.writerow([node_old2new[old]])


def write_edges_ln(outdir, edges):
    fields = [
        "id", "channel_id", "counter_edge_id", "from_node_id", "to_node_id",
        "balance(millisat)", "fee_base(millisat)", "fee_proportional", "min_htlc(millisat)", "timelock",
    ]
    with open(os.path.join(outdir, "edges_ln.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in sorted(edges, key=lambda x: x["id"]):
            w.writerow({
                "id": e["id"],
                "channel_id": e["channel_id"],
                "counter_edge_id": e["counter_edge_id"],
                "from_node_id": e["from_new"],
                "to_node_id": e["to_new"],
                "balance(millisat)": e["balance_msat"],
                "fee_base(millisat)": e["fee_base_msat"],
                "fee_proportional": e["fee_ppm"],
                "min_htlc(millisat)": e["min_htlc_msat"],
                "timelock": e["timelock"],
            })


def write_channels_ln(outdir, channels_meta_new, key_new_to_eid, allow_half_duplex):
    fields = ["id", "edge1_id", "edge2_id", "node1_id", "node2_id", "capacity(millisat)"]
    with open(os.path.join(outdir, "channels_ln.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(channels_meta_new.keys()):
            m = channels_meta_new[cid]
            n1, n2 = m["node1_id"], m["node2_id"]
            e1 = key_new_to_eid.get((cid, n1, n2))
            e2 = key_new_to_eid.get((cid, n2, n1))
            if not allow_half_duplex:
                if e1 is None or e2 is None:
                    continue
            else:
                e1 = -1 if e1 is None else e1
                e2 = -1 if e2 is None else e2
            w.writerow({
                "id": cid,
                "edge1_id": e1,
                "edge2_id": e2,
                "node1_id": n1,
                "node2_id": n2,
                "capacity(millisat)": m["capacity_msat"],
            })


def write_node_mapping(outdir, node_old2new, old_to_pubkey):
    with open(os.path.join(outdir, "node_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pubkey"])
        for old in sorted(old_to_pubkey.keys()):
            w.writerow([node_old2new[old], old_to_pubkey[old]])


def write_channel_mapping(outdir, ch_old2new, ch_meta):
    old2scid = {m["old_channel_id"]: sc for sc, m in ch_meta.items()}
    rows = [(ch_old2new[old], old2scid[old]) for old in ch_old2new if old in old2scid]
    rows.sort(key=lambda x: x[0])
    with open(os.path.join(outdir, "channel_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "short_channel_id"])
        for new, sc in rows:
            w.writerow([new, sc])


def sanity_check(outdir):
    edges = {}
    with open(os.path.join(outdir, "edges_ln.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["id"])
            edges[i] = row
    M = len(edges)
    assert M > 0, "no edges"
    assert set(edges.keys()) == set(range(0, M)), "edges.id must be contiguous 0..M-1"
    chans = {}
    with open(os.path.join(outdir, "channels_ln.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["id"])
            chans[cid] = row
    C = len(chans)
    assert C > 0, "no channels"
    assert set(chans.keys()) == set(range(0, C)), "channels.id must be contiguous 0..C-1"
    for cid, row in chans.items():
        e1 = int(row["edge1_id"])
        e2 = int(row["edge2_id"])
        assert (-1 <= e1 < M) and (-1 <= e2 < M), f"Channel {cid} references missing edge"
        if e1 != -1:
            assert int(edges[e1]["channel_id"]) == cid
        if e2 != -1:
            assert int(edges[e2]["channel_id"]) == cid
    for e in edges.values():
        ch = int(e["channel_id"])
        assert 0 <= ch < C, f"Edge {e['id']} has channel_id {ch} out of 0..{C-1}"
    print("CSV sanity: OK")


def main():
    ap = argparse.ArgumentParser(
        description="Convert lncli snapshot (nodes + edges) → *_ln.csv for Cloth simulator",
    )
    ap.add_argument("snapshot", help="Path to lightning_snapshot_YYYY-MM-DD.json")
    ap.add_argument("outdir", help="Output directory for CSVs")
    ap.add_argument(
        "--allow-half-duplex",
        action="store_true",
        help="Keep single-direction channels (synthetic reverse edge with default balance)",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data, nodes, edges = load_lncli(args.snapshot)
    node_ids = get_node_ids_lncli(nodes, edges)
    if not node_ids:
        raise ValueError("No nodes found in nodes array or edges")
    pubkey_to_old, node_old2new, old_to_pubkey = build_node_mapping_lncli(node_ids)
    cand, ch_meta = build_candidates_and_meta_lncli(edges, pubkey_to_old, node_old2new)
    if not cand:
        raise ValueError("No valid edges found in edges array")
    if not args.allow_half_duplex:
        cand = synthesize_missing_reverse_edges(cand)
    else:
        cand = enforce_bidirectional(cand)
    if not cand:
        raise ValueError("No edges remaining after bidirectional filter")
    edges_result, channels_meta_new, key_new_to_eid, ch_old2new = renumber_all_lncli(cand, ch_meta, node_old2new)

    write_nodes_ln(args.outdir, node_old2new, old_to_pubkey)
    write_edges_ln(args.outdir, edges_result)
    write_channels_ln(args.outdir, channels_meta_new, key_new_to_eid, allow_half_duplex=args.allow_half_duplex)
    write_node_mapping(args.outdir, node_old2new, old_to_pubkey)
    write_channel_mapping(args.outdir, ch_old2new, ch_meta)
    sanity_check(args.outdir)
    print(f"Wrote to {args.outdir}: nodes_ln.csv, edges_ln.csv, channels_ln.csv, node_mapping.csv, channel_mapping.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
