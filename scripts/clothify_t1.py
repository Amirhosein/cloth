#!/usr/bin/env python3
"""
Convert T1_snapshot.json (directed graph with nodes + adjacencies/edges) to
Cloth simulator CSVs: nodes_ln.csv, edges_ln.csv, channels_ln.csv, plus mappings.

Usage:
  python clothify_t1.py T1_snapshot.json data/
  python clothify_t1.py --reformat T1_snapshot.json data/   # fix single-line JSON first
Options:
  --allow-half-duplex   Keep single-direction channels (synthetic reverse edge with default balance)
  --reformat            If the snapshot is minified (one long line), rewrite it pretty in place, then run
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
    if x is None:
        return None
    if isinstance(x, int):
        return x
    m = re.match(r"^(\d+)msat$", str(x))
    return int(m.group(1)) if m else None


def is_minified_json(path):
    """Return True if the file appears to be single-line/minified JSON."""
    with open(path, "r") as f:
        first_line = f.readline()
        second = f.readline()
    return bool(first_line) and not second and len(first_line) > 500


def reformat_t1_json_in_place(path):
    """Rewrite path with pretty-printed JSON (indent=2). Use when file is single-line."""
    with open(path, "r") as f:
        data = json.load(f)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _flatten_adjacency(adjacency):
    """Convert adjacency list-of-lists (adjacency[i] = edges from node i) to a flat list of edge dicts."""
    flat = []
    for out_edges in adjacency:
        if not isinstance(out_edges, list):
            continue
        for e in out_edges:
            if isinstance(e, dict):
                flat.append(e)
    return flat


def load_t1(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("T1 snapshot must be a JSON object")
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("T1 snapshot must have 'nodes' array")
    # Support both flat lists (adjacencies/edges/links) and list-of-lists ("adjacency" = per-node outgoing edges)
    edge_list = data.get("adjacencies") or data.get("edges") or data.get("links")
    if edge_list is None and data.get("adjacency") is not None:
        edge_list = _flatten_adjacency(data["adjacency"])
    if not isinstance(edge_list, list):
        raise ValueError("T1 snapshot must have 'adjacency', 'adjacencies', 'edges', or 'links'")
    return data, nodes, edge_list


def get_node_ids(nodes, edge_list):
    """Collect all node IDs from nodes array and edge list (source/destination or source/target)."""
    ids = set()
    for n in nodes:
        pid = n.get("id") if isinstance(n, dict) else n
        if pid is not None:
            ids.add(pid)
    for r in edge_list:
        if not isinstance(r, dict):
            continue
        for key in ("source", "destination", "target", "from"):
            v = r.get(key)
            if v is not None:
                ids.add(v)
    return ids


def is_channel_like(record):
    """True if record has source and destination (channel-like)."""
    return isinstance(record, dict) and record.get("source") and record.get("destination")


def build_node_mapping_t1(node_ids):
    """Return pubkey_to_old, node_old2new, old_to_pubkey (0-based new ids)."""
    sorted_ids = sorted(node_ids)
    pubkey_to_old = {pk: i + 1 for i, pk in enumerate(sorted_ids)}
    old_to_pubkey = {i + 1: pk for i, pk in enumerate(sorted_ids)}
    node_old2new = {old: i for i, old in enumerate(sorted(pubkey_to_old.values()))}
    return pubkey_to_old, node_old2new, old_to_pubkey


def build_candidates_and_meta_t1(edge_list, pubkey_to_old, node_old2new):
    """
    Build edge candidates and channel meta from T1 adjacencies.
    Each record can be channel-like (source, destination, satoshis, fees, ...) or minimal (source, target).
    Deduplicate by directed edge (u, v). Use synthetic short_channel_id for channels.
    """
    ch_meta = {}  # scid -> {old_channel_id, node1_old, node2_old, capacity_msat}
    scid_to_old = {}
    cand = {}  # (old_ch, u_old, v_old) -> edge dict
    channel_key_to_old_ch = {}  # (n1_old, n2_old) -> old_ch
    next_old_ch = 1

    for r in edge_list:
        if not isinstance(r, dict):
            continue
        src = r.get("source") or r.get("from")
        dst = r.get("destination") or r.get("target")
        if src == dst:
            continue
        if not src or not dst:
            continue
        u = pubkey_to_old.get(src)
        v = pubkey_to_old.get(dst)
        if u is None or v is None:
            continue
        n1, n2 = (u, v) if u <= v else (v, u)
        ch_key = (n1, n2)
        if ch_key not in channel_key_to_old_ch:
            channel_key_to_old_ch[ch_key] = next_old_ch
            scid = f"t1_{next_old_ch}"
            scid_to_old[scid] = next_old_ch
            cap_sat = int(r.get("satoshis") or 0) if is_channel_like(r) else 0
            cap_msat = cap_sat * 1000 if cap_sat else 0
            if not cap_msat and r.get("htlc_maximum_msat"):
                cap_msat = int(msat_to_int(r.get("htlc_maximum_msat")) or 0) * 2
            if not cap_msat:
                cap_msat = DEFAULT_CAPACITY_MSAT
            ch_meta[scid] = {
                "old_channel_id": next_old_ch,
                "node1_old": n1,
                "node2_old": n2,
                "capacity_msat": cap_msat,
            }
            next_old_ch += 1
        old_ch = channel_key_to_old_ch[ch_key]
        if (old_ch, u, v) in cand:
            continue  # deduplicate
        cap_msat = ch_meta[f"t1_{old_ch}"]["capacity_msat"]
        half_msat = cap_msat // 2 if cap_msat else DEFAULT_CAPACITY_MSAT // 2
        fee_base = int(r.get("base_fee_millisatoshi") or r.get("fee_base_msat") or 0) or DEFAULT_FEE_BASE_MSAT
        fee_ppm = int(r.get("fee_per_millionth") or r.get("fee_proportional_millionths") or 0) or DEFAULT_FEE_PPM
        min_htlc = int(msat_to_int(r.get("htlc_minimum_msat") or r.get("htlc_minimim_msat")) or 0) or DEFAULT_MIN_HTLC_MSAT
        timelock = int(r.get("delay") or r.get("cltv_expiry_delta") or 0) or DEFAULT_TIMELOCK
        if is_channel_like(r):
            cand[(old_ch, u, v)] = {
                "old_channel_id": old_ch,
                "from_old": u,
                "to_old": v,
                "balance_msat": int(half_msat),
                "fee_base_msat": fee_base,
                "fee_ppm": fee_ppm,
                "min_htlc_msat": min_htlc,
                "timelock": timelock,
            }
        else:
            cand[(old_ch, u, v)] = {
                "old_channel_id": old_ch,
                "from_old": u,
                "to_old": v,
                "balance_msat": int(half_msat),
                "fee_base_msat": fee_base,
                "fee_ppm": fee_ppm,
                "min_htlc_msat": min_htlc,
                "timelock": timelock,
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


def renumber_all_t1(cand, ch_meta, node_old2new):
    """Same structure as clothify renumber_all: edges, channels_meta_new, key_new_to_eid, ch_old2new."""
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


# Reuse writers and sanity_check from clothify
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
        description="Convert T1 snapshot (nodes + adjacencies/edges) → *_ln.csv for Cloth simulator",
    )
    ap.add_argument("snapshot", help="Path to T1_snapshot.json")
    ap.add_argument("outdir", help="Output directory for CSVs")
    ap.add_argument(
        "--allow-half-duplex",
        action="store_true",
        help="Keep single-direction channels (synthetic reverse edge with default balance)",
    )
    ap.add_argument(
        "--reformat",
        action="store_true",
        help="If the snapshot is single-line/minified JSON, rewrite it pretty (indent=2) in place first.",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.reformat and is_minified_json(args.snapshot):
        print("Reformatting snapshot to pretty JSON in place ...", file=sys.stderr)
        reformat_t1_json_in_place(args.snapshot)

    data, nodes, edge_list = load_t1(args.snapshot)
    node_ids = get_node_ids(nodes, edge_list)
    if not node_ids:
        raise ValueError("No nodes found in nodes array or edge list")
    pubkey_to_old, node_old2new, old_to_pubkey = build_node_mapping_t1(node_ids)
    cand, ch_meta = build_candidates_and_meta_t1(edge_list, pubkey_to_old, node_old2new)
    if not cand:
        raise ValueError("No valid edges found in adjacencies/edges/links")
    if not args.allow_half_duplex:
        cand = synthesize_missing_reverse_edges(cand)
    else:
        cand = enforce_bidirectional(cand)
    if not cand:
        raise ValueError("No edges remaining after bidirectional filter")
    edges, channels_meta_new, key_new_to_eid, ch_old2new = renumber_all_t1(cand, ch_meta, node_old2new)

    write_nodes_ln(args.outdir, node_old2new, old_to_pubkey)
    write_edges_ln(args.outdir, edges)
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
