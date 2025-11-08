#!/usr/bin/env python3
# ln_snapshot_to_cloth_csvs.py
#
# Usage:
#   python ln_snapshot_to_cloth_csvs.py snapshot.json out/
# Options:
#   --allow-half-duplex   # keep single-direction channels by synthesizing the reverse edge

import argparse, json, csv, os, re, sys
from collections import defaultdict

def msat_to_int(x):
    if x is None: return None
    if isinstance(x, int): return x
    m = re.match(r"^(\d+)msat$", str(x))
    return int(m.group(1)) if m else None

def load_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list): return data
    if isinstance(data, dict) and isinstance(data.get("channels"), list): return data["channels"]
    raise ValueError("Snapshot must be a list or {'channels':[...]}")

# ---------- mappings (old ids) ----------
def build_node_mapping(records):
    pubkey_to_old = {}
    old_to_pubkey = {}
    nxt = 1
    for r in records:
        for k in ("source", "destination"):
            pk = r.get(k)
            if pk and pk not in pubkey_to_old:
                pubkey_to_old[pk] = nxt
                old_to_pubkey[nxt] = pk
                nxt += 1
    olds = sorted(old_to_pubkey.keys())
    node_old2new = {old:i for i, old in enumerate(olds)}  # 0..N-1
    return pubkey_to_old, node_old2new, old_to_pubkey

def build_scid_mapping(records):
    scid_to_old = {}
    nxt = 1
    for r in records:
        sc = r.get("short_channel_id")
        if sc and sc not in scid_to_old:
            scid_to_old[sc] = nxt
            nxt += 1
    return scid_to_old

# ---------- per-channel meta using old ids ----------
def collect_channel_meta(records, pubkey_to_old, scid_to_old):
    meta = {}  # scid -> info
    for r in records:
        sc = r.get("short_channel_id"); src = r.get("source"); dst = r.get("destination")
        if not sc or not src or not dst: continue
        a = pubkey_to_old.get(src); b = pubkey_to_old.get(dst)
        if not a or not b: continue
        cap_sat = int(r.get("satoshis") or 0)
        cap_msat = cap_sat * 1000
        n1, n2 = (a, b) if a <= b else (b, a)
        if sc not in meta:
            meta[sc] = {
                "old_channel_id": scid_to_old[sc],
                "node1_old": n1, "node2_old": n2,
                "capacity_msat": cap_msat
            }
        else:
            if meta[sc]["capacity_msat"] == 0 and cap_msat > 0:
                meta[sc]["capacity_msat"] = cap_msat
    return meta

# ---------- edge candidates ----------
def build_edge_candidates(records, pubkey_to_old, scid_to_old, ch_meta):
    cand = {}  # key: (old_ch, u_old, v_old) -> dict
    for r in records:
        sc = r.get("short_channel_id"); src = r.get("source"); dst = r.get("destination")
        if not sc or not src or not dst: continue
        old_ch = scid_to_old.get(sc)
        u = pubkey_to_old.get(src); v = pubkey_to_old.get(dst)
        if not old_ch or not u or not v: continue
        cap_msat = ch_meta.get(sc, {}).get("capacity_msat", 0)
        half_msat = cap_msat // 2 if cap_msat else 0
        cand[(old_ch, u, v)] = {
            "old_channel_id": old_ch,
            "from_old": u,
            "to_old": v,
            "balance_msat": int(half_msat),
            "fee_base_msat": int(r.get("base_fee_millisatoshi", 0) or 0),
            "fee_ppm": int(r.get("fee_per_millionth", 0) or 0),
            "min_htlc_msat": int(msat_to_int(r.get("htlc_minimum_msat")) or 0),
            "timelock": int(r.get("delay", 0) or 0),
        }
    return cand

def enforce_bidirectional(cand):
    keep, by_ch = {}, defaultdict(set)
    for (ch,u,v) in cand.keys():
        by_ch[ch].add((u,v))
    for k,e in cand.items():
        ch,u,v = k
        if (v,u) in by_ch[ch]:
            keep[k] = e
    return keep

def synthesize_missing_reverse_edges(cand):
    """
    For any (ch,u,v) missing its reverse (v,u), create it with:
      - balance = 0
      - policy copied from forward (best-effort fallback)
    """
    keys = list(cand.keys())
    for (ch,u,v) in keys:
        rev = (ch,v,u)
        if rev not in cand:
            fwd = cand[(ch,u,v)]
            cand[rev] = {
                "old_channel_id": ch,
                "from_old": v,
                "to_old": u,
                "balance_msat": 0,
                "fee_base_msat": fwd["fee_base_msat"],
                "fee_ppm": fwd["fee_ppm"],
                "min_htlc_msat": fwd["min_htlc_msat"],
                "timelock": fwd["timelock"],
            }
    return cand

# ---------- renumber to 0-based + rewrite ----------
def renumber_all(cand, ch_meta, node_old2new):
    # edges: ids 0..M-1
    keys = sorted(cand.keys())
    key_to_eid = {k:i for i,k in enumerate(keys)}
    edges = []
    for k in keys:
        old_ch, u_old, v_old = k
        e = cand[k]
        eid = key_to_eid[k]
        rev = (old_ch, v_old, u_old)
        counter = key_to_eid.get(rev, -1)  # after synthesize/enforce, this should exist
        edges.append({
            "id": eid,
            "old_channel_id": e["old_channel_id"],
            "from_new": node_old2new[e["from_old"]],
            "to_new":   node_old2new[e["to_old"]],
            "counter_edge_id": counter,
            "balance_msat": e["balance_msat"],
            "fee_base_msat": e["fee_base_msat"],
            "fee_ppm": e["fee_ppm"],
            "min_htlc_msat": e["min_htlc_msat"],
            "timelock": e["timelock"],
        })

    # channels: only those present in edges → ids 0..C-1
    present_old_ch = sorted({e["old_channel_id"] for e in edges})
    ch_old2new = {old:i for i,old in enumerate(present_old_ch)}

    # rewrite edges' channel_id
    for e in edges:
        e["channel_id"] = ch_old2new[e["old_channel_id"]]

    # channels meta (new ids, new node ids)
    channels_meta_new = {}
    old2scid = { m["old_channel_id"]: sc for sc,m in ch_meta.items() }
    for old in present_old_ch:
        cid = ch_old2new[old]
        sc = old2scid.get(old)
        m = ch_meta[sc]
        channels_meta_new[cid] = {
            "id": cid,
            "short_channel_id": sc,
            "node1_id": node_old2new[m["node1_old"]],
            "node2_id": node_old2new[m["node2_old"]],
            "capacity_msat": m["capacity_msat"],
        }

    # (new_ch, from_new, to_new) → edge_id
    key_new_to_eid = {(e["channel_id"], e["from_new"], e["to_new"]): e["id"] for e in edges}

    return edges, channels_meta_new, key_new_to_eid, ch_old2new

# ---------- writers ----------
def write_nodes_ln(outdir, node_old2new, old_to_pubkey):
    with open(os.path.join(outdir, "nodes_ln.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id"])
        for old in sorted(old_to_pubkey.keys()):
            w.writerow([node_old2new[old]])

def write_edges_ln(outdir, edges):
    fields = ["id","channel_id","counter_edge_id","from_node_id","to_node_id",
              "balance(millisat)","fee_base(millisat)","fee_proportional","min_htlc(millisat)","timelock"]
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
    fields = ["id","edge1_id","edge2_id","node1_id","node2_id","capacity(millisat)"]
    with open(os.path.join(outdir, "channels_ln.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(channels_meta_new.keys()):
            m = channels_meta_new[cid]
            n1, n2 = m["node1_id"], m["node2_id"]
            e1 = key_new_to_eid.get((cid, n1, n2))
            e2 = key_new_to_eid.get((cid, n2, n1))
            if not allow_half_duplex:
                # by construction e1 & e2 exist (enforced/synthesized earlier)
                if e1 is None or e2 is None:  # extremely defensive
                    continue
            else:
                # if user somehow disables synthesize, tolerate a missing edge with -1
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
    # id (new, 0-based) -> pubkey
    with open(os.path.join(outdir, "node_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","pubkey"])
        for old in sorted(old_to_pubkey.keys()):
            w.writerow([node_old2new[old], old_to_pubkey[old]])

def write_channel_mapping(outdir, ch_old2new, ch_meta):
    # id (new, 0-based) -> short_channel_id
    # ch_meta is scid-keyed; invert using the old id
    old2scid = { m["old_channel_id"]: sc for sc,m in ch_meta.items() }
    rows = []
    for old, new in ch_old2new.items():
        sc = old2scid.get(old)
        if sc:
            rows.append((new, sc))
    rows.sort(key=lambda x: x[0])
    with open(os.path.join(outdir, "channel_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","short_channel_id"])
        for new, sc in rows:
            w.writerow([new, sc])

# ---------- sanity ----------
def sanity_check(outdir):
    # edges
    edges = {}
    with open(os.path.join(outdir, "edges_ln.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["id"]); edges[i] = row
    M = len(edges); assert M>0, "no edges"
    assert set(edges.keys()) == set(range(0, M)), "edges.id must be contiguous 0..M-1"

    # channels
    chans = {}
    with open(os.path.join(outdir, "channels_ln.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["id"]); chans[cid] = row
    C = len(chans); assert C>0, "no channels"
    assert set(chans.keys()) == set(range(0, C)), "channels.id must be contiguous 0..C-1"

    # channel refs and edge channel_id bounds
    for cid,row in chans.items():
        e1 = int(row["edge1_id"]); e2 = int(row["edge2_id"])
        assert (-1 <= e1 < M) and (-1 <= e2 < M), f"Channel {cid} references missing edge"
        if e1 != -1:
            assert int(edges[e1]["channel_id"]) == cid
        if e2 != -1:
            assert int(edges[e2]["channel_id"]) == cid

    for e in edges.values():
        ch = int(e["channel_id"])
        assert 0 <= ch < C, f"Edge {e['id']} has channel_id {ch} out of 0..{C-1}"

    print("CSV sanity: OK")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Convert LN snapshot → *_ln.csv (0-based, contiguous IDs) + mappings")
    ap.add_argument("snapshot")
    ap.add_argument("outdir")
    ap.add_argument("--allow-half-duplex", action="store_true",
                    help="Keep single-direction channels (synthetic reverse edges with balance=0)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    recs = load_records(args.snapshot)

    # mappings & meta (old ids)
    pubkey_to_old, node_old2new, old_to_pubkey = build_node_mapping(recs)
    scid_to_old = build_scid_mapping(recs)
    ch_meta = collect_channel_meta(recs, pubkey_to_old, scid_to_old)

    # edges (old ids)
    cand = build_edge_candidates(recs, pubkey_to_old, scid_to_old, ch_meta)

    # handle duplex policy
    if args.allow_half_duplex:
        cand = synthesize_missing_reverse_edges(cand)
    else:
        cand = enforce_bidirectional(cand)

    # renumber to 0-based and rewrite
    edges, channels_meta_new, key_new_to_eid, ch_old2new = renumber_all(cand, ch_meta, node_old2new)

    # write *_ln + mapping files
    write_nodes_ln(args.outdir, node_old2new, old_to_pubkey)
    write_edges_ln(args.outdir, edges)
    write_channels_ln(args.outdir, channels_meta_new, key_new_to_eid, allow_half_duplex=args.allow_half_duplex)
    write_node_mapping(args.outdir, node_old2new, old_to_pubkey)
    write_channel_mapping(args.outdir, ch_old2new, ch_meta)

    # sanity
    sanity_check(args.outdir)

    print(f"Wrote to {args.outdir}: nodes_ln.csv, edges_ln.csv, channels_ln.csv, node_mapping.csv, channel_mapping.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
