#!/usr/bin/env python3
# ln_snapshot_to_cloth_csvs.py
#
# Usage:
#   python ln_snapshot_to_cloth_csvs.py snapshot.json out/
# Optional:
#   --allow-half-duplex

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

# ---------- Mapping builders ----------
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
    # compact to 0-based contiguous ids
    olds = sorted(old_to_pubkey.keys())
    node_old2new = {old:i for i, old in enumerate(olds)}       # 0..N-1
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

# ---------- Channel meta (endpoints, capacity) using old ids ----------
def collect_channel_meta(records, pubkey_to_old, scid_to_old):
    meta = {}  # key: scid str
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
                "old_channel_id": scid_to_old[sc],  # provisional old numeric
                "node1_old": n1, "node2_old": n2,
                "capacity_msat": cap_msat
            }
        else:
            if meta[sc]["capacity_msat"] == 0 and cap_msat > 0:
                meta[sc]["capacity_msat"] = cap_msat
    return meta

# ---------- Build per-direction edge candidates keyed by (old_ch, old_u, old_v) ----------
def build_edge_candidates(records, pubkey_to_old, scid_to_old, ch_meta):
    cand = {}
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

def filter_bidirectional(cand, allow_half_duplex=False):
    if allow_half_duplex: return cand
    keep, by_ch = {}, defaultdict(set)
    for (ch,u,v) in cand.keys():
        by_ch[ch].add((u,v))
    for k,e in cand.items():
        ch,u,v = k
        if (v,u) in by_ch[ch]:
            keep[k] = e
    return keep

# ---------- Renumber everything to 0-based contiguous and rewrite refs ----------
def renumber_all(cand, ch_meta, node_old2new):
    # 1) edges → assign ids 0..M-1 and record reverse ids
    keys = sorted(cand.keys())
    key_to_eid = {k:i for i,k in enumerate(keys)}          # 0..M-1
    edges = []
    for k in keys:
        old_ch, u_old, v_old = k
        e = cand[k]
        eid = key_to_eid[k]
        rev = (old_ch, v_old, u_old)
        counter = key_to_eid.get(rev, -1)                  # -1 only if half-duplex allowed
        edges.append({
            "id": eid,
            "old_channel_id": e["old_channel_id"],
            "from_old": u_old,
            "to_old": v_old,
            "counter_edge_id": counter,
            "balance_msat": e["balance_msat"],
            "fee_base_msat": e["fee_base_msat"],
            "fee_ppm": e["fee_ppm"],
            "min_htlc_msat": e["min_htlc_msat"],
            "timelock": e["timelock"],
        })

    # 2) channels → compact to 0..C-1 using only channels present in edges
    present_old_ch = sorted({e["old_channel_id"] for e in edges})
    ch_old2new = {old:i for i,old in enumerate(present_old_ch)}  # 0..C-1

    # 3) rewrite edges: channel_id (new), node ids (new)
    for e in edges:
        e["channel_id"] = ch_old2new[e["old_channel_id"]]
        e["from_new"] = node_old2new[e["from_old"]]
        e["to_new"]   = node_old2new[e["to_old"]]

    # 4) build channels rows meta with new ids + new node ids
    channels_meta_new = {}
    for sc, m in ch_meta.items():
        old = m["old_channel_id"]
        if old not in ch_old2new: continue
        cid = ch_old2new[old]
        channels_meta_new[cid] = {
            "id": cid,
            "node1_id": node_old2new[m["node1_old"]],
            "node2_id": node_old2new[m["node2_old"]],
            "capacity_msat": m["capacity_msat"],
        }

    # 5) rebuild (new_ch, from_new, to_new) → edge_id (for channel edge pairing)
    key_new_to_eid = {(e["channel_id"], e["from_new"], e["to_new"]): e["id"] for e in edges}

    return edges, channels_meta_new, key_new_to_eid

# ---------- Writers ----------
def write_nodes(outdir, node_old2new, old_to_pubkey):
    # nodes_ln.csv — id only (0..N-1)
    path = os.path.join(outdir, "nodes_ln.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id"])
        for old in sorted(old_to_pubkey.keys()):
            w.writerow([node_old2new[old]])

def write_edges(outdir, edges):
    # edges_ln.csv — 0-based ids
    path = os.path.join(outdir, "edges_ln.csv")
    fields = ["id","channel_id","counter_edge_id","from_node_id","to_node_id",
              "balance(millisat)","fee_base(millisat)","fee_proportional","min_htlc(millisat)","timelock"]
    edges_sorted = sorted(edges, key=lambda e: e["id"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in edges_sorted:
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

def write_channels(outdir, channels_meta_new, key_new_to_eid):
    # channels_ln.csv — 0-based ids; require bidirectional (edges exist both ways)
    path = os.path.join(outdir, "channels_ln.csv")
    fields = ["id","edge1_id","edge2_id","node1_id","node2_id","capacity(millisat)"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(channels_meta_new.keys()):
            m = channels_meta_new[cid]
            n1, n2 = m["node1_id"], m["node2_id"]
            e1 = key_new_to_eid.get((cid, n1, n2))
            e2 = key_new_to_eid.get((cid, n2, n1))
            if e1 is None or e2 is None:
                # should not happen when bidirectional filter is enforced
                continue
            w.writerow({
                "id": cid,
                "edge1_id": e1,
                "edge2_id": e2,
                "node1_id": n1,
                "node2_id": n2,
                "capacity(millisat)": m["capacity_msat"],
            })

# ---------- Sanity checks ----------
def sanity_check(outdir):
    import itertools
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
    # refs
    for cid,row in chans.items():
        e1 = int(row["edge1_id"]); e2 = int(row["edge2_id"])
        assert 0 <= e1 < M and 0 <= e2 < M, f"Channel {cid} references missing edge"
        a,b = edges[e1], edges[e2]
        assert int(a["channel_id"]) == int(b["channel_id"]) == cid, f"Edges must carry channel_id={cid}"
        assert a["from_node_id"] == b["to_node_id"] and a["to_node_id"] == b["from_node_id"], "Edges must be reverse"
    # channel_id range on edges
    for e in edges.values():
        ch = int(e["channel_id"])
        assert 0 <= ch < C, f"Edge {e['id']} has channel_id {ch} out of 0..{C-1}"
    print("CSV sanity: OK")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Convert LN snapshot → *_ln.csv (zero-based, contiguous IDs)")
    ap.add_argument("snapshot")
    ap.add_argument("outdir")
    ap.add_argument("--allow-half-duplex", action="store_true",
                    help="Keep single-direction channels (NOT recommended for this C code)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    recs = load_records(args.snapshot)

    # 1) mappings (old ids)
    pubkey_to_old, node_old2new, old_to_pubkey = build_node_mapping(recs)
    scid_to_old = build_scid_mapping(recs)

    # 2) channel meta (old ids)
    ch_meta = collect_channel_meta(recs, pubkey_to_old, scid_to_old)

    # 3) edge candidates (old ids)
    cand = build_edge_candidates(recs, pubkey_to_old, scid_to_old, ch_meta)

    # 4) enforce bidirectional unless flag says otherwise
    cand = filter_bidirectional(cand, allow_half_duplex=args.allow_half_duplex)

    # 5) renumber to 0-based contiguous & rewrite references
    edges, channels_meta_new, key_new_to_eid = renumber_all(cand, ch_meta, node_old2new)

    # 6) write files
    write_nodes(args.outdir, node_old2new, old_to_pubkey)
    write_edges(args.outdir, edges)
    write_channels(args.outdir, channels_meta_new, key_new_to_eid)

    # 7) sanity checks
    sanity_check(args.outdir)

    print(f"Wrote: nodes_ln.csv, edges_ln.csv, channels_ln.csv in {args.outdir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
