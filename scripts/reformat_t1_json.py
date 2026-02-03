#!/usr/bin/env python3
"""
Reformat T1_snapshot.json from single-line minified JSON to pretty-printed
multi-line JSON so it is readable and works with tools that choke on long lines.

Usage:
  python reformat_t1_json.py [input.json [output.json]]
  Default: reads Cloth/T1_snapshot.json, writes Cloth/T1_snapshot_pretty.json

To fix the file in place then run clothify_t1, you can instead use:
  python clothify_t1.py --reformat Cloth/T1_snapshot.json data/
"""

import json
import os
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(script_dir)
    default_input = os.path.join(base, "T1_snapshot.json")
    default_output = os.path.join(base, "T1_snapshot_pretty.json")

    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        output_path = default_output
    else:
        input_path = default_input
        output_path = default_output

    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path} ...", file=sys.stderr)
    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Writing {output_path} (indent=2) ...", file=sys.stderr)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done. Use with clothify_t1: python clothify_t1.py {output_path} <outdir>", file=sys.stderr)


if __name__ == "__main__":
    main()
