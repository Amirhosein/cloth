#!/bin/bash
# Run convert_edges.py and adversary_simulation.py for a simulation directory
#
# Usage:
#   ./run_adversary_analysis.sh <simulation_dir> <data_dir>
#
# Example:
#   ./run_adversary_analysis.sh ./results/simulation8_T5_1_5_random ./data/T5

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <simulation_dir> <data_dir>"
    echo ""
    echo "Arguments:"
    echo "  simulation_dir  Path to simulation results directory (contains outpayments_output.csv, outedges_output.csv)"
    echo "  data_dir        Path to data directory (contains channels_ln.csv, nodes_ln.csv)"
    echo ""
    echo "Example:"
    echo "  $0 ./results/simulation8_T5_1_5_random ./data/T5"
    exit 1
fi

SIM_DIR="$1"
DATA_DIR="$2"

# Resolve to absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR_ABS="$(cd "$SIM_DIR" && pwd)"
DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"

# Input files
PAYMENTS_INPUT="${SIM_DIR_ABS}/outpayments_output.csv"
EDGES_INPUT="${SIM_DIR_ABS}/outedges_output.csv"
CHANNELS_FILE="${DATA_DIR_ABS}/channels_ln.csv"
NODES_FILE="${DATA_DIR_ABS}/nodes_ln.csv"

# Output files
PAYMENTS_NODE_OUTPUT="${SIM_DIR_ABS}/outpayments_output_node.csv"

# Verify input files exist
for file in "$PAYMENTS_INPUT" "$EDGES_INPUT" "$CHANNELS_FILE" "$NODES_FILE"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

echo "Step 1: Converting payment routes from edge IDs to node IDs..."
python3 "${SCRIPT_DIR}/convert_edges.py" \
    --payments "$PAYMENTS_INPUT" \
    --edges "$EDGES_INPUT" \
    --output "$PAYMENTS_NODE_OUTPUT"

if [ ! -f "$PAYMENTS_NODE_OUTPUT" ]; then
    echo "Error: convert_edges.py did not produce output file: $PAYMENTS_NODE_OUTPUT"
    exit 1
fi

echo ""
echo "Step 2: Running adversary simulation..."
python3 "${SCRIPT_DIR}/adversary_simulation.py" \
    --payments "$PAYMENTS_NODE_OUTPUT" \
    --channels "$CHANNELS_FILE" \
    --nodes "$NODES_FILE" \
    --output-dir "$SIM_DIR_ABS" --logs

echo ""
echo "Analysis complete! Results written to: $SIM_DIR_ABS"
