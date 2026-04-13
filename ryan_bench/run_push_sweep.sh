#!/usr/bin/env bash
# Sweep over input (prompt) lengths for push_gpu_buffer_benchmark.py.
# Fixed knobs match the original defaults; only --prompt-length varies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$SCRIPT_DIR/push_gpu_buffer_benchmark.py"
LOG_DIR="$SCRIPT_DIR/logs/push_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Fixed parameters
NUM_PROMPTS=10
OUTPUT_LEN=64
BUFFER_SIZE=$((160000 * (256 + 1)))
QPS=12.0

# Sweep values
PROMPT_LENGTHS=(30 60 90 120)

echo "Logs will be written to: $LOG_DIR"

for PROMPT_LENGTH in "${PROMPT_LENGTHS[@]}"; do
    LOG_FILE="$LOG_DIR/prompt_len_${PROMPT_LENGTH}.log"
    echo "========================================"
    echo "Running: prompt_length=${PROMPT_LENGTH}"
    echo "Log: $LOG_FILE"
    echo "========================================"

    "${PYTHON:-python3}" "$BENCHMARK" \
        --num-prompts "$NUM_PROMPTS" \
        --prompt-length "$PROMPT_LENGTH" \
        --output-len "$OUTPUT_LEN" \
        --buffer-size "$BUFFER_SIZE" \
        --qps "$QPS" \
        2>&1 | tee "$LOG_FILE"

    echo "Done: prompt_length=${PROMPT_LENGTH}"
    echo ""
done

echo "All runs complete. Logs saved to: $LOG_DIR"
