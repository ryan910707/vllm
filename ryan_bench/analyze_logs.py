#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Analyze benchmark log files produced by push_gpu_buffer_benchmark.py.

Usage:
    python analyze_logs.py <log_file_or_folder> [log_file_or_folder ...]

For each log file, reports:
  - Prefill compute time  : total prefill generation loop duration
  - Prefill wait time     : sum of all "Waited Xs for buffer space" messages
  - TTFT per request      : timestamp of i-th "Drop-selected KV cache"
                            minus timestamp of "Decode model initialized"
  - Throughput            : total output tokens / decode total time  (tokens/s)
"""

import re
import sys
import os
from datetime import datetime
from pathlib import Path

TS_FMT = "%Y-%m-%d %H:%M:%S.%f"

def parse_ts(line: str) -> datetime | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", line)
    if m:
        return datetime.strptime(m.group(1), TS_FMT)
    return None


def analyze(path: str) -> dict:
    with open(path) as f:
        lines = f.readlines()

    result = {"file": os.path.basename(path)}

    # --- config from benchmark header ---
    num_prompts = output_len = prompt_length = None
    for line in lines:
        m = re.search(r"CONFIG num_prompts=(\d+) prompt_length=(\d+) output_len=(\d+)", line)
        if m:
            num_prompts = int(m.group(1))
            prompt_length = int(m.group(2))
            output_len = int(m.group(3))
            break

    result["num_prompts"] = num_prompts
    result["prompt_length"] = prompt_length
    result["output_len"] = output_len

    # --- prefill compute time ---
    prefill_compute = None
    for line in lines:
        m = re.search(r"Prefill generation loop completed in ([\d.]+)s", line)
        if m:
            prefill_compute = float(m.group(1))
            break
    result["prefill_compute_s"] = prefill_compute

    # --- prefill wait time (sum all "Waited Xs for buffer space") ---
    wait_times = []
    for line in lines:
        m = re.search(r"Waited ([\d.]+)s for buffer space", line)
        if m:
            wait_times.append(float(m.group(1)))
    result["prefill_wait_s"] = sum(wait_times) if wait_times else 0.0
    result["prefill_wait_count"] = len(wait_times)

    # --- TTFT per request ---
    # TTFT[i] = timestamp of i-th "Drop-selected KV cache from buffer"
    #         - timestamp of "Decode model initialized"
    decode_init_ts = None
    drop_select_ts: list[datetime] = []

    for line in lines:
        ts = parse_ts(line)
        if ts is None:
            # drop_select logs use a different timestamp format (INFO 04-11 HH:MM:SS.mmm)
            m2 = re.search(r"INFO (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)", line)
            if m2 and "Drop-selected KV cache from buffer" in line:
                time_str = m2.group(3)
                month = int(m2.group(1))
                day = int(m2.group(2))
                t = datetime.strptime(time_str, "%H:%M:%S.%f")
                if decode_init_ts:
                    t = t.replace(year=decode_init_ts.year, month=month, day=day)
                drop_select_ts.append(t)
            continue
        if "Decode model initialized" in line:
            decode_init_ts = ts
            continue
        if "Drop-selected KV cache from buffer" in line:
            drop_select_ts.append(ts)

    ttfts = []
    if decode_init_ts and drop_select_ts:
        for ds_ts in drop_select_ts:
            ttfts.append((ds_ts - decode_init_ts).total_seconds())

    result["ttft_per_req_s"] = ttfts
    result["ttft_avg_s"] = sum(ttfts) / len(ttfts) if ttfts else None
    result["ttft_min_s"] = min(ttfts) if ttfts else None
    result["ttft_max_s"] = max(ttfts) if ttfts else None

    # --- decode total time ---
    decode_total = None
    for line in lines:
        m = re.search(r"Decode total time: ([\d.]+)s", line)
        if m:
            decode_total = float(m.group(1))
            break
    result["decode_total_s"] = decode_total

    # --- throughput: total output tokens / decode total time ---
    if decode_total and num_prompts is not None and output_len is not None and decode_total > 0:
        total_tokens = num_prompts * output_len
        result["throughput_tok_s"] = total_tokens / decode_total
        result["total_output_tokens"] = total_tokens
    else:
        result["throughput_tok_s"] = None
        result["total_output_tokens"] = None

    return result


def fmt(v, unit="", decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}{unit}"


def print_report(r: dict):
    sep = "=" * 60
    print(sep)
    print(f"File            : {r['file']}")
    print(f"Config          : {r['num_prompts']} prompts, "
          f"prompt_len={r['prompt_length']}, output_len={r['output_len']}")
    print(sep)
    print(f"Prefill compute : {fmt(r['prefill_compute_s'], 's')}")
    print(f"Prefill wait    : {fmt(r['prefill_wait_s'], 's')}  "
          f"({r['prefill_wait_count']} wait events)")
    print(sep)
    ttfts = r["ttft_per_req_s"]
    if ttfts:
        print(f"TTFT (per req)  :")
        for i, t in enumerate(ttfts):
            print(f"  req {i:>2d}        : {t:.3f}s")
        print(f"TTFT avg        : {fmt(r['ttft_avg_s'], 's')}")
        print(f"TTFT min        : {fmt(r['ttft_min_s'], 's')}")
        print(f"TTFT max        : {fmt(r['ttft_max_s'], 's')}")
    else:
        print("TTFT            : N/A (no decode timestamps found)")
    print(sep)
    print(f"Decode total    : {fmt(r['decode_total_s'], 's')}")
    print(f"Total out tokens: {r['total_output_tokens'] or 'N/A'}")
    print(f"Throughput      : {fmt(r['throughput_tok_s'], ' tok/s')}  "
          f"({r['total_output_tokens'] or '?'} tokens / {fmt(r['decode_total_s'], 's')})")
    print(sep)
    print()


def collect_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(Path(p).glob("*.log")):
                files.append(str(f))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file_or_folder> [...]")
        sys.exit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No log files found.")
        sys.exit(1)

    for f in files:
        r = analyze(f)
        print_report(r)
