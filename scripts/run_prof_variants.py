#!/usr/bin/env python3
"""
Run rocprofv3 sys-trace across combinations of p, ub, and b.

- p values drawn from {2048, 4096, 8192}
- ub and b values drawn from {512, 1024, 2048, 4096, 8192}
- Constraints: b == p, ub == p
- Output directory naming follows: ./prof_dir/p{p}_ub{ub}_b{b}_{YYYYMMDD_HHMMSS}
- Logs are streamed to stdout and saved to {out_dir}/log.txt via tee
- Paths are relative to running from the repository root

Usage:
  python3 scripts/run_prof_variants.py

Notes:
  - Expects:
      ./build_hip/bin/llama-bench
      ../models/Llama-3.1-8B-Instruct-BF16.gguf
    Adjust MODEL or BIN if needed.
"""
import os
import itertools
import shutil
import subprocess
import datetime

# Parameter values
P_VALUES = [2048, 4096, 8192]
UB_B_VALUES = [2048, 4096, 8192]
# UB_B_VALUES = [512, 1024, 2048, 4096, 8192]

# Paths (adjust if your layout differs)
BASE_DIR = "./prof_dir"
MODEL = "../models/Llama-3.1-8B-Instruct-BF16.gguf"
TYPE_VAR = ""  # set to "graph" for graph, "" for empty string
NO_TRACE = False  # set to True to skip rocprofv3 and use 5 repetitions

# Adjust RUN_TS suffix based on TYPE_VAR
if TYPE_VAR == "graph":
    RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_graph")
    BIN = "./build_graph/bin/llama-bench"
else:
    RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BIN = "./build/bin/llama-bench"

def run_combo(p: int, ub: int, b: int) -> None:
    """Run a single combination, optionally without rocprofv3."""
    out_dir = os.path.join(BASE_DIR, f"{RUN_TS}/p{p}_ub{ub}_b{b}")
    os.makedirs(out_dir, exist_ok=True)

    repetitions = 5 if NO_TRACE else 1

    if NO_TRACE:
        cmd = (
            f"{BIN} -m {MODEL} -r {repetitions} -n 0 -p {p} -ub {ub} -b {b} "
            f"| tee {out_dir}/log.txt"
        )
    else:
        cmd = (
            f"rocprofv3 --output-format csv --sys-trace -d {out_dir} --truncate-kernels -- "
            f"{BIN} -m {MODEL} -r {repetitions} -n 0 -p {p} -ub {ub} -b {b} "
            f"| tee {out_dir}/log.txt"
        )

    print(f"\n=== Running p={p}, ub={ub}, b={b} ===")
    print(cmd)
    # Use shell=True to support the `| tee` pipeline
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed (exit {result.returncode}) for p={p}, ub={ub}, b={b}")

def main() -> None:
    # Basic pre-flight checks
    if not os.path.exists(BIN):
        print(f"ERROR: Binary not found at {BIN}")
        return
    if not os.path.exists(MODEL):
        print(f"ERROR: Model file not found at {MODEL}")
        return
    if not NO_TRACE and shutil.which("rocprofv3") is None:
        print("ERROR: 'rocprofv3' not found in PATH")
        return

    # Iterate combinations; enforce constraint b <= p
    for p, ub, b in itertools.product(P_VALUES, UB_B_VALUES, UB_B_VALUES):
        if b > p or ub > p or ub > b or ub != p or b != ub:
            continue
        run_combo(p, ub, b)

if __name__ == "__main__":
    main()

