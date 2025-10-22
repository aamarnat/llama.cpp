#!/usr/bin/env python3
"""
Analyze ROCm kernel trace CSVs under a profiling directory.

Inputs:
  - Number of CUs in the system (integer)
  - Root folder path containing variant subfolders like: p2048_ub512_b512
    e.g. /home/aamarnat/projects/ROCr_AIE/llama.cpp_gold/prof_dir/20251001_150115

Behavior:
  - Discover immediate subfolders matching: p<p>_ub<ub>_b<b> (extract p, ub, b)
  - Find CSVs under each subfolder matching: */*_kernel_trace.csv
  - Parse each CSV and find the halfway point based on occurrences of a kernel name containing 'rms_norm_f32'
      * If there are N occurrences, the halfway point starts at the (N//2 + 1)-th occurrence (1-indexed).
        Equivalently, start from the row at zero-based occurrence index N//2, inclusive.
      * Only consider rows from that selected 'rms_norm_f32' entry to the end of the file.
  - For those rows, compute:
      1) time_us = (End_Timestamp - Start_Timestamp) / 1000
      2) Total_Workgroups = (Grid_Size_X / Workgroup_Size_X) *
                            (Grid_Size_Y / Workgroup_Size_Y) *
                            (Grid_Size_Z / Workgroup_Size_Z)
         (computed as float; assumes non-zero workgroup sizes)
      3) CU_Utilization = min(Total_Workgroups / Number_of_CUs, 1.0) * 100

Output:
  - For each subfolder containing *_kernel_trace.csv (e.g., bel-phx4), writes a file:
      <subfolder>/<variant_name>.csv
    Example:
      Input:  .../p2048_ub512_b512/bel-phx4/16880_kernel_trace.csv
      Output: .../p2048_ub512_b512/bel-phx4/p2048_ub512_b512.csv

Usage:
  python scripts/analyze_kernel_traces.py --cus 136 --root /path/to/prof_dir/20251001_150115
  python scripts/analyze_kernel_traces.py -c 136 -r /home/aamarnat/projects/ROCr_AIE/llama.cpp_gold/prof_dir/20251001_150115

Notes:
  - This script streams CSVs and performs two passes per file to avoid loading large files into memory.
  - If no 'rms_norm_f32' is found in a file, that file is skipped with a warning to stderr.
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, Iterator, List, Tuple


VARIANT_DIR_RE = re.compile(r"^p(?P<p>\d+)_ub(?P<ub>\d+)_b(?P<b>\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ROCm kernel traces with halfway point logic.")
    p.add_argument("-c", "--cus", type=int, required=True, help="Number of CUs in the system")
    p.add_argument("-r", "--root", type=str, required=True, help="Root folder with p*_ub*_b* subfolders")
    p.add_argument(
        "--match-substring",
        type=str,
        default="rms_norm_f32",
        help="Substring in Kernel_Name to identify occurrences for halfway point (default: rms_norm_f32)",
    )
    return p.parse_args()


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def iter_variant_dirs(root: str) -> Iterator[Tuple[str, int, int, int]]:
    """Yield (path, p, ub, b) for subdirectories matching p<p>_ub<ub>_b<b>."""
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        print(f"ERROR: root path not found: {root}", file=sys.stderr)
        return
    except Exception as e:
        print(f"ERROR: failed to list root '{root}': {e}", file=sys.stderr)
        return

    for name in sorted(entries):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        m = VARIANT_DIR_RE.match(name)
        if m:
            p = int(m.group("p"))
            ub = int(m.group("ub"))
            b = int(m.group("b"))
            yield full, p, ub, b


def count_occurrences(csv_path: str, kernel_col: str, match_substring: str) -> int:
    count = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or kernel_col not in reader.fieldnames:
            raise KeyError(f"Kernel column '{kernel_col}' not found in {csv_path}. Headers: {reader.fieldnames}")
        for row in reader:
            if match_substring in (row.get(kernel_col) or ""):
                count += 1
    return count


def process_rows_from_half(csv_path: str, start_occurrence_0based: int, kernel_col: str, match_substring: str) -> Iterator[Dict[str, str]]:
    """Yield rows (as dicts) from the selected occurrence onward."""
    occ_index = 0
    started = False
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if match_substring in (row.get(kernel_col) or ""):
                if occ_index == start_occurrence_0based:
                    started = True
                occ_index += 1
            if started:
                yield row


def compute_metrics(row: Dict[str, str], num_cus: int) -> Tuple[float, float, float]:
    """
    Returns (time_us, total_workgroups, cu_utilization_pct)
    """
    start_ts = safe_float(row.get("Start_Timestamp", "0"), 0.0)
    end_ts = safe_float(row.get("End_Timestamp", "0"), 0.0)
    time_us = (end_ts - start_ts) / 1000.0

    wsx = max(safe_float(row.get("Workgroup_Size_X", "0"), 0.0), 1.0)
    wsy = max(safe_float(row.get("Workgroup_Size_Y", "0"), 0.0), 1.0)
    wsz = max(safe_float(row.get("Workgroup_Size_Z", "0"), 0.0), 1.0)

    gsx = safe_float(row.get("Grid_Size_X", "0"), 0.0)
    gsy = safe_float(row.get("Grid_Size_Y", "0"), 0.0)
    gsz = safe_float(row.get("Grid_Size_Z", "0"), 0.0)

    total_workgroups = (gsx / wsx) * (gsy / wsy) * (gsz / wsz)

    if num_cus <= 0:
        cu_utilization = 0.0
    else:
        cu_utilization = min(total_workgroups / float(num_cus), 1.0) * 100.0

    return time_us, total_workgroups, cu_utilization


def find_csvs(variant_dir: str) -> List[str]:
    # Pattern: <variant_dir>/*/*_kernel_trace.csv
    pattern = os.path.join(variant_dir, "*", "*_kernel_trace.csv")
    files = sorted(glob.glob(pattern))
    return files


def group_csvs_by_subdir(csv_paths: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in csv_paths:
        subdir = os.path.dirname(path)
        groups.setdefault(subdir, []).append(path)
    return groups


def main() -> None:
    args = parse_args()
    root = args.root
    num_cus = args.cus
    match_substring = args.match_substring

    for vdir, p, ub, b in iter_variant_dirs(root):
        csv_paths = find_csvs(vdir)
        if not csv_paths:
            print(f"WARNING: no *_kernel_trace.csv under {vdir}", file=sys.stderr)
            continue

        groups = group_csvs_by_subdir(csv_paths)
        variant_name = os.path.basename(vdir)

        for subdir, files in groups.items():
            out_path = os.path.join(subdir, f"{variant_name}.csv")
            try:
                with open(out_path, "w", newline="") as out_f:
                    out_writer = csv.writer(out_f)
                    # Output CSV header
                    out_writer.writerow([
                        "variant_dir",
                        "p",
                        "ub",
                        "b",
                        "csv_path",
                        "Dispatch_Id",
                        "Kernel_Id",
                        "Kernel_Name",
                        "Start_Timestamp",
                        "End_Timestamp",
                        "time_us",
                        "Workgroup_Size_X",
                        "Workgroup_Size_Y",
                        "Workgroup_Size_Z",
                        "Grid_Size_X",
                        "Grid_Size_Y",
                        "Grid_Size_Z",
                        "Total_Workgroups",
                        "CU_Utilization_pct",
                    ])

                    for csv_path in files:
                        try:
                            # First pass: count occurrences of the kernel substring
                            total_occ = count_occurrences(csv_path, "Kernel_Name", match_substring)
                        except Exception as e:
                            print(f"ERROR: failed counting occurrences in {csv_path}: {e}", file=sys.stderr)
                            continue

                        if total_occ == 0:
                            print(f"WARNING: no occurrences of '{match_substring}' in {csv_path}; skipping", file=sys.stderr)
                            continue

                        start_occ_0 = total_occ // 2  # zero-based index to start from
                        try:
                            rows_iter = process_rows_from_half(csv_path, start_occ_0, "Kernel_Name", match_substring)
                            for row in rows_iter:
                                time_us, total_wg, cu_util_pct = compute_metrics(row, num_cus)
                                out_writer.writerow([
                                    vdir,
                                    p,
                                    ub,
                                    b,
                                    csv_path,
                                    row.get("Dispatch_Id", ""),
                                    row.get("Kernel_Id", ""),
                                    row.get("Kernel_Name", ""),
                                    row.get("Start_Timestamp", ""),
                                    row.get("End_Timestamp", ""),
                                    f"{time_us:.3f}",
                                    row.get("Workgroup_Size_X", ""),
                                    row.get("Workgroup_Size_Y", ""),
                                    row.get("Workgroup_Size_Z", ""),
                                    row.get("Grid_Size_X", ""),
                                    row.get("Grid_Size_Y", ""),
                                    row.get("Grid_Size_Z", ""),
                                    f"{total_wg:.6f}",
                                    f"{cu_util_pct:.2f}",
                                ])
                        except Exception as e:
                            print(f"ERROR: processing {csv_path}: {e}", file=sys.stderr)
                            continue
            except Exception as e:
                print(f"ERROR: cannot write output file '{out_path}': {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()
