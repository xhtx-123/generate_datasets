#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM3 Paper Dataset Generator – Flexible File Count (D1, D2, D3)

Strictly maintains the distribution shapes of D1, D2, and D3 (D3 == original D4),
while fixing the TOTAL size of EACH dataset to exactly 2 GB.

The number of files is user-defined. Mean block size auto-scales to keep 2GB.

Usage Examples:
  # Generate all three datasets (D1, D2, D3) with 32,768 files each
  python generate_data.py 32768

  # Generate only D1 with 10,000 files
  python generate_data.py 10000 D1

  # Generate D1 and D3 with 50,000 files
  python generate_data.py 50000 D1,D3

  # Dry-run preview
  python generate_data.py 32768 --dry-run
"""

import os
import math
import json
import random
import argparse
import statistics
import time
from pathlib import Path

# ---------- Fixed parameters ----------
DEFAULT_SEED = 42
BLOCK_SIZE = 64
TARGET_BYTES = 2 * 1024 ** 3          # 2 GB per dataset
TOTAL_BLOCKS = TARGET_BYTES // BLOCK_SIZE
PATTERN = b'SM3_BENCHMARK_PATTERN_v1.0_'


def write_file(path, size_bytes):
    """Write a file of exactly `size_bytes` by repeating the fixed pattern."""
    with open(path, 'wb') as f:
        plen = len(PATTERN)
        written = 0
        while written < size_bytes:
            chunk = min(plen, size_bytes - written)
            f.write(PATTERN[:chunk])
            written += chunk


def adjust_to_total(blocks, total_blocks):
    """
    Adjust a list of block counts so that sum(blocks) == total_blocks.
    Files with 0 blocks are set to 1 first.
    """
    for i in range(len(blocks)):
        if blocks[i] == 0:
            blocks[i] = 1

    current = sum(blocks)
    if current == total_blocks:
        return blocks

    # Adjust upwards or downwards
    if current < total_blocks:
        diff = total_blocks - current
        sorted_idx = sorted(range(len(blocks)), key=lambda i: blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            blocks[idx] += 1
            diff -= 1
    else:  # current > total_blocks
        diff = current - total_blocks
        sorted_idx = sorted(range(len(blocks)), key=lambda i: blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            if blocks[idx] > 1:
                blocks[idx] -= 1
                diff -= 1

    assert sum(blocks) == total_blocks
    return blocks


def generate_sizes_d1(num_files, total_blocks, seed):
    """
    D1: Degenerate (fixed) distribution.
    Every file gets exactly (total_blocks / num_files) blocks.
    """
    rng = random.Random(seed)  # kept for consistency, but not used
    mean = total_blocks / num_files
    base = int(math.floor(mean))
    frac = mean - base

    blocks = [base] * num_files
    # Distribute the fractional remainder to some files
    remainder = total_blocks - sum(blocks)
    for i in range(remainder):
        blocks[i] += 1

    return blocks


def generate_sizes_d2(num_files, total_blocks, seed):
    """
    D2: Discrete Uniform distribution.
    Range: [mean/16, mean*1.94], which perfectly matches original D2
    (original mean=1024, low=64, high=1984).
    """
    rng = random.Random(seed)
    mean = total_blocks / num_files
    low = max(1, int(mean / 16))
    high = max(low + 1, int(mean * 1.94))

    blocks = [rng.randint(low, high) for _ in range(num_files)]
    return adjust_to_total(blocks, total_blocks)


def generate_sizes_d3(num_files, total_blocks, seed):
    """
    D3 (originally D4): Piecewise log‑uniform mixture.
    Segments: 0.15% large (32–64 MB scaled), 20% medium, rest small.
    """
    rng = random.Random(seed)
    mean = total_blocks / num_files

    large_count = max(1, int(num_files * 0.0015))
    medium_count = int(num_files * 0.20)
    small_count = num_files - large_count - medium_count

    # Scale the original block ranges by (mean / 1024)
    # Original large: 2^19 ~ 2^20, scaled -> mean * 512 ~ mean * 1024
    # Original medium: 2^11 ~ 2^18, scaled -> mean * 2 ~ mean * 256
    # Original small: 2^0 ~ 2^11, scaled -> 1 ~ mean * 2
    large_blocks = [
        rng.randint(int(mean * 512), int(mean * 1024))
        for _ in range(large_count)
    ]
    medium_blocks = [
        rng.randint(int(mean * 2), int(mean * 256))
        for _ in range(medium_count)
    ]
    small_blocks = [
        rng.randint(1, int(mean * 2))
        for _ in range(small_count)
    ]

    all_blocks = large_blocks + medium_blocks + small_blocks
    rng.shuffle(all_blocks)

    return adjust_to_total(all_blocks, total_blocks)


def generate_dataset(output_dir, name, num_files, gen_func, seed):
    """Generate a full dataset and save statistics."""
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    blocks = gen_func(num_files, TOTAL_BLOCKS, seed)
    sizes = [b * BLOCK_SIZE for b in blocks]

    start = time.time()
    for i, sz in enumerate(sizes):
        write_file(out_path / f"file_{i:06d}_{sz}B.bin", sz)
    elapsed = time.time() - start

    mean_b = statistics.mean(blocks)
    stdev_b = statistics.stdev(blocks) if len(blocks) > 1 else 0.0
    cv = stdev_b / mean_b if mean_b > 0 else 0.0

    stats = {
        "name": name,
        "num_files": num_files,
        "total_bytes": sum(sizes),
        "total_MB": sum(sizes) / (1024 * 1024),
        "total_GB": sum(sizes) / (1024 ** 3),
        "target_GB": TARGET_BYTES / (1024 ** 3),
        "mean_blocks": mean_b,
        "std_blocks": stdev_b,
        "cv": cv,
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "max_MB": max(sizes) / (1024 * 1024),
        "seed": seed,
        "gen_time_sec": elapsed,
    }

    with open(out_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate SM3 datasets (D1, D2, D3) with flexible file count, fixed 2GB total."
    )
    parser.add_argument(
        "num_files",
        type=int,
        help="Number of files to generate in EACH dataset"
    )
    parser.add_argument(
        "datasets",
        nargs='?',
        default="D1,D2,D3",
        help="Comma-separated dataset names (D1, D2, D3). Default: D1,D2,D3"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="./datasets",
        help="Root output directory (default: ./datasets)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview statistics without writing files"
    )
    args = parser.parse_args()

    # Parse dataset list
    if ',' in args.datasets:
        dataset_names = [x.strip() for x in args.datasets.split(',')]
    else:
        dataset_names = [args.datasets.strip()]

    valid_names = {"D1", "D2", "D3"}
    for name in dataset_names:
        if name not in valid_names:
            print(f"Error: invalid dataset '{name}'. Available: D1, D2, D3")
            return

    gen_map = {
        "D1": generate_sizes_d1,
        "D2": generate_sizes_d2,
        "D3": generate_sizes_d3,
    }

    print("=" * 80)
    print("SM3 Dataset Generator (Flexible Count, Fixed 2GB)")
    print(f"  Files per dataset: {args.num_files:,}")
    print(f"  Target size per dataset: {TARGET_BYTES / (1024**3):.2f} GB")
    print(f"  Random seed: {args.seed}")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Estimated statistics:\n")
        for name in dataset_names:
            # Generate a small sample to estimate mean/CV, or just calculate mathematically
            # For accurate preview, we generate the full list in memory but don't write files.
            blocks = gen_map[name](args.num_files, TOTAL_BLOCKS, args.seed)
            avg = statistics.mean(blocks)
            stdev = statistics.stdev(blocks) if len(blocks) > 1 else 0.0
            cv = stdev / avg if avg > 0 else 0.0
            max_mb = max(blocks) * BLOCK_SIZE / (1024 * 1024)
            print(f"  {name}: avg = {avg:.2f} blocks, CV = {cv:.4f}, max = {max_mb:.2f} MB")
        return

    all_stats = []
    total_start = time.time()

    for name in dataset_names:
        print(f"\nGenerating {name}...")
        stats = generate_dataset(
            output_dir=args.output,
            name=name,
            num_files=args.num_files,
            gen_func=gen_map[name],
            seed=args.seed
        )
        all_stats.append(stats)
        print(f"  Actual: {stats['total_GB']:.3f} GB, CV = {stats['cv']:.4f}, "
              f"max file = {stats['max_MB']:.2f} MB")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 90)
    print(f"Summary (all datasets have {args.num_files} files, fixed 2GB each)")
    print("=" * 90)
    print(f"{'Dataset':<10} {'Size(GB)':<12} {'Mean(blocks)':<14} {'CV':<12} {'Max(MB)':<12}")
    for s in all_stats:
        print(f"{s['name']:<10} {s['total_GB']:<12.3f} "
              f"{s['mean_blocks']:<14.2f} {s['cv']:<12.4f} {s['max_MB']:<12.2f}")

    print("\n" + "=" * 90)
    print(f"All datasets saved under: {Path(args.output).absolute()}")
    print(f"Total generation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
