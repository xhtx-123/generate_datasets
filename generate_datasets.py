#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM3 Paper Dataset Generator – Flexible File Count (D1, D2, D3)

Datasets:
  D1: Uniform distribution (scaled to keep 2GB total)
  D2: Normal (Gaussian) distribution (scaled to keep 2GB total)
  D3: Piecewise log‑uniform mixture (original D4 style)

All datasets: user-defined file count, fixed 2 GB total each.
Fully reproducible (seed = 42 by default).

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
    Efficiently adjust block counts so that sum(blocks) == total_blocks.
    Uses proportional scaling + round-robin remainder distribution.
    O(N) and works even when the initial sum is far from target.
    """
    n = len(blocks)
    # Ensure no zero blocks
    for i in range(n):
        if blocks[i] == 0:
            blocks[i] = 1

    current = sum(blocks)
    if current == total_blocks:
        return blocks

    # Scale proportionally
    scale = total_blocks / current
    new_blocks = [max(1, int(round(b * scale))) for b in blocks]

    # Adjust remainder
    current_new = sum(new_blocks)
    diff = total_blocks - current_new

    if diff > 0:
        idx = sorted(range(n), key=lambda i: new_blocks[i], reverse=True)
        for i in range(diff):
            new_blocks[idx[i % n]] += 1
    elif diff < 0:
        idx = sorted(range(n), key=lambda i: new_blocks[i], reverse=True)
        for i in range(-diff):
            if new_blocks[idx[i % n]] > 1:
                new_blocks[idx[i % n]] -= 1

    assert sum(new_blocks) == total_blocks, f"Sum={sum(new_blocks)}, target={total_blocks}"
    return new_blocks


def generate_sizes_d1(num_files, total_blocks, seed):
    """
    D1: Uniform distribution (scaled to match original D2 range).
    Blocks uniformly drawn from [mean/16, mean*1.94].
    This preserves the exact shape of the original D2 (CV ~ 0.58).
    """
    rng = random.Random(seed)
    mean = total_blocks / num_files
    low = max(1, int(mean / 16))
    high = max(low + 1, int(mean * 1.94))

    blocks = [rng.randint(low, high) for _ in range(num_files)]
    return adjust_to_total(blocks, total_blocks)


def generate_sizes_d2(num_files, total_blocks, seed):
    """
    D2: Normal (Gaussian) distribution.
    Blocks drawn from a truncated normal with mean = total_blocks/num_files,
    standard deviation = mean * 0.3 (CV ≈ 0.30).
    Values are clipped to [1, 2*mean] to avoid extremes.
    """
    rng = random.Random(seed)
    mean = total_blocks / num_files
    sigma = mean * 0.3   # CV ≈ 0.30

    blocks = []
    for _ in range(num_files):
        val = -1
        while val < 1:   # Re-sample if non-positive
            val = int(round(rng.gauss(mean, sigma)))
        # Clip to avoid absurdly large outliers (optional)
        if val > 2 * mean:
            val = int(2 * mean)
        blocks.append(val)

    return adjust_to_total(blocks, total_blocks)


def generate_sizes_d3(num_files, total_blocks, seed):
    """
    D3 (originally D4): Piecewise log‑uniform mixture.
    Segments: 0.15% large (scaled ~512–1024×mean), 20% medium (~2–256×mean), rest small.
    """
    rng = random.Random(seed)
    mean = total_blocks / num_files

    large_count = max(1, int(num_files * 0.0015))
    medium_count = int(num_files * 0.20)
    small_count = num_files - large_count - medium_count

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
        description="Generate SM3 datasets (D1: Uniform, D2: Normal, D3: Mixture) "
                    "with flexible file count, fixed 2GB total."
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
    print("SM3 Dataset Generator (D1: Uniform, D2: Normal, D3: Mixture)")
    print(f"  Files per dataset: {args.num_files:,}")
    print(f"  Target size per dataset: {TARGET_BYTES / (1024**3):.2f} GB")
    print(f"  Random seed: {args.seed}")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Estimated statistics:\n")
        for name in dataset_names:
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
