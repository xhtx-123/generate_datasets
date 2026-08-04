#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM3 Paper Dataset Generator – Flexible File Count (D1, D2, D3)

Datasets (byte‑level generation, parameters directly comparable to literature):
  D1: Uniform – files uniformly distributed (CV ≈ 0.58)
  D2: Log‑Normal – σ = 1.2 (CV ≈ 1.57, within empirical range [1.4, 40])
  D3: Double Pareto – Laplace scale b = 0.5, yielding moderate heavy tail (CV ≈ 2.0)

All datasets: user‑defined file count, fixed total size (2.5 GB default).
Fully reproducible (seed = 42 by default).

Usage:
  python generate_data.py <num_files> [D1,D2,D3] [--seed N] [--output DIR] [--dry-run]
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
TARGET_BYTES = int(2.5 * 1024 ** 3)          # 2.5 GB per dataset
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


def adjust_to_total(sizes, total_bytes):
    """
    Efficiently adjust file sizes so that sum(sizes) == total_bytes.
    Uses proportional scaling + round‑robin remainder distribution.
    O(N) and works even when the initial sum is far from target.
    All values are integers (bytes).
    """
    n = len(sizes)
    # Ensure no zero‑byte files
    for i in range(n):
        if sizes[i] == 0:
            sizes[i] = 1

    current = sum(sizes)
    if current == total_bytes:
        return sizes

    # Scale proportionally
    scale = total_bytes / current
    new_sizes = [max(1, int(round(s * scale))) for s in sizes]

    # Adjust remainder
    current_new = sum(new_sizes)
    diff = total_bytes - current_new

    if diff > 0:
        idx = sorted(range(n), key=lambda i: new_sizes[i], reverse=True)
        for i in range(diff):
            new_sizes[idx[i % n]] += 1
    elif diff < 0:
        idx = sorted(range(n), key=lambda i: new_sizes[i], reverse=True)
        for i in range(-diff):
            if new_sizes[idx[i % n]] > 1:
                new_sizes[idx[i % n]] -= 1

    assert sum(new_sizes) == total_bytes, f"Sum={sum(new_sizes)}, target={total_bytes}"
    return new_sizes


# ---------- Distribution generators (byte‑level) ----------
def generate_sizes_d1(num_files, total_bytes, seed):
    """
    D1: Uniform distribution.
    Sizes uniformly drawn from [mean/16, mean*1.94], where mean = total_bytes / num_files.
    CV ≈ 0.58.
    """
    rng = random.Random(seed)
    mean = total_bytes / num_files
    low = max(1, int(mean / 16))
    high = max(low + 1, int(mean * 1.94))

    sizes = [rng.randint(low, high) for _ in range(num_files)]
    return adjust_to_total(sizes, total_bytes)


def generate_sizes_d2(num_files, total_bytes, seed):
    """
    D2: Log‑Normal distribution.
    sigma = 1.2  ->  CV ≈ sqrt(exp(1.44) - 1) ≈ 1.57.
    This falls within the empirical range (σ ≈ 1.4–40) reported in literature.
    The arithmetic mean is preserved as total_bytes / num_files.
    """
    rng = random.Random(seed)
    mean = total_bytes / num_files
    sigma = 1.2
    mu = math.log(mean) - (sigma ** 2) / 2.0

    sizes = []
    for _ in range(num_files):
        val = int(round(rng.lognormvariate(mu, sigma)))
        if val < 1:
            val = 1
        sizes.append(val)

    return adjust_to_total(sizes, total_bytes)


def generate_sizes_d3(num_files, total_bytes, seed):
    """
    D3: Double Pareto distribution (generated via Laplace).
    Let Y ~ Laplace(μ, b), then X = exp(Y) follows a Double Pareto.
    With b = 0.5 (Laplace scale), the tail is moderately heavy (CV ≈ 1.8–2.2).
    The arithmetic mean is preserved as total_bytes / num_files.

    Correct mean preservation for Double Pareto:
      E[exp(Y)] = exp(μ) / (1 - b^2)  for b < 1.
    Thus μ = log(mean * (1 - b^2)).
    """
    rng = random.Random(seed)
    mean = total_bytes / num_files
    b = 0.5
    # Correct formula for Double Pareto mean
    mu = math.log(mean * (1 - b ** 2))

    sizes = []
    for _ in range(num_files):
        # Laplace: Y = μ + b * (E1 - E2), where E1, E2 ~ Exp(1)
        e1 = rng.expovariate(1)
        e2 = rng.expovariate(1)
        Y = mu + b * (e1 - e2)
        val = int(round(math.exp(Y)))
        if val < 1:
            val = 1
        sizes.append(val)

    return adjust_to_total(sizes, total_bytes)


def generate_dataset(output_dir, name, num_files, gen_func, seed):
    """Generate a full dataset and save statistics."""
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    sizes = gen_func(num_files, TARGET_BYTES, seed)

    start = time.time()
    for i, sz in enumerate(sizes):
        write_file(out_path / f"file_{i:06d}_{sz}B.bin", sz)
    elapsed = time.time() - start

    mean_b = statistics.mean(sizes)
    stdev_b = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
    cv = stdev_b / mean_b if mean_b > 0 else 0.0

    stats = {
        "name": name,
        "num_files": num_files,
        "total_bytes": sum(sizes),
        "total_MB": sum(sizes) / (1024 * 1024),
        "total_GB": sum(sizes) / (1024 ** 3),
        "target_GB": TARGET_BYTES / (1024 ** 3),
        "mean_bytes": mean_b,
        "std_bytes": stdev_b,
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
        description="Generate SM3 datasets (D1: Uniform, D2: Log‑Normal, D3: Double Pareto) "
                    "with flexible file count, fixed total size (byte‑level generation)."
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
        help="Comma‑separated dataset names (D1, D2, D3). Default: D1,D2,D3"
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
    print("SM3 Dataset Generator (Byte‑level: D1 Uniform, D2 Log‑Normal, D3 Double Pareto)")
    print(f"  Files per dataset: {args.num_files:,}")
    print(f"  Target size per dataset: {TARGET_BYTES / (1024**3):.2f} GB")
    print(f"  Random seed: {args.seed}")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Estimated statistics (in bytes):\n")
        for name in dataset_names:
            sizes = gen_map[name](args.num_files, TARGET_BYTES, args.seed)
            avg = statistics.mean(sizes)
            stdev = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
            cv = stdev / avg if avg > 0 else 0.0
            max_mb = max(sizes) / (1024 * 1024)
            print(f"  {name}: avg = {avg:.2f} B, CV = {cv:.4f}, max = {max_mb:.2f} MB")
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
    print(f"Summary (all datasets have {args.num_files} files, fixed {TARGET_BYTES/(1024**3):.1f}GB each)")
    print("=" * 90)
    print(f"{'Dataset':<10} {'Size(GB)':<12} {'Mean(B)':<14} {'CV':<12} {'Max(MB)':<12}")
    for s in all_stats:
        print(f"{s['name']:<10} {s['total_GB']:<12.3f} "
              f"{s['mean_bytes']:<14.2f} {s['cv']:<12.4f} {s['max_MB']:<12.2f}")

    print("\n" + "=" * 90)
    print(f"All datasets saved under: {Path(args.output).absolute()}")
    print(f"Total generation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
