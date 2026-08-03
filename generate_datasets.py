#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM3 Paper Dataset Generator – On‑demand Edition

Supported datasets:
  D1: Fixed 64 KB, CV = 0.00
  D2: Uniform 64 B – 64 KB, CV ≈ 0.58
  D4: Directly constructed mixed distribution (64 B – 64 MB, max file > 30 MB)
  D5: Heavy‑tail random: 99.75% tiny + 0.25% 256 MB, CV > 6.5

All datasets: 32,768 files, total size ≈ 2 GB (adjustable)
Fixed seed = 42, fully reproducible.

Usage:
  # Generate all datasets
  python generate_datasets.py --output ./data --seed 42

  # Generate only D4
  python generate_datasets.py --output ./data --datasets D4

  # Preview D4 without writing
  python generate_datasets.py --dry-run --datasets D4
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
NUM_FILES = 32768
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


def generate_sizes_general(num_files, total_blocks, dist_func, seed):
    """
    Generic weight‑scaling method (used for D1, D2, D5).

    Generates a list of block counts whose sum equals `total_blocks`.
    Each file gets a weight drawn from `dist_func`; block counts are
    allocated proportionally to these weights and then adjusted to meet
    the exact total.
    """
    rng = random.Random(seed)
    weights = [dist_func(rng, i) for i in range(num_files)]
    total_weight = sum(weights)
    if total_weight == 0:
        return [1] * num_files

    raw = [w * total_blocks / total_weight for w in weights]
    blocks = [int(math.floor(b)) for b in raw]
    fracs = [b - math.floor(b) for b in raw]

    current = sum(blocks)
    remainder = total_blocks - current

    # Distribute remaining blocks based on fractional parts
    if remainder > 0:
        indices = sorted(range(num_files), key=lambda i: fracs[i], reverse=True)
        for idx in indices[:remainder]:
            blocks[idx] += 1

    # Ensure no zero‑block files
    for i in range(num_files):
        if blocks[i] == 0:
            blocks[i] = 1

    # Final adjustment to match total_blocks exactly
    current = sum(blocks)
    if current > total_blocks:
        diff = current - total_blocks
        sorted_idx = sorted(range(num_files), key=lambda i: blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            if blocks[idx] > 1:
                blocks[idx] -= 1
                diff -= 1
    elif current < total_blocks:
        diff = total_blocks - current
        sorted_idx = sorted(range(num_files), key=lambda i: blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            blocks[idx] += 1
            diff -= 1

    assert sum(blocks) == total_blocks
    return blocks


def generate_sizes_d4(num_files, total_blocks, seed):
    """
    Special generator for D4 that directly constructs block counts.

    Goals: produce a multi‑segment mixture with several files up to
    tens of MB, while keeping the total size close to the target.

    Design:
      - 50 large files (≈ 0.15%): each 32–64 MB
      - 20% medium files: 128 KB – 16 MB
      - Remaining files: 64 B – 128 KB (log‑uniform)
    The list is shuffled, then scaled and adjusted to hit `total_blocks`.
    """
    rng = random.Random(seed)

    large_count = 50
    medium_count = int(num_files * 0.20)
    small_count = num_files - large_count - medium_count

    # Large: 2^19–2^20 blocks (32–64 MB)
    large_blocks = [2 ** rng.randint(19, 20) for _ in range(large_count)]

    # Medium: 2^11–2^18 blocks (128 KB – 16 MB)
    medium_blocks = [2 ** rng.randint(11, 18) for _ in range(medium_count)]

    # Small: 2^0–2^11 blocks (64 B – 128 KB)
    small_blocks = [2 ** rng.randint(0, 11) for _ in range(small_count)]

    all_blocks = large_blocks + medium_blocks + small_blocks
    rng.shuffle(all_blocks)

    current = sum(all_blocks)
    scale = total_blocks / current
    scaled_blocks = [max(1, int(b * scale)) for b in all_blocks]

    # Adjust to exact total
    current = sum(scaled_blocks)
    diff = total_blocks - current
    if diff > 0:
        sorted_idx = sorted(range(num_files), key=lambda i: scaled_blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            scaled_blocks[idx] += 1
            diff -= 1
    elif diff < 0:
        sorted_idx = sorted(range(num_files), key=lambda i: scaled_blocks[i], reverse=True)
        for idx in sorted_idx:
            if diff == 0:
                break
            if scaled_blocks[idx] > 1:
                scaled_blocks[idx] -= 1
                diff += 1

    assert sum(scaled_blocks) == total_blocks
    return scaled_blocks


def generate_dataset(output_dir, name, num_files, total_bytes, dist_func, seed):
    """
    Generate a complete dataset.

    Creates a subdirectory under `output_dir`, writes all files,
    and saves a `stats.json` with summary statistics.
    """
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)
    total_blocks = total_bytes // BLOCK_SIZE

    if name.startswith("D4"):
        blocks = generate_sizes_d4(num_files, total_blocks, seed)
    else:
        blocks = generate_sizes_general(num_files, total_blocks, dist_func, seed)

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


# ---------- Distribution functions (weight generators) ----------
def d1_fixed(rng, idx):
    """D1: always 1024 blocks = 64 KB."""
    return 1024


def d2_uniform_small(rng, idx):
    """D2: uniform between 64 and 1984 blocks (≈ 4 KB – 124 KB)."""
    return rng.randint(64, 1984)


def d4_mixed(rng, idx):
    """
    Placeholder for D4 – not used because D4 uses a special direct constructor.
    Kept only for mapping consistency.
    """
    return 1024


def d5_heavy_tail(rng, idx):
    """
    D5: heavy‑tail random.
    99.75% of files are tiny (1–64 blocks), 0.25% are huge (4M blocks = 256 MB).
    """
    if rng.random() < 0.9975:
        return rng.randint(1, 64)
    else:
        return 4 * 1024 * 1024   # 256 MB


# ---------- Main entry point ----------
def main():
    parser = argparse.ArgumentParser(
        description="SM3 Dataset Generator (on‑demand)"
    )
    parser.add_argument("--output", default="./datasets",
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (for reproducibility)")
    parser.add_argument("--target-size", default="2GB",
                        help="Target total size for each dataset (e.g., 2GB, 1500MB)")
    parser.add_argument("--datasets", default="D1,D2,D4,D5",
                        help="Comma‑separated list of datasets to generate (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only preview statistics, do not write any files")
    args = parser.parse_args()

    # Parse target size
    s = args.target_size.upper()
    if "GB" in s:
        target_bytes = int(float(s.replace("GB", "").strip()) * 1024 ** 3)
    elif "MB" in s:
        target_bytes = int(float(s.replace("MB", "").strip()) * 1024 ** 2)
    else:
        target_bytes = int(float(s) * 1024 ** 2)
    target_bytes = (target_bytes // BLOCK_SIZE) * BLOCK_SIZE

    requested = [x.strip() for x in args.datasets.split(',') if x.strip()]
    dist_map = {
        "D1": d1_fixed,
        "D2": d2_uniform_small,
        "D4": d4_mixed,   # not used directly, but kept for key existence
        "D5": d5_heavy_tail,
    }

    for name in requested:
        if name not in dist_map:
            print(f"Error: unknown dataset '{name}'. Available: {', '.join(dist_map.keys())}")
            return

    configs = [(name, dist_map[name]) for name in requested]

    print("=" * 75)
    print("SM3 Paper Dataset Generator")
    print(f"  Files per dataset: {NUM_FILES:,}")
    print(f"  Target size: {target_bytes / (1024 ** 3):.2f} GB")
    print(f"  Random seed: {args.seed}")
    print(f"  Datasets to generate: {', '.join(requested)}")
    if "D4" in requested:
        print("  D4 mode: direct construction (50 large files 32‑64 MB, "
              "20% medium, rest small)")
    print("=" * 75)

    if args.dry_run:
        print("\n[DRY RUN] Estimated statistics per dataset:\n")
        for name, func in configs:
            if name == "D4":
                # Simulate D4 without writing
                rng = random.Random(args.seed)
                large_count = 50
                medium_count = int(NUM_FILES * 0.20)
                small_count = NUM_FILES - large_count - medium_count
                large_blocks = [2 ** rng.randint(19, 20) for _ in range(large_count)]
                medium_blocks = [2 ** rng.randint(11, 18) for _ in range(medium_count)]
                small_blocks = [2 ** rng.randint(0, 11) for _ in range(small_count)]
                all_blocks = large_blocks + medium_blocks + small_blocks
                avg = statistics.mean(all_blocks)
                stdev = statistics.stdev(all_blocks)
                cv_est = stdev / avg if avg > 0 else 0
                est_gb = avg * NUM_FILES * BLOCK_SIZE / (1024 ** 3)
                max_mb = max(all_blocks) * BLOCK_SIZE / (1024 * 1024)
                print(f"  D4: avg = {avg:.1f} blocks, CV ≈ {cv_est:.4f}, "
                      f"estimated {est_gb:.3f} GB, max = {max_mb:.1f} MB")
            else:
                rng = random.Random(args.seed)
                weights = [func(rng, i) for i in range(10000)]
                avg = statistics.mean(weights)
                stdev = statistics.stdev(weights)
                cv_est = stdev / avg if avg > 0 else 0
                est_total_blocks = avg * NUM_FILES
                est_gb = est_total_blocks * BLOCK_SIZE / (1024 ** 3)
                max_mb = max(weights) * BLOCK_SIZE / (1024 * 1024)
                print(f"  {name}: avg = {avg:.1f} blocks, CV ≈ {cv_est:.4f}, "
                      f"estimated {est_gb:.3f} GB, max = {max_mb:.1f} MB")
        return

    all_stats = []
    total_start = time.time()

    for name, dist_func in configs:
        print(f"\nGenerating {name}...")
        stats = generate_dataset(
            output_dir=args.output,
            name=name,
            num_files=NUM_FILES,
            total_bytes=target_bytes,
            dist_func=dist_func,
            seed=args.seed
        )
        all_stats.append(stats)
        print(f"  Actual: {stats['total_GB']:.3f} GB, CV = {stats['cv']:.4f}, "
              f"max file = {stats['max_MB']:.1f} MB")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 90)
    print(f"Summary for paper experiments (all datasets have {NUM_FILES} files)")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Size(GB)':<12} {'Mean(blocks)':<14} {'Std(blocks)':<14} {'CV':<10} {'Max(MB)':<10}")
    for s in all_stats:
        print(f"{s['name']:<20} {s['total_GB']:<12.3f} "
              f"{s['mean_blocks']:<14.2f} {s['std_blocks']:<14.2f} {s['cv']:<10.4f} {s['max_MB']:<10.1f}")

    print("\n" + "=" * 90)
    print(f"All datasets saved under: {Path(args.output).absolute()}")
    print(f"Total generation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()