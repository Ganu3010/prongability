#!/usr/bin/env python3
"""
Post-collection tooling:
  1. Stratification coverage report across the axes called out in the spec
     (distance bucket, relative-velocity bucket, occlusion, object
     count/type in O, weather, scenario type), flagging near-zero buckets.
  2. Episode-level (not frame-level) train/val/test split, written as JSON
     lists of episode directory names.

Usage:
    python postprocess.py --dataset ./rulebook_dataset --train 0.7 --val 0.15 --test 0.15
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-bucket-flag", type=int, default=20,
                    help="Flag any stratification bucket with fewer than this many object-instances")
    return p.parse_args()


def load_episodes(dataset_dir):
    episodes = []
    for name in sorted(os.listdir(dataset_dir)):
        ep_dir = os.path.join(dataset_dir, name)
        meta_path = os.path.join(ep_dir, "metadata.json")
        if os.path.isdir(ep_dir) and os.path.isfile(meta_path):
            with open(meta_path) as f:
                records = json.load(f)
            episodes.append((name, records))
    return episodes


def coverage_report(episodes, min_flag):
    scenario_counts = Counter()
    weather_counts = Counter()
    distance_counts = Counter()
    velocity_counts = Counter()
    occlusion_counts = Counter()
    object_count_bucket = Counter()  # empty / single / multiple, per frame
    object_class_counts = Counter()
    combo_counts = defaultdict(int)  # (weather, scenario_type) frame coverage

    for _, records in episodes:
        for rec in records:
            scenario_counts[rec["scenario_type"]] += 1
            weather_counts[rec["weather_preset"]] += 1
            combo_counts[(rec["weather_preset"], rec["scenario_type"])] += 1

            in_O_objs = [o for o in rec["objects"] if o["in_O"]]
            if len(in_O_objs) == 0:
                object_count_bucket["empty"] += 1
            elif len(in_O_objs) == 1:
                object_count_bucket["single"] += 1
            else:
                object_count_bucket["multiple"] += 1

            for obj in rec["objects"]:
                distance_counts[obj["distance_bucket"]] += 1
                velocity_counts[obj["velocity_bucket"]] += 1
                occlusion_counts[obj["occlusion"]] += 1
                object_class_counts[obj["obj_class"]] += 1

    def show(title, counter):
        print(f"\n{title}")
        total = sum(counter.values()) or 1
        for k, v in sorted(counter.items(), key=lambda kv: -kv[1]):
            flag = "  <-- LOW COVERAGE, target a follow-up run" if v < min_flag else ""
            print(f"  {k:20s} {v:8d}  ({100*v/total:5.1f}%){flag}")

    print("=" * 70)
    print("STRATIFICATION COVERAGE REPORT")
    print("=" * 70)
    show("Scenario type (frames)", scenario_counts)
    show("Weather preset (frames)", weather_counts)
    show("Object count in O (frames)", object_count_bucket)
    show("Distance bucket (object-instances)", distance_counts)
    show("Relative velocity bucket (object-instances)", velocity_counts)
    show("Occlusion level (object-instances)", occlusion_counts)
    show("Object class (object-instances)", object_class_counts)

    print("\nWeather x Scenario-type frame coverage:")
    for (weather, sc), v in sorted(combo_counts.items()):
        flag = "  <-- LOW COVERAGE" if v < min_flag else ""
        print(f"  {weather:20s} x {sc:12s} {v:8d}{flag}")


def episode_split(episodes, train_frac, val_frac, test_frac, seed):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "split fractions must sum to 1.0"
    names = [name for name, _ in episodes]
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split = {
        "train": sorted(names[:n_train]),
        "val": sorted(names[n_train:n_train + n_val]),
        "test": sorted(names[n_train + n_val:]),
    }
    return split


def main():
    args = parse_args()
    episodes = load_episodes(args.dataset)
    if not episodes:
        print(f"No episodes with metadata.json found under {args.dataset}")
        return

    coverage_report(episodes, args.min_bucket_flag)

    split = episode_split(episodes, args.train, args.val, args.test, args.seed)
    split_path = os.path.join(args.dataset, "split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nEpisode-level split written to {split_path}")
    print(f"  train: {len(split['train'])} episodes")
    print(f"  val:   {len(split['val'])} episodes")
    print(f"  test:  {len(split['test'])} episodes")


if __name__ == "__main__":
    main()
