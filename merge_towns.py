#!/usr/bin/env python3
"""
Merges run_cluster.sh's per-town output (rulebook_dataset/Town01/episode_0000,
rulebook_dataset/Town02/episode_0000, ...) into a single flat directory with
globally renumbered, non-colliding episode IDs -- so postprocess.py and
probing.py can be run once across the whole merged dataset instead of once
per town.

Each episode's metadata.json has its embedded episode_id field rewritten to
match the new global numbering (not just the folder name), so probe_table.csv
rows stay unambiguous -- otherwise two different towns' "episode_id=0" would
collide once flattened into one CSV.

Usage:
    python merge_towns.py --input ./rulebook_dataset --output ./rulebook_dataset_merged
    python postprocess.py --dataset ./rulebook_dataset_merged
    python probing.py --dataset ./rulebook_dataset_merged --output ./rulebook_dataset_merged/probe_table.csv
"""

import argparse
import json
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Parent dir containing per-town subdirectories (e.g. ./rulebook_dataset from run_cluster.sh)")
    p.add_argument("--output", required=True,
                   help="Where to write the merged, globally-renumbered dataset")
    p.add_argument("--move", action="store_true",
                   help="Move episodes instead of copying (saves disk space, but empties the per-town "
                        "directories in the process -- use --move only once you're done inspecting them separately)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    town_dirs = sorted(
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
    )
    if not town_dirs:
        print(f"No subdirectories found under {args.input}")
        return

    global_id = 0
    total_frames = 0
    for town_dir in town_dirs:
        town_path = os.path.join(args.input, town_dir)
        episode_dirs = sorted(
            d for d in os.listdir(town_path)
            if os.path.isdir(os.path.join(town_path, d)) and os.path.isfile(os.path.join(town_path, d, "metadata.json"))
        )
        if not episode_dirs:
            print(f"  {town_dir}: no episodes found, skipping")
            continue

        for ep_dir in episode_dirs:
            src = os.path.join(town_path, ep_dir)
            dst_name = f"episode_{global_id:04d}"
            dst = os.path.join(args.output, dst_name)

            if os.path.exists(dst):
                raise FileExistsError(f"{dst} already exists -- refusing to overwrite. "
                                       f"Clear --output or pick a fresh directory before merging.")

            if args.move:
                shutil.move(src, dst)
            else:
                shutil.copytree(src, dst)

            meta_path = os.path.join(dst, "metadata.json")
            with open(meta_path) as f:
                records = json.load(f)
            for rec in records:
                rec["episode_id"] = global_id
            with open(meta_path, "w") as f:
                json.dump(records, f)

            total_frames += len(records)
            print(f"  {town_dir}/{ep_dir} -> {dst_name} ({len(records)} frames)")
            global_id += 1

    print(f"\nMerged {global_id} episodes ({total_frames} frames total) from {len(town_dirs)} town(s) into {args.output}")
    print("Now run:")
    print(f"  python postprocess.py --dataset {args.output}")
    print(f"  python probing.py --dataset {args.output} --output {os.path.join(args.output, 'probe_table.csv')}")


if __name__ == "__main__":
    main()