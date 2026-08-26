#!/usr/bin/env python3
"""
Assembles the final probe-ready dataset from a temporal manifest
(build_temporal_dataset.py) and a DINOv3 embedding cache
(encode_frames.py): each output record has ONLY the window's frame
encodings and the target variables -- no paths, no episode/timestep
bookkeeping, no scenario metadata.

Split into train/val/test files using split.json (from postprocess.py)
BEFORE dropping episode_id, since episode-level split membership is the
only reason episode_id would be needed at all -- once split, each
example is self-contained.

Default target set is the rulebook-relevant variables identified earlier
(nearest_in_O__*, nearest_any__*, any_colliding, num_objects_*) --
deliberately excluding ego__*/action__* (the ego's own kinematics/control,
a different kind of target: proprioceptive rather than about the scene
around it) unless you pass --include-ego-action.

Usage:
    python build_encoded_dataset.py \\
        --temporal-manifest ./temporal_dataset.jsonl \\
        --embeddings ./frame_embeddings.npz \\
        --split ./rulebook_dataset_merged/split.json \\
        --output-dir ./encoded_dataset
"""

import argparse
import json
import os

import numpy as np

CORE_TARGET_COLUMNS = [
    "num_objects_total",
    "num_objects_in_O",
    "any_colliding",
    "nearest_in_O__distance",
    "nearest_in_O__signed_velocity",
    "nearest_in_O__relative_angle_deg",
    "nearest_in_O__occlusion",
    "nearest_in_O__obj_class",
    "nearest_any__distance",
    "nearest_any__signed_velocity",
    "nearest_any__relative_angle_deg",
    "nearest_any__occlusion",
    "nearest_any__obj_class",
]

EGO_ACTION_COLUMNS = [
    "ego__x", "ego__y", "ego__theta", "ego__v_e", "ego__a_e",
    "action__throttle", "action__brake", "action__steer",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal-manifest", required=True)
    p.add_argument("--embeddings", required=True, help="Output of encode_frames.py (.npz)")
    p.add_argument("--split", required=True, help="split.json from postprocess.py")
    p.add_argument("--output-dir", default="./encoded_dataset")
    p.add_argument("--include-ego-action", action="store_true",
                   help="Also include ego kinematics + applied control as targets, "
                         "not just the object/rulebook-relevant variables")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    target_columns = CORE_TARGET_COLUMNS + (EGO_ACTION_COLUMNS if args.include_ego_action else [])

    print(f"Loading embedding cache from {args.embeddings} ...")
    cache = np.load(args.embeddings, allow_pickle=False)
    path_to_idx = {p: i for i, p in enumerate(cache["paths"])}
    embeddings = cache["embeddings"]
    print(f"  {len(path_to_idx)} cached embeddings, dim={embeddings.shape[1]}")

    with open(args.split) as f:
        split = json.load(f)
    episode_to_split = {}
    for split_name, episode_dirs in split.items():
        for ep_dir in episode_dirs:
            episode_to_split[ep_dir] = split_name

    out_records = {"train": [], "val": [], "test": []}
    n_missing_embedding = 0
    n_missing_split = 0
    n_total = 0

    with open(args.temporal_manifest) as f:
        for line in f:
            row = json.loads(line)
            n_total += 1

            ep_dir = f"episode_{row['episode_id']:04d}"
            split_name = episode_to_split.get(ep_dir)
            if split_name is None:
                n_missing_split += 1
                continue

            try:
                frame_encodings = [embeddings[path_to_idx[p]].tolist() for p in row["frame_window"]]
            except KeyError:
                n_missing_embedding += 1
                continue

            record = {"frame_encodings": frame_encodings}
            for col in target_columns:
                record[col] = row.get(col)

            out_records[split_name].append(record)

    for split_name, records in out_records.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"  {split_name}: {len(records)} examples -> {out_path}")

    print(f"\n{n_total} manifest rows processed.")
    if n_missing_split:
        print(f"WARNING: {n_missing_split} rows skipped -- episode not found in split.json "
              f"(manifest and split.json likely built from different dataset directories)")
    if n_missing_embedding:
        print(f"WARNING: {n_missing_embedding} rows skipped -- a frame in their window wasn't in the "
              f"embedding cache (did encode_frames.py run against the same temporal-manifest?)")


if __name__ == "__main__":
    main()