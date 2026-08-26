#!/usr/bin/env python3
"""
Builds a temporal-window dataset: each example pairs a WINDOW of the last
N frames' RGB paths (for DINOv3 encoding downstream -- this script only
builds the manifest, it doesn't run any encoder) with the CURRENT frame's
rulebook-variable targets (same columns as probing.py's flatten_record).

Window convention (read this before using):
    By default, the window is frames [t-N+1, ..., t] -- it INCLUDES the
    current frame as the last element, not just the N frames before it.
    That's the setup for "does temporal context ending at frame t help
    predict frame t's variables." If you instead want a strictly causal
    test -- predict frame t's variables from ONLY the preceding frames,
    with frame t's own RGB held out entirely -- pass --exclude-current.
    These are different experiments; pick deliberately, not by default.

Windows never cross episode boundaries. Episodes are independent
simulation runs (different scene, route, scenario) -- frame 199 of one
episode has no temporal relationship to frame 0 of the next.

Output is JSONL (one JSON object per line), not CSV, since each row
carries a variable-length list of frame paths that doesn't fit a flat
table. All paths (frame_window entries, and this row's own
rgb_frame/depth_frame/radar_points) are DATASET-ROOT-relative (e.g.
"episode_0003/rgb/000198.png"), not episode-relative like
probing.py's CSV output -- so you can load them directly by joining with
your dataset root, without reconstructing episode directory names from
episode_id yourself.

Usage:
    python build_temporal_dataset.py --dataset ./rulebook_dataset_merged \\
        --window-size 8 --output ./temporal_dataset.jsonl

    # strictly causal variant (current frame's own RGB excluded from input):
    python build_temporal_dataset.py --dataset ./rulebook_dataset_merged \\
        --window-size 8 --exclude-current --output ./temporal_dataset_causal.jsonl
"""

import argparse
import json
import os

from probing import flatten_record


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--window-size", type=int, default=8,
                   help="Number of frames per input window (includes the current frame "
                        "unless --exclude-current is set)")
    p.add_argument("--exclude-current", action="store_true",
                   help="Hold the current frame's own RGB out of the window -- window becomes "
                        "the N frames strictly BEFORE the current frame, a stricter causal-only test")
    p.add_argument("--stride", type=int, default=1,
                   help="Step between consecutive windows' current-frame timestep within an "
                        "episode. 1 = a window for every valid frame (max overlap/data usage); "
                        "raise this to reduce overlap between consecutive examples.")
    p.add_argument("--output", default="./temporal_dataset.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    episode_dirs = sorted(
        d for d in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, d)) and os.path.isfile(os.path.join(args.dataset, d, "metadata.json"))
    )
    if not episode_dirs:
        print(f"No episodes with metadata.json found under {args.dataset}")
        return

    n_examples = 0
    n_skipped_episodes = 0
    min_frames_needed = args.window_size + (1 if args.exclude_current else 0)

    with open(args.output, "w") as out_f:
        for ep in episode_dirs:
            with open(os.path.join(args.dataset, ep, "metadata.json")) as f:
                records = json.load(f)
            if len(records) < min_frames_needed:
                n_skipped_episodes += 1
                continue

            # records are already in ascending timestep order (0, 1, 2, ...)
            # as written by collect_data.py.
            start_t = min_frames_needed - 1
            for t in range(start_t, len(records), args.stride):
                if args.exclude_current:
                    window_records = records[t - args.window_size: t]
                else:
                    window_records = records[t - args.window_size + 1: t + 1]

                frame_window = [os.path.join(ep, r["rgb_frame"]) for r in window_records]

                current = records[t]
                row = flatten_record(current)
                # make this row's own path fields dataset-root-relative too,
                # consistent with frame_window (probing.py's CSV keeps these
                # episode-relative instead -- different convention, deliberately)
                row["rgb_frame"] = os.path.join(ep, current["rgb_frame"])
                row["depth_frame"] = os.path.join(ep, current["depth_frame"])
                row["radar_points"] = os.path.join(ep, current["radar_points"])

                row["frame_window"] = frame_window
                row["window_size"] = args.window_size
                row["excludes_current"] = args.exclude_current
                out_f.write(json.dumps(row) + "\n")
                n_examples += 1

    print(f"Wrote {n_examples} windowed example(s) to {args.output}")
    if n_skipped_episodes:
        print(f"Skipped {n_skipped_episodes} episode(s) with fewer than {min_frames_needed} frames "
              f"(not enough to form a single window of size {args.window_size}"
              f"{' + 1 held-out current frame' if args.exclude_current else ''})")


if __name__ == "__main__":
    main()