#!/usr/bin/env python3
"""
Flattens each frame's ego state + object list into one row per frame,
written as CSV — join this against your encoder's latents by
(episode_id, timestep) or by rgb_frame/depth_frame path to build a probe
train/eval table.

Per-object variables (distance, signed_velocity, relative_angle_deg,
occlusion, colliding, in_lane, in_front, in_O) are reduced to a few
summary columns per frame (nearest in-O object, nearest any object,
counts) since the object list is variable-length. If you want full
per-object rows instead (one row per object per frame rather than one
row per frame), read metadata.json directly — the "objects" field is
already flat, structured data.

Usage:
    python probing.py --dataset ./rulebook_dataset --output ./probe_table.csv
"""

import argparse
import csv
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="./probe_table.csv")
    return p.parse_args()


def flatten_record(record):
    row = {
        "episode_id": record["episode_id"],
        "timestep": record["timestep"],
        "map": record["map"],
        "weather_preset": record["weather_preset"],
        "scenario_type": record["scenario_type"],
        "scenario_name": record["scenario_name"] or "",
        "rgb_frame": record["rgb_frame"],
        "depth_frame": record["depth_frame"],
        "radar_points": record["radar_points"],
    }

    ego = record["ego_state"]
    row["ego__x"] = ego["x"]
    row["ego__y"] = ego["y"]
    row["ego__theta"] = ego["theta"]
    row["ego__v_e"] = ego["v_e"]
    row["ego__a_e"] = ego["a_e"]

    action = record["action"]
    row["action__throttle"] = action["throttle"]
    row["action__brake"] = action["brake"]
    row["action__steer"] = action["steer"]

    objects = record["objects"]
    row["num_objects_total"] = len(objects)
    row["num_objects_in_O"] = sum(1 for o in objects if o["in_O"])
    row["any_colliding"] = int(any(o["colliding"] for o in objects))

    in_O = [o for o in objects if o["in_O"]]
    if in_O:
        nearest_in_O = min(in_O, key=lambda o: o["distance"])
        row["nearest_in_O__distance"] = nearest_in_O["distance"]
        row["nearest_in_O__signed_velocity"] = nearest_in_O["signed_velocity"]
        row["nearest_in_O__relative_angle_deg"] = nearest_in_O["relative_angle_deg"]
        row["nearest_in_O__occlusion"] = nearest_in_O["occlusion"]
        row["nearest_in_O__obj_class"] = nearest_in_O["obj_class"]
    else:
        row["nearest_in_O__distance"] = ""
        row["nearest_in_O__signed_velocity"] = ""
        row["nearest_in_O__relative_angle_deg"] = ""
        row["nearest_in_O__occlusion"] = ""
        row["nearest_in_O__obj_class"] = ""

    if objects:
        nearest_any = min(objects, key=lambda o: o["distance"])
        row["nearest_any__distance"] = nearest_any["distance"]
        row["nearest_any__signed_velocity"] = nearest_any["signed_velocity"]
        row["nearest_any__relative_angle_deg"] = nearest_any["relative_angle_deg"]
        row["nearest_any__occlusion"] = nearest_any["occlusion"]
        row["nearest_any__obj_class"] = nearest_any["obj_class"]
    else:
        row["nearest_any__distance"] = ""
        row["nearest_any__signed_velocity"] = ""
        row["nearest_any__relative_angle_deg"] = ""
        row["nearest_any__occlusion"] = ""
        row["nearest_any__obj_class"] = ""

    return row


def main():
    args = parse_args()
    episode_dirs = sorted(
        d for d in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, d)) and os.path.isfile(os.path.join(args.dataset, d, "metadata.json"))
    )
    if not episode_dirs:
        print(f"No episodes with metadata.json found under {args.dataset}")
        return

    rows = []
    for ep in episode_dirs:
        with open(os.path.join(args.dataset, ep, "metadata.json")) as f:
            records = json.load(f)
        for rec in records:
            rows.append(flatten_record(rec))

    if not rows:
        print("No frames found.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows, {len(fieldnames)} columns, to {args.output}")


if __name__ == "__main__":
    main()
