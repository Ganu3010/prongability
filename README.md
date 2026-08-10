# CARLA Data Collector — RGB + Depth + Radar + Rulebook Variables

Collects RGB, depth, and radar from CARLA, plus the raw per-object and
ego-kinematic variables a rulebook would consume — distance, relative
velocity, relative angle, occlusion, and collision — computed directly
from simulator state. No config file, no rule-engine dependency, no
external project needed. Intended for training linear/non-linear probes
on frozen-encoder latents to check whether these quantities are
recoverable from the encoded representation.

## Variables extracted (per object, per frame)

`rulebook.py` builds a ground-truth object list every tick — every
nearby vehicle/pedestrian/cyclist within 80m — with:

- `distance` — straight-line distance from ego (meters)
- `signed_velocity` — relative closing speed (+ = approaching, - = receding)
- `relative_angle_deg` — signed angle from ego's forward vector (0 = straight
  ahead, +90 = directly right, -90 = directly left, ±180 = behind)
- `occlusion` — "none"/"partial"/"heavy", estimated by projecting the
  object's 3D bounding box into the depth camera and comparing expected
  vs. recorded depth
- `colliding` — bool, from a two-tick bounding-box sweep (ego's box this
  tick + previous tick) intersected against the object's box, so
  fast-moving collisions aren't missed by a single-frame overlap check
- `in_lane` / `in_front` / `in_O` (`in_lane and in_front`) — geometric
  relevance flags
- `bbox_3d` — 8 world-space corner points
- `distance_bucket` / `velocity_bucket` — for stratification (near/mid/far,
  approaching/receding/stationary)

Plus per-frame ego kinematics (`ego_state`: x, y, theta, v_e [m/s], a_e
[m/s²]) and the actuation command actually applied (`action`:
throttle/brake/steer).

No `RuleBook`, no `cfg.common.rulebook`, no `praeception`, no
`agents.tools.misc` — just `shapely` for the collision-polygon
intersection test, which is the only non-stdlib/non-CARLA dependency:
```bash
pip install shapely
```

**Still a placeholder:** `control.py`'s `BaselineController` is a generic
constant-time-headway gap follower, not tied to any specific control
law — it only exists to drive episodes so there's varied
distance/velocity/braking behavior for the extracted variables to have
signal in. Swap it if your ego's driving policy itself needs to match
something specific.

## Building a probe-ready table

`probing.py` flattens each frame's ego state + object list into one CSV
row:

```bash
python probing.py --dataset ./rulebook_dataset --output ./probe_table.csv
```

Columns: `episode_id`, `timestep`, `map`, `weather_preset`,
`scenario_type`, `scenario_name`, `rgb_frame`/`depth_frame`/`radar_points`
(for joining against your latent-extraction pass), `ego__*` kinematics,
`action__*`, `num_objects_total`, `num_objects_in_O`, `any_colliding`, and
`nearest_in_O__*` / `nearest_any__*` (distance, signed velocity, relative
angle, occlusion, class) as convenience scalar targets. Missing values
(no object in O that frame) are left blank rather than zero-filled, since
"no object" is meaningfully different from "distance = 0."

The object list is variable-length per frame, so this table only carries
the nearest-object summary. If you want full per-object rows (one row per
object per frame, e.g. for a probe that predicts variables for every
visible object rather than just the nearest), read `metadata.json`
directly — the `objects` field is already flat, structured data, ready to
`explode()` in pandas.

## Files

- `collect_data.py` — main entry point, orchestrates episodes across
  towns x weather x scenario-type combinations.
- `rulebook.py` — the actual variable extraction (distance, angle,
  velocity, occlusion, collision) described above.
- `probing.py` — flattens `metadata.json` files into one CSV row per frame.
- `control.py` — baseline longitudinal controller (category a) + noise
  injector (Gaussian accel/steer noise, brake dropout, reaction delay —
  category b).
- `scenarios.py` — scripted adversarial scenarios: cut-in, jaywalk,
  hard-braking lead vehicle, occlusion (category c).
- `postprocess.py` — stratification coverage report + episode-level
  train/val/test split.

## Usage

Start the CARLA server first, then:

```bash
python collect_data.py \
    --towns Town01 Town03 Town05 \
    --weather ClearNoon WetNoon HardRainNoon CloudySunset \
    --episodes-per-combo 3 \
    --frames-per-episode 150 \
    --output ./rulebook_dataset
```

With 3 towns x 4 weather presets x 3 episodes x 150 frames, that's
5,400 frames — trim `--episodes-per-combo` / `--frames-per-episode` for a
smaller target, or widen it if some scenario/weather combos come back
thin on variable coverage.

Scenario type is chosen per-episode via `--scenario-weights` (default
`0.3 0.3 0.4` for baseline/noise/adversarial — weighted toward
adversarial since that's where non-degenerate variable values — close
calls, near-collisions, hard braking — actually show up; a purely
rule-compliant baseline policy mostly produces "nothing interesting
happening" frames).

For a quick local smoke test before a full/cluster run, cut frames and
episodes way down and watch the CARLA window:
```bash
python collect_data.py --towns Town01 --weather ClearNoon \
    --episodes-per-combo 1 --frames-per-episode 50 \
    --num-background-vehicles 5 --output ./test_run
```

Then check coverage, generate the split, and build the probe table:

```bash
python postprocess.py --dataset ./rulebook_dataset --train 0.7 --val 0.15 --test 0.15
python probing.py --dataset ./rulebook_dataset --output ./probe_table.csv
```

`postprocess.py` prints per-bucket counts across distance/velocity/
occlusion/object-count/weather/scenario-type axes and flags any bucket
under `--min-bucket-flag` (default 20 object-instances).

## Output structure

```
rulebook_dataset/
  episode_0000/
    rgb/000000.png ...
    depth/000000.png ...       # LogarithmicDepth-encoded PNG
    radar/000000.json ...      # list of {depth, azimuth, altitude, velocity}
    metadata.json              # one record per timestep, schema below
  episode_0001/
    ...
  split.json                  # {"train": [...], "val": [...], "test": [...]}
probe_table.csv                # flattened, one row per frame (from probing.py)
```

Each `metadata.json` record: `episode_id`, `timestep`, `map`,
`weather_preset`, `rgb_frame`/`depth_frame`/`radar_points` (relative
paths, with timestamps implicit in `episode_id`+`timestep`), `ego_state`,
`action`, `objects` (full per-object variable list described above),
`scenario_type`, `scenario_name`.

## Known simplifications worth knowing about

- **Occlusion estimation** projects each object's 8 bounding-box corners
  into the depth camera and compares expected vs. recorded depth. It's a
  reasonable proxy but not pixel-perfect segmentation-based occlusion —
  if you need tighter ground truth, an instance segmentation camera with
  CARLA's per-actor instance IDs would be more precise (not included
  here since the sensor suite is RGB + depth + radar only).
- **Adversarial scenarios** are scripted directly against the CARLA API
  rather than through `scenario_runner`/OpenSCENARIO. Fine at this scale;
  worth migrating to `scenario_runner` for a broader standardized
  scenario library at much larger scale.
- **Radar is logged raw** (per-detection depth/azimuth/altitude/velocity)
  and isn't used anywhere in the variable extraction — only ground-truth
  simulator state feeds `rulebook.py`, never a perception model's output.
  Radar is there for your perception pipeline to consume separately.
- **Collision detection** only looks at the ego's own two-tick bbox sweep
  vs. other actors — it won't catch two background vehicles colliding
  with each other (not relevant to ego-centric probing, but worth knowing
  if you ever want that too).
