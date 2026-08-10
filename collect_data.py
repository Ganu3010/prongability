#!/usr/bin/env python3
"""
CARLA data collection pipeline: RGB + depth + radar, plus the raw
per-object and ego-kinematic variables a rulebook would consume
(distance, relative velocity, relative angle, occlusion, collision),
computed directly from simulator state — no external rule-engine or
config dependency (see rulebook.py). Runs across baseline /
noise-injected / adversarial / weather-and-map-diverse episodes, one
directory per episode, split by episode (not frame) downstream via
postprocess.py. Intended for probing whether encoded latents can predict
these variables individually.

See control.py module docstring for a caveat about BaselineController
not being verified against any specific paper's control law — it's a
generic gap-follower used to drive episodes, nothing more.

Usage:
    python collect_data.py \\
        --towns Town01 Town03 Town05 \\
        --weather ClearNoon WetNoon HardRainNoon CloudySunset \\
        --episodes-per-combo 3 \\
        --frames-per-episode 150 \\
        --output ./rulebook_dataset
"""

import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import sys

import carla
import numpy as np

import control
import rulebook
import scenarios


IMAGE_W, IMAGE_H, FOV = 640, 640, 90.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--towns", nargs="+", default=["Town01", "Town03"])
    p.add_argument("--weather", nargs="+",
                    default=["ClearNoon", "WetNoon", "HardRainNoon", "CloudySunset"],
                    help="Names must match carla.WeatherParameters presets")
    p.add_argument("--episodes-per-combo", type=int, default=2,
                    help="Episodes to run per (town, weather) combination")
    p.add_argument("--frames-per-episode", type=int, default=150)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--output", default="./rulebook_dataset")
    p.add_argument("--num-background-vehicles", type=int, default=15)
    p.add_argument("--target-speed-kmh", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=42)
    # roughly balance categories (a)/(b)/(c) per the spec, not baseline-dominated
    p.add_argument("--scenario-weights", nargs=3, type=float, default=[0.3, 0.3, 0.4],
                    metavar=("BASELINE_W", "NOISE_W", "ADVERSARIAL_W"))
    return p.parse_args()


def decode_depth_meters(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b, g, r = arr[:, :, 0].astype(np.float64), arr[:, :, 1].astype(np.float64), arr[:, :, 2].astype(np.float64)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (1000.0 * normalized).astype(np.float32)  # CARLA depth camera max range = 1000m


def decode_radar(radar_data: carla.RadarMeasurement):
    points = []
    for det in radar_data:
        points.append({"depth": det.depth, "azimuth": det.azimuth, "altitude": det.altitude, "velocity": det.velocity})
    return points


def compute_steer(ego_vehicle, carla_map, lookahead=4.0) -> float:
    """Minimal lane-keeping steering: heading error toward a waypoint a few
    meters ahead on the current lane."""
    transform = ego_vehicle.get_transform()
    wp = carla_map.get_waypoint(transform.location)
    next_wps = wp.next(lookahead)
    if not next_wps:
        return 0.0
    target = next_wps[0].transform.location
    fwd = transform.get_forward_vector()
    dx, dy = target.x - transform.location.x, target.y - transform.location.y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return 0.0
    dx, dy = dx / dist, dy / dist
    dot = fwd.x * dx + fwd.y * dy
    cross = fwd.x * dy - fwd.y * dx
    angle = math.atan2(cross, dot)
    return max(-1.0, min(1.0, angle * 1.5))


def choose_scenario_type(rng, weights):
    return rng.choices(["baseline", "noise", "adversarial"], weights=weights, k=1)[0]


def make_sensor_transform():
    return carla.Transform(carla.Location(x=1.6, z=1.7))


def run_episode(world, tm, carla_map, blueprint_library, args, episode_id, town, weather_name,
                 scenario_type, scenario_name, out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("rgb", "depth", "radar"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    spawn_points = carla_map.get_spawn_points()
    ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    ego_spawn_point = rng.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    if ego_vehicle is None:
        print(f"  [episode {episode_id}] failed to spawn ego, skipping")
        return
    world.tick()  # settle ego into the simulation before anything reads its transform

    vehicle_actors = [ego_vehicle]  # destroyed last: ego, background traffic, scenario vehicles
    walker_actors = []  # destroyed after controllers, before vehicles
    controller_actors = []  # destroyed first among non-sensor actors
    sensor_actors = []  # destroyed before everything else

    # background traffic — exclude the ego's own spawn point from the pool
    # so a background vehicle doesn't silently fail to spawn on top of ego
    veh_bps = [bp for bp in blueprint_library.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels")) == 4]
    available_spawns = [sp for sp in spawn_points if sp.location != ego_spawn_point.location]
    bg_spawns = rng.sample(available_spawns, min(args.num_background_vehicles, len(available_spawns)))
    bg_spawned = 0
    for sp in bg_spawns:
        bp = rng.choice(veh_bps)
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            vehicle_actors.append(actor)
            bg_spawned += 1
    print(f"  [episode {episode_id}] background vehicles: {bg_spawned}/{len(bg_spawns)} spawned "
          f"(requested {args.num_background_vehicles}, {len(spawn_points)} total spawn points on {town})")

    # scenario-specific actors
    scenario_state = {"triggered": False, "trigger_tick": rng.randint(20, args.frames_per_episode - 20)
                       if args.frames_per_episode > 40 else args.frames_per_episode // 2}
    if scenario_type == "adversarial":
        if scenario_name == "cut_in":
            a = scenarios.spawn_cut_in_vehicle(world, ego_vehicle, blueprint_library, tm)
            if a:
                vehicle_actors.append(a)
                scenario_state["cut_in_actor"] = a
                print(f"  [episode {episode_id}] cut_in vehicle spawned (actor {a.id}), "
                      f"will trigger at tick {scenario_state['trigger_tick']}")
            else:
                print(f"  [episode {episode_id}] WARNING: cut_in vehicle failed to spawn "
                      f"(likely no adjacent lane at this spawn point) — episode will only have background traffic, if any")
        elif scenario_name == "jaywalk":
            w, c = scenarios.spawn_jaywalker(world, ego_vehicle, blueprint_library)
            if c:
                controller_actors.append(c)
            if w:
                walker_actors.append(w)
            if not (w and c):
                print(f"  [episode {episode_id}] WARNING: jaywalker failed to spawn")
        elif scenario_name == "hard_brake_lead":
            a = scenarios.spawn_hard_braking_lead(world, ego_vehicle, blueprint_library, tm)
            if a:
                vehicle_actors.append(a)
                scenario_state["lead_actor"] = a
            else:
                print(f"  [episode {episode_id}] WARNING: hard_brake_lead vehicle failed to spawn")
        elif scenario_name == "occlusion":
            blocker, hidden = scenarios.spawn_occlusion_blocker(world, ego_vehicle, blueprint_library, tm)
            if blocker:
                vehicle_actors.append(blocker)
            if hidden:
                # spawn_occlusion_blocker's "hidden" actor is a walker with no controller
                walker_actors.append(hidden)

    # In synchronous mode, get_actors() reflects the last-ticked world
    # snapshot -- actors spawned since the last tick won't show up (or be
    # safely destroyable) until a tick happens. Tick once here so
    # everything spawned above actually "exists" before sensors attach and
    # the main loop starts reading world state.
    world.tick()

    world_vehicle_count = len(world.get_actors().filter("vehicle.*"))
    print(f"  [episode {episode_id}] total vehicle actors in world after spawning + settle tick: {world_vehicle_count}")

    # sensors
    sensor_transform = make_sensor_transform()
    K = rulebook.build_camera_intrinsics(IMAGE_W, IMAGE_H, FOV)

    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(IMAGE_W))
    rgb_bp.set_attribute("image_size_y", str(IMAGE_H))
    rgb_bp.set_attribute("fov", str(FOV))
    rgb_cam = world.spawn_actor(rgb_bp, sensor_transform, attach_to=ego_vehicle)

    depth_bp = blueprint_library.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(IMAGE_W))
    depth_bp.set_attribute("image_size_y", str(IMAGE_H))
    depth_bp.set_attribute("fov", str(FOV))
    depth_cam = world.spawn_actor(depth_bp, sensor_transform, attach_to=ego_vehicle)

    radar_bp = blueprint_library.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "35")
    radar_bp.set_attribute("vertical_fov", "10")
    radar_bp.set_attribute("range", "100")
    radar = world.spawn_actor(radar_bp, sensor_transform, attach_to=ego_vehicle)

    import queue
    rgb_q, depth_q, radar_q = queue.Queue(), queue.Queue(), queue.Queue()
    rgb_cam.listen(rgb_q.put)
    depth_cam.listen(depth_q.put)
    radar.listen(radar_q.put)
    sensor_actors.extend([rgb_cam, depth_cam, radar])

    baseline_ctrl = control.BaselineController(target_speed_kmh=args.target_speed_kmh)
    noise = control.NoiseInjector(seed=rng.randint(0, 1_000_000)) if scenario_type == "noise" else None

    metadata_records = []
    nearest_lead = None
    applied_control = carla.VehicleControl()
    prev_ego_bbox = None
    prev_ego_transform = None
    ego_start_loc = ego_vehicle.get_transform().location
    closest_ever = None

    try:
        for t in range(args.frames_per_episode):
            # trigger scripted scenario events at a designated tick
            if scenario_type == "adversarial" and not scenario_state["triggered"] and t >= scenario_state["trigger_tick"]:
                if scenario_name == "cut_in" and "cut_in_actor" in scenario_state:
                    scenarios.trigger_cut_in(tm, scenario_state["cut_in_actor"])
                elif scenario_name == "hard_brake_lead" and "lead_actor" in scenario_state:
                    scenarios.trigger_hard_brake(scenario_state["lead_actor"])
                scenario_state["triggered"] = True
            if scenario_type == "adversarial" and scenario_name == "hard_brake_lead" and scenario_state["triggered"]:
                # hold the brake for a few ticks after triggering
                if t < scenario_state["trigger_tick"] + 15 and "lead_actor" in scenario_state:
                    scenarios.trigger_hard_brake(scenario_state["lead_actor"])

            # compute + apply control from previous tick's perception
            steer = compute_steer(ego_vehicle, carla_map)
            lead_distance = nearest_lead.distance if nearest_lead else None
            lead_relative_speed = -nearest_lead.signed_velocity if nearest_lead else None  # convert to control.py's sign convention
            ego_speed_now = math.hypot(ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y)
            base_control = baseline_ctrl.step(ego_speed_now, lead_distance, lead_relative_speed)
            base_control.steer = steer
            applied_control = noise.apply(base_control) if noise else base_control
            ego_vehicle.apply_control(applied_control)

            world.tick()

            rgb_img = rgb_q.get()
            depth_img = depth_q.get()
            radar_data = radar_q.get()

            depth_arr = decode_depth_meters(depth_img)
            objects = rulebook.extract_object_list(
                world, ego_vehicle, depth_arr, depth_cam, K, IMAGE_W, IMAGE_H,
                prev_ego_bbox=prev_ego_bbox, prev_ego_transform=prev_ego_transform,
            )
            in_O = [o for o in objects if o.in_O]
            nearest_lead = min(in_O, key=lambda o: o.distance) if in_O else None

            d = rulebook.closest_actor_distance(world, ego_vehicle)
            if d is not None and (closest_ever is None or d < closest_ever):
                closest_ever = d

            # update collision-sweep state for next tick
            prev_ego_bbox = ego_vehicle.bounding_box
            prev_ego_transform = ego_vehicle.get_transform()

            rgb_path = os.path.join("rgb", f"{t:06d}.png")
            depth_path = os.path.join("depth", f"{t:06d}.png")
            radar_path = os.path.join("radar", f"{t:06d}.json")
            rgb_img.save_to_disk(os.path.join(out_dir, rgb_path))
            depth_img.save_to_disk(os.path.join(out_dir, depth_path), carla.ColorConverter.LogarithmicDepth)
            radar_points = decode_radar(radar_data)
            with open(os.path.join(out_dir, radar_path), "w") as f:
                json.dump(radar_points, f)

            transform = ego_vehicle.get_transform()
            vel = ego_vehicle.get_velocity()
            acc = ego_vehicle.get_acceleration()
            speed = math.hypot(vel.x, vel.y)
            accel_mag = math.hypot(acc.x, acc.y)

            record = {
                "episode_id": episode_id,
                "timestep": t,
                "map": town,
                "weather_preset": weather_name,
                "rgb_frame": rgb_path,
                "depth_frame": depth_path,
                "radar_points": radar_path,
                "ego_state": {
                    "x": transform.location.x, "y": transform.location.y,
                    "theta": math.radians(transform.rotation.yaw),
                    "v_e": speed, "a_e": accel_mag,
                },
                "action": {
                    "throttle": applied_control.throttle,
                    "brake": applied_control.brake,
                    "steer": applied_control.steer,
                },
                "objects": [dataclasses.asdict(o) for o in objects],
                "scenario_type": scenario_type,
                "scenario_name": scenario_name,
            }
            metadata_records.append(record)

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata_records, f)

        frames_with_objects = sum(1 for r in metadata_records if len(r["objects"]) > 0)
        frames_with_in_O = sum(1 for r in metadata_records if any(o["in_O"] for o in r["objects"]))
        ego_end_loc = ego_vehicle.get_transform().location
        ego_displacement = ego_start_loc.distance(ego_end_loc)
        print(f"  [episode {episode_id}] {town}/{weather_name}/{scenario_type}"
              f"{'/' + scenario_name if scenario_name else ''} -> {len(metadata_records)} frames, "
              f"{frames_with_objects} with >=1 object nearby, {frames_with_in_O} with an object in_O")
        closest_str = f"{closest_ever:.1f}m" if closest_ever is not None else "N/A (no actors detected at all)"
        print(f"  [episode {episode_id}] ego displacement over episode: {ego_displacement:.1f}m, "
              f"closest any-actor distance ever seen: {closest_str}")

    finally:
        # destroy in dependency order: sensors -> controllers -> walkers -> vehicles.
        # Getting this wrong (e.g. destroying a walker before its AI controller)
        # throws a C++ std::runtime_error from a background thread that Python's
        # try/except can't catch, crashing the whole process instead of raising
        # a catchable exception -- so order matters here, not just try/except.
        for actor in sensor_actors:
            if not actor.is_alive:
                continue
            try:
                actor.stop()
                actor.destroy()
            except RuntimeError as e:
                print(f"  [episode {episode_id}] warning: failed to destroy sensor {actor.id}: {e}")
        for actor in controller_actors:
            if not actor.is_alive:
                continue
            try:
                actor.stop()
                actor.destroy()
            except RuntimeError as e:
                print(f"  [episode {episode_id}] warning: failed to destroy controller {actor.id}: {e}")
        for actor in walker_actors:
            if not actor.is_alive:
                continue
            try:
                actor.destroy()
            except RuntimeError as e:
                print(f"  [episode {episode_id}] warning: failed to destroy walker {actor.id}: {e}")
        for actor in vehicle_actors:
            if not actor.is_alive:
                continue
            try:
                actor.destroy()
            except RuntimeError as e:
                print(f"  [episode {episode_id}] warning: failed to destroy vehicle {actor.id}: {e}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    # Recovery step: if a previous run crashed hard (e.g. a C++ exception that
    # bypassed Python's finally blocks) while synchronous mode was on, the
    # server can be left hung waiting for a tick from a client that's gone.
    # Forcing synchronous_mode off is a settings change, not a tick, so it
    # gets through even to a stuck server and un-sticks it before we start.
    try:
        recovery_world = client.get_world()
        recovery_settings = recovery_world.get_settings()
        if recovery_settings.synchronous_mode:
            print("Detected synchronous_mode left on from a previous run — resetting before starting.")
            recovery_settings.synchronous_mode = False
            recovery_world.apply_settings(recovery_settings)
    except RuntimeError as e:
        print(f"Warning: couldn't check/reset synchronous mode on connect: {e}")

    combos = list(itertools.product(args.towns, args.weather))
    episode_id = 0

    for town, weather_name in combos:
        print(f"Loading {town} ...")
        world = client.load_world(town)
        carla_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        weather = getattr(carla.WeatherParameters, weather_name, None)
        if weather is None:
            print(f"  WARNING: unknown weather preset '{weather_name}', skipping combo")
            continue
        world.set_weather(weather)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        original_settings = world.get_settings()  # independent snapshot, safe to hold onto
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)

        try:
            for _ in range(args.episodes_per_combo):
                scenario_type = choose_scenario_type(rng, args.scenario_weights)
                scenario_name = rng.choice(scenarios.SCENARIO_NAMES) if scenario_type == "adversarial" else None
                out_dir = os.path.join(args.output, f"episode_{episode_id:04d}")
                run_episode(world, tm, carla_map, blueprint_library, args, episode_id,
                            town, weather_name, scenario_type, scenario_name, out_dir, rng)
                episode_id += 1
        finally:
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)

    print(f"\nDone. {episode_id} episodes written to {args.output}")
    print("Run postprocess.py --dataset <output dir> to get stratification coverage and an episode-level split.")


if __name__ == "__main__":
    sys.exit(main())