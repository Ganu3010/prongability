"""
Scripted adversarial scenarios for category (c): cut-ins, jaywalking,
hard-braking lead vehicles, and occlusion setups.

NOTE: this implements each scenario directly via the CARLA Python API
rather than through `scenario_runner` / the Leaderboard OpenSCENARIO
library. That keeps this phase (2k-5k frames) self-contained with no
extra framework to install. If you later want richer, standardized
scenario libraries (useful once you scale to tens of thousands of
frames), scenario_runner is the natural upgrade path — these functions
can stay as-is for quick/custom scenarios alongside it.
"""

import random

import carla


def spawn_cut_in_vehicle(world, ego_vehicle, blueprint_library, tm, lead_distance=25.0,
                          adjacent_lane_offset=3.5, trigger_distance=15.0):
    """Spawns a vehicle in the adjacent lane ahead of the ego, which will
    later be commanded to cut in front. Returns the actor; caller should
    poll ego-to-actor distance and call `trigger_cut_in` when appropriate."""
    ego_wp = world.get_map().get_waypoint(ego_vehicle.get_transform().location)
    ahead_wp = ego_wp.next(lead_distance)[0]
    adjacent_wp = ahead_wp.get_right_lane() or ahead_wp.get_left_lane()
    if adjacent_wp is None:
        return None
    bp = random.choice(blueprint_library.filter("vehicle.*"))
    actor = world.try_spawn_actor(bp, adjacent_wp.transform)
    if actor is not None:
        actor.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(actor, 100)
    return actor


def trigger_cut_in(tm, cut_in_actor):
    """Forces the previously spawned adjacent-lane vehicle to change into
    the ego's lane now (simulates a sudden cut-in)."""
    tm.force_lane_change(cut_in_actor, True)  # True = change to the left; use False as needed per lane geometry


def spawn_jaywalker(world, ego_vehicle, blueprint_library, lead_distance=20.0, cross_speed=2.0):
    """Spawns a pedestrian at the roadside near a point ahead of the ego and
    sends it walking across the road, unscripted-crossing style (no
    crosswalk). Returns (walker_actor, controller_actor)."""
    ego_transform = ego_vehicle.get_transform()
    ego_wp = world.get_map().get_waypoint(ego_transform.location)
    ahead_wp = ego_wp.next(lead_distance)[0]

    right_vector = ahead_wp.transform.get_right_vector()
    lane_half_width = ahead_wp.lane_width / 2.0
    spawn_loc = ahead_wp.transform.location + carla.Location(
        x=right_vector.x * (lane_half_width + 2.0), y=right_vector.y * (lane_half_width + 2.0), z=1.0
    )
    target_loc = ahead_wp.transform.location + carla.Location(
        x=-right_vector.x * (lane_half_width + 2.0), y=-right_vector.y * (lane_half_width + 2.0), z=1.0
    )

    walker_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
    walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
    if walker is None:
        return None, None
    controller_bp = blueprint_library.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
    world.tick()
    controller.start()
    controller.go_to_location(target_loc)
    controller.set_max_speed(cross_speed)
    return walker, controller


def spawn_hard_braking_lead(world, ego_vehicle, blueprint_library, tm, lead_distance=20.0):
    """Spawns a lead vehicle directly ahead in the ego's lane, driving
    normally under autopilot. Caller should later call
    `trigger_hard_brake` to force a sudden stop."""
    ego_wp = world.get_map().get_waypoint(ego_vehicle.get_transform().location)
    ahead_wp = ego_wp.next(lead_distance)[0]
    bp = random.choice(blueprint_library.filter("vehicle.*"))
    actor = world.try_spawn_actor(bp, ahead_wp.transform)
    if actor is not None:
        actor.set_autopilot(True, tm.get_port())
    return actor


def trigger_hard_brake(lead_actor):
    """Overrides autopilot for one tick with a hard-brake command. Call
    repeatedly for a few ticks to hold the brake."""
    lead_actor.set_autopilot(False)
    lead_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))


def spawn_occlusion_blocker(world, ego_vehicle, blueprint_library, tm, blocker_distance=12.0,
                             hidden_object_distance=22.0):
    """Spawns a large vehicle close to the ego (the occluder) and a second
    object further ahead in roughly the same visual line, so the depth/RGB
    cameras see a partially-or-fully hidden object. Returns (blocker, hidden_actor)."""
    ego_wp = world.get_map().get_waypoint(ego_vehicle.get_transform().location)

    blocker_wp = ego_wp.next(blocker_distance)[0]
    large_vehicle_bps = [
        bp for bp in blueprint_library.filter("vehicle.*")
        if bp.id in ("vehicle.carlamotors.carlacola", "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2")
    ]
    blocker_bp = random.choice(large_vehicle_bps) if large_vehicle_bps else random.choice(blueprint_library.filter("vehicle.*"))
    blocker = world.try_spawn_actor(blocker_bp, blocker_wp.transform)
    if blocker is not None:
        blocker.set_autopilot(True, tm.get_port())

    hidden_wp = ego_wp.next(hidden_object_distance)[0]
    adjacent_wp = hidden_wp.get_right_lane() or hidden_wp
    hidden_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
    hidden_actor = world.try_spawn_actor(hidden_bp, adjacent_wp.transform)

    return blocker, hidden_actor


SCENARIO_NAMES = ("cut_in", "jaywalk", "hard_brake_lead", "occlusion")
