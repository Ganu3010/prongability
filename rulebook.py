"""
Ground-truth per-object and ego-kinematic variable extraction — the raw
quantities a rulebook (yours or anyone else's) would consume: distance,
relative velocity, relative angle, occlusion, and collision, computed
directly from CARLA's simulator state. No config, no rule engine, no
external project dependency — just geometry.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Object list extraction
# ---------------------------------------------------------------------------

@dataclass
class ObjectRecord:
    actor_id: int
    obj_class: str  # "vehicle" / "pedestrian" / "cyclist"
    bbox_3d: list  # 8 world-space corner points [[x,y,z], ...]
    distance: float  # straight-line distance from ego, meters
    signed_velocity: float  # + = closing (approaching), - = receding
    relative_angle_deg: float  # signed angle from ego's forward vector, -180..180, 0 = straight ahead
    occlusion: str  # "none" / "partial" / "heavy"
    colliding: bool  # ego's bounding-box sweep (this tick + previous tick) intersects this actor's bbox
    in_lane: bool  # same road_id + lane_id as ego
    in_front: bool  # positive forward-vector projection
    in_O: bool  # in_lane and in_front (the "relevant object" set most rulebooks care about)
    distance_bucket: str = ""
    velocity_bucket: str = ""


def _classify_actor(actor) -> Optional[str]:
    tid = actor.type_id
    if tid.startswith("vehicle."):
        # CARLA doesn't cleanly separate "cyclist" from "vehicle" in type_id;
        # bicycles/motorcycles have 2 wheels.
        try:
            wheels = int(actor.attributes.get("number_of_wheels", 4))
        except (TypeError, ValueError):
            wheels = 4
        return "cyclist" if wheels == 2 else "vehicle"
    if tid.startswith("walker.pedestrian"):
        return "pedestrian"
    return None


def _distance_bucket(d: float) -> str:
    if d < 5.0:
        return "near"
    elif d < 30.0:
        return "mid"
    return "far"


def _velocity_bucket(signed_v: float, threshold: float = 0.3) -> str:
    if abs(signed_v) < threshold:
        return "stationary"
    return "approaching" if signed_v > 0 else "receding"


def _relative_angle_deg(ego_forward, dx: float, dy: float) -> float:
    """Signed angle (degrees) between ego's forward vector and the vector
    to the actor, in the ground plane. 0 = straight ahead, +90 = directly
    right, -90 = directly left, ±180 = directly behind."""
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    ux, uy = dx / dist, dy / dist
    dot = ego_forward.x * ux + ego_forward.y * uy
    cross = ego_forward.x * uy - ego_forward.y * ux
    return math.degrees(math.atan2(cross, dot))


def _world_to_camera_uv_and_depth(world_point, camera_actor, K, image_w, image_h):
    """Project a world-space point into the depth camera's pixel coordinates.
    Returns (u, v, depth_from_camera) or None if behind the camera / off-frame."""
    cam_transform = camera_actor.get_transform()
    world_to_cam = cam_transform.get_inverse_matrix()
    point = [world_point.x, world_point.y, world_point.z, 1.0]
    cam_point = [
        sum(world_to_cam[row][col] * point[col] for col in range(4))
        for row in range(4)
    ]
    # UE4 camera space -> standard camera space (x=right, y=down, z=forward)
    x, y, z = cam_point[1], -cam_point[2], cam_point[0]
    if z <= 0.05:
        return None
    px = K[0][0] * x / z + K[0][2]
    py = K[1][1] * y / z + K[1][2]
    if px < 0 or px >= image_w or py < 0 or py >= image_h:
        return None
    return px, py, z


def build_camera_intrinsics(image_w: int, image_h: int, fov_deg: float):
    focal = image_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = [[focal, 0.0, image_w / 2.0], [0.0, focal, image_h / 2.0], [0.0, 0.0, 1.0]]
    return K


def estimate_occlusion(actor, depth_image_array, camera_actor, K, image_w, image_h, depth_tolerance=1.5):
    """
    Cross-check the object's 3D bounding-box corners against the depth
    camera to estimate occlusion. depth_image_array must be a 2D array of
    per-pixel depth in meters (decode carla.Image with the raw/linear
    depth conversion before calling this).

    Returns "none" / "partial" / "heavy" based on the fraction of
    projected corners whose expected depth roughly matches the camera's
    recorded depth at that pixel (i.e. not blocked by something closer).
    """
    bbox = actor.bounding_box
    verts = bbox.get_world_vertices(actor.get_transform())
    visible = 0
    total_in_frame = 0
    for v in verts:
        proj = _world_to_camera_uv_and_depth(v, camera_actor, K, image_w, image_h)
        if proj is None:
            continue
        u, v_px, expected_depth = proj
        total_in_frame += 1
        recorded_depth = depth_image_array[int(v_px), int(u)]
        if recorded_depth >= expected_depth - depth_tolerance:
            visible += 1

    if total_in_frame == 0:
        return "heavy"  # entirely out of frame / behind ego -> treat conservatively
    frac_visible = visible / total_in_frame
    if frac_visible > 0.75:
        return "none"
    elif frac_visible > 0.15:
        return "partial"
    return "heavy"


def _detect_collision(ego_bbox, ego_transform, prev_ego_bbox, prev_ego_transform, actor):
    """Sweep the ego's bounding box across this tick and the previous tick
    (catches fast-moving collisions a single-frame overlap test would
    miss), test intersection against the actor's current bbox. Returns
    False if there's no previous-tick state yet (first frame of an
    episode)."""
    if prev_ego_bbox is None or prev_ego_transform is None:
        return False

    target_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    target_polygon = Polygon([[v.x, v.y] for v in target_verts])

    self_verts = ego_bbox.get_world_vertices(ego_transform)
    self_points = [[v.x, v.y] for v in self_verts]
    prev_verts = prev_ego_bbox.get_world_vertices(prev_ego_transform)
    self_points.extend([[v.x, v.y] for v in prev_verts])
    self_polygon = Polygon(self_points).convex_hull

    return self_polygon.intersects(target_polygon)


def closest_actor_distance(world, ego_vehicle) -> Optional[float]:
    """Diagnostic helper: minimum distance to any vehicle/pedestrian/cyclist,
    with NO 80m cutoff — used to distinguish 'nothing ever got close' from
    'things were close but outside the normal extraction radius'."""
    ego_loc = ego_vehicle.get_transform().location
    closest = None
    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue
        if _classify_actor(actor) is None:
            continue
        d = actor.get_location().distance(ego_loc)
        if closest is None or d < closest:
            closest = d
    return closest


def extract_object_list(world, ego_vehicle, depth_image_array, camera_actor, K, image_w, image_h,
                         prev_ego_bbox=None, prev_ego_transform=None) -> List[ObjectRecord]:
    """
    Build the ground-truth object list: every nearby vehicle/pedestrian/
    cyclist (up to 80m) with its distance, relative velocity, relative
    angle, occlusion, and collision status relative to the ego vehicle.

    prev_ego_bbox / prev_ego_transform: the ego's bounding box and
    transform from the *previous* tick, used for collision sweep
    detection. Pass None on the first frame of an episode (collision will
    just be reported as False that frame).
    """
    carla_map = world.get_map()
    ego_transform = ego_vehicle.get_transform()
    ego_forward = ego_transform.get_forward_vector()
    ego_loc = ego_transform.location
    ego_velocity = ego_vehicle.get_velocity()
    ego_wp = carla_map.get_waypoint(ego_loc)
    ego_bbox = ego_vehicle.bounding_box

    records = []
    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue
        obj_class = _classify_actor(actor)
        if obj_class is None:
            continue

        actor_loc = actor.get_transform().location
        dx, dy, dz = actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y, actor_loc.z - ego_loc.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > 80.0:
            continue

        forward_component = dx * ego_forward.x + dy * ego_forward.y
        in_front = forward_component > 0

        actor_wp = carla_map.get_waypoint(actor_loc)
        in_lane = (
            actor_wp is not None
            and ego_wp is not None
            and actor_wp.road_id == ego_wp.road_id
            and actor_wp.lane_id == ego_wp.lane_id
        )
        in_O = bool(in_front and in_lane)

        relative_angle_deg = _relative_angle_deg(ego_forward, dx, dy)

        actor_velocity = actor.get_velocity()
        rel_vx = actor_velocity.x - ego_velocity.x
        rel_vy = actor_velocity.y - ego_velocity.y
        if distance > 1e-3:
            signed_velocity = -(rel_vx * dx + rel_vy * dy) / distance  # + = closing
        else:
            signed_velocity = 0.0

        occlusion = estimate_occlusion(actor, depth_image_array, camera_actor, K, image_w, image_h)
        colliding = _detect_collision(ego_bbox, ego_transform, prev_ego_bbox, prev_ego_transform, actor)

        bbox = actor.bounding_box
        verts = bbox.get_world_vertices(actor.get_transform())
        bbox_3d = [[v.x, v.y, v.z] for v in verts]

        rec = ObjectRecord(
            actor_id=actor.id,
            obj_class=obj_class,
            bbox_3d=bbox_3d,
            distance=distance,
            signed_velocity=signed_velocity,
            relative_angle_deg=relative_angle_deg,
            occlusion=occlusion,
            colliding=bool(colliding),
            in_lane=bool(in_lane),
            in_front=bool(in_front),
            in_O=in_O,
        )
        rec.distance_bucket = _distance_bucket(distance)
        rec.velocity_bucket = _velocity_bucket(signed_velocity)
        records.append(rec)

    return records