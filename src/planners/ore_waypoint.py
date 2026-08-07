import numpy as np


def _scan_candidates(env, cfg, stride=None, current=None):
    """
    Scans the field for ore-boundary candidates -- cells within
    boundary_band of target_ore that are not obstacles. Score is
    uncertainty minus distance-from-target-ore, minus (if `current`
    is given) a distance-from-robot penalty -- so the best remaining
    candidate is "unexplored, on the boundary, AND nearby", not just
    the first two. This is what keeps legs local instead of jumping
    across the whole map every time several points tie on
    uncertainty/ore-match alone (which is common early on, when most
    of a large 3D volume is still uncertainty=1).

    Passing current=None (the default) reproduces the exact old
    behavior -- no distance term at all -- so nothing changes unless
    a caller opts in by passing the robot's current position.

    Fully shape-agnostic: this makes no assumption about the ore
    body's geometry (no fitted centroid/radius/orbit). It just reads
    the field directly, so it works the same whether the body is a
    blob, a vein, a branching structure, or several disconnected
    bodies. Adding a distance-from-robot term doesn't change that --
    it's still driven entirely by field values, just now including
    "how far is this from where I am" as one more field-derived
    quantity, exactly like a frontier-exploration utility function
    (information gain minus travel cost).

    Vectorized: filtering is a single numpy boolean-mask pass over
    the (strided) field; only surviving candidates get converted to
    world coordinates and (if requested) get a distance computed.
    """
    is3d = env.is3d

    if stride is None:
        target_samples_per_axis = 120 if is3d else 400
        stride = max(1, env.nx // target_samples_per_axis)

    if is3d:
        field = env.field[::stride, ::stride, ::stride]
        uncertainty = env.uncertainty[::stride, ::stride, ::stride]
    else:
        field = env.field[::stride, ::stride]
        uncertainty = env.uncertainty[::stride, ::stride]

    band = cfg.boundary_band * 2
    on_boundary = np.abs(field - cfg.target_ore) < band
    not_obstacle = field <= cfg.obstacle_threshold
    valid = on_boundary & not_obstacle

    idx = np.argwhere(valid)
    if len(idx) == 0:
        return []

    idx_full = idx * stride  # map strided indices back to full-res grid indices

    ore_vals = field[tuple(idx.T)]
    unc_vals = uncertainty[tuple(idx.T)]

    if is3d:
        # env.field shape is (nx, ny, nz) -> idx rows are (gx, gy, gz),
        # matching grid_to_world's argument order directly
        world_pts = [env.grid_to_world(*row) for row in idx_full]
    else:
        # env.field shape is (ny, nx) -> idx rows are (gy, gx), so
        # grid_to_world needs them swapped to (gx, gy)
        world_pts = [env.grid_to_world(row[1], row[0]) for row in idx_full]

    scores = unc_vals - np.abs(ore_vals - cfg.target_ore)

    if current is not None:
        world_pts_arr = np.array(world_pts, dtype=float)
        current_arr = np.array(current, dtype=float)
        dists = np.linalg.norm(world_pts_arr - current_arr, axis=1)

        # normalized by a "preferred lookahead radius" (spec: ~50m
        # ahead) rather than an absolute meter penalty, so the weight
        # doesn't need re-tuning if the map scale changes
        distance_weight = getattr(cfg, "inspection_distance_weight", 1.0)
        preferred_radius = getattr(cfg, "inspection_preferred_radius_m", 50.0)
        scores = scores - distance_weight * (dists / max(preferred_radius, 1e-6))

    order = np.argsort(-scores)

    candidates = [
        (float(scores[k]), world_pts[k], float(unc_vals[k]))
        for k in order
    ]
    return candidates


def find_best_ore_point(env, cfg, stride=None, current=None):
    """
    Returns (score, point, uncertainty) for the single most interesting
    remaining ore-boundary location, or None if nothing qualifies.

    Call this ONCE PER LEG, not once up front. reveal_around() zeroes
    uncertainty near any point already visited, so already-inspected
    boundary cells stop scoring well and the best remaining candidate
    naturally shifts to unexplored territory.

    Pass current=<robot's current (x,y[,z])> to also prefer nearby
    candidates over far ones when they'd otherwise tie -- this is
    optional and purely additive; omitting it reproduces the old
    global (distance-blind) selection exactly.
    """
    candidates = _scan_candidates(env, cfg, stride=stride, current=current)
    if not candidates:
        return None
    return candidates[0]