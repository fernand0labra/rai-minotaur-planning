import sys
from dataclasses import dataclass, fields


@dataclass
class Config:

    planner: str = "astar"
    map_path: str = "maps/ore_field_2d.npz"

    # -------------------------
    # Start / goal coordinates
    # -------------------------
    start_x: float = -80.0
    start_y: float = -80.0
    start_z: float = 0.0

    goal_x: float = 80.0
    goal_y: float = 80.0
    goal_z: float = 0.0

    # -------------------------
    # Cost map
    # -------------------------
    obstacle_threshold: float = 1.8
    target_ore: float = 1.2
    boundary_band: float = 0.5
    w_prospectivity: float = 1.0
    w_obstacle: float = 1000.0
    w_uncertainty: float = 0.2

    # -------------------------
    # Ore inspection (receding-horizon, shape-agnostic)
    # -------------------------
    # Safety cap: inspection stops after this many waypoints even if
    # the field somehow keeps scoring above threshold everywhere.
    # Exists purely so a bad cost tuning can't produce an infinite
    # inspection loop -- not meant to be the normal stopping reason.
    inspection_max_waypoints: int = 20

    # Normal stopping reason: once the single best remaining candidate
    # has uncertainty at or below this, the boundary is considered
    # sufficiently inspected and the robot proceeds to goal. Lower =
    # more thorough inspection, more legs. Raised from 0.15 -> 0.35:
    # combined with a small reveal radius, 0.15 was accepting "barely
    # still uncertain" leftover patches near already-visited spots as
    # worth another whole leg, producing far more laps than intended.
    inspection_uncertainty_threshold: float = 0.35

    # Radius cleared around each visited inspection waypoint.
    # Expressed as a multiple of grid resolution rather than a fixed
    # meter value, so it scales sensibly across differently-resolved
    # maps. Raised from 8 -> 20: a small radius only clears a thin
    # patch per stop, so with a low threshold the loop kept re-finding
    # nearby unrevealed cells and taking many small extra legs instead
    # of a handful of genuinely spread-out ones.
    inspection_reveal_radius_cells: float = 20.0

    window_ahead_m: float = 50.0
    window_side_m: float = 25.0

    # -------------------------
    # RRT
    # -------------------------
    rrt_step: float = 5.0
    rrt_goal_bias: float = 0.1
    rrt_max_iters: int = 20000
    rrt_star_radius: float = 15.0
    rrt_star_rebuild_interval: int = 25

    # -------------------------
    # Hybrid A*
    # -------------------------
    max_steer_deg: float = 10.0
    turn_penalty: float = 0.05
    hybrid_step_m: float = 3.0
    hybrid_goal_tol_m: float = 4.0
    hybrid_heuristic_weight: float = 4.0
    hybrid_steer_options: int = 5

    astar_heuristic_weight: float = 1.0

    animate: bool = False

    # These are only defaults.
    # main.py will override depending on environment dimension.

    def start(self, is3d=False):
        if is3d:
            return (self.start_x, self.start_y, self.start_z)
        return (self.start_x, self.start_y)

    def goal(self, is3d=False):
        if is3d:
            return (self.goal_x, self.goal_y, self.goal_z)
        return (self.goal_x, self.goal_y)


def parse_config(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    cfg = Config()
    valid_fields = {f.name: f.type for f in fields(cfg)}

    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Bad arg {arg}")
        key, val = arg.split("=", 1)
        if key not in valid_fields:
            raise ValueError(f"Unknown config key {key}")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            parsed = val.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            parsed = int(val)
        elif isinstance(current, float):
            parsed = float(val)
        else:
            parsed = val
        setattr(cfg, key, parsed)
    return cfg