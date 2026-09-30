import sys

from dataclasses import dataclass, fields

###

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
        if   isinstance(current, bool):   parsed = val.lower() in ("1", "true", "yes")
        elif isinstance(current, int):    parsed = int(val)
        elif isinstance(current, float):  parsed = float(val)
        else:                             parsed = val

        setattr(cfg, key, parsed)

    return cfg

###

@dataclass
class Config:

    def start(self):
        return (self.start_x, self.start_y, self.start_z)

    def goal(self):
        return (self.goal_x, self.goal_y, self.goal_z)

    planner: str = "astar"
    map_path: str = "./maps/env3d_smooth.npz"

    # -------------------------
    # Start / goal coordinates
    # -------------------------
    start_x: float = 0.9
    start_y: float = -0.9
    start_z: float = -0.9

    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_z: float = 0.9

    # -------------------------
    # Cost map
    # -------------------------

    target_susceptibility: float = 0.5
    target_gradient: float = 0.8
    frontier_tolerance: float = 0.05

    w_prospectivity: float = 1.0
    w_obstacle: float = 1000.0
    w_uncertainty: float = 0.2

    # -------------------------
    # Ore inspection
    # -------------------------

    inspection_max_waypoints: int = 150  # Planning Safety cap
    inspection_reveal_radius_cells: float = 5.0  # Radius cleared around each visited inspection waypoint
    inspection_distance_weight: float = 1.0
    inspection_preferred_radius_m: float = 0.05
    inspection_max_frontier_distance_m: float = 0.3
    inspection_max_turn_angle_deg: float = 45.0
    inspection_entry_distance_m: float = 0.2
    inspection_heading_weight: float = 0.5

    # -------------------------
    # A*
    # -------------------------

    astar_distance_weight: float = 2.0
    max_turn_angle_deg: float = 10.0
    astar_motion_radius_cells: int = 6

    # -------------------------
    # RRT
    # -------------------------
    rrt_step: float = 0.05
    rrt_goal_bias: float = 0.1
    rrt_max_iters: int = 20000
    rrt_star_radius: float = 0.15
    rrt_star_rebuild_interval: int = 25
