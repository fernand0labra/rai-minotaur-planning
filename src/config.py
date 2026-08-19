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
    map_path: str = "./data/magnetic_sensor.csv"

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

    target_susceptibility: float = 0.5

    w_prospectivity: float = 1.0
    w_obstacle: float = 1000.0
    w_uncertainty: float = 0.2

    # -------------------------
    # Ore inspection
    # -------------------------

    inspection_max_waypoints: int = 100  # Planning Safety cap
    inspection_reveal_radius_cells: float = 10.0  # Radius cleared around each visited inspection waypoint
    inspection_distance_weight: float = 1.0
    inspection_preferred_radius_m: float = 50.0

    # -------------------------
    # A*
    # -------------------------

    astar_distance_weight: float = 1.0

    # -------------------------
    # RRT
    # -------------------------
    rrt_step: float = 5.0
    rrt_goal_bias: float = 0.1
    rrt_max_iters: int = 20000
    rrt_star_radius: float = 15.0
    rrt_star_rebuild_interval: int = 25