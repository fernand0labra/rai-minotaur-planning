import sys
from dataclasses import dataclass, fields


def parse_config(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config()
    valid_fields = {field.name for field in fields(cfg)}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Bad argument: {arg!r}; expected key=value")
        key, value = arg.split("=", 1)
        if key not in valid_fields:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            parsed = value.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        else:
            parsed = value
        setattr(cfg, key, parsed)
    return cfg


@dataclass
class Config:
    map_path: str = "./maps/env3d_smooth.npz"
    output_path: str = "./maps/magnetic_navigation_run.npz"
    html_path: str = "magnetic_borehole_navigation.html"
    create_plot: bool = True
    show_gradient: bool = True
    show_core_cylinders: bool = True
    show_core_textures_2d: bool = True
    core_plot_radius_scale: float = 0.5

    # Existing scenario endpoints: the underground ends of borehole 1 and 2.
    start_x: float =  0.75
    start_y: float = -0.75
    start_z: float = -0.9
    goal_x:  float = -0.5
    goal_y:  float =  0.5
    goal_z:  float =  0.0

    # Borehole collars are at the top of the mapped volume unless explicitly set.
    borehole_radius_m: float = 0.035

    # One sensor/control update for every 10 cm of forward movement.
    step_length_m: float = 0.075
    reveal_radius_m: float = 0.05
    goal_tolerance_m: float = 0.1
    max_steps: int = 1000

    # MATLAB magnetic_tracking.m law: normal regulation + constant tangent motion.
    target_susceptibility: float = 0.8
    ore_threshold: float = 0.8
    susceptibility_gain: float = 0.9
    tangential_speed: float = 0.35
    gradient_epsilon: float = 1e-8

    # Existing steering geometry/controller integration settings.
    alpha_deg: float = 0.38
    controller_dt: float = 0.01
    controller_iterations: int = 50
    controller_direction_tolerance: float = 1e-2
    controller_rate_tolerance: float = 1e-3

    def start(self):
        return (self.start_x, self.start_y, self.start_z)

    def goal(self):
        return (self.goal_x, self.goal_y, self.goal_z)
