import numpy as np
from environment.environment import Environment
from config import parse_config
from planners.astar import AStarPlanner
from planners.hybrid_astar import HybridAStarPlanner
from planners.rrt import RRTPlanner
from planners.rrt_star import RRTStarPlanner
from planners.ore_waypoint import find_best_ore_point
from visualization.plotter import Plotter
from visualization.plotter3d import Plotter3D


PLANNERS = {
    "astar": AStarPlanner,
    "hybrid_astar": HybridAStarPlanner,
    "rrt": RRTPlanner,
    "rrt_star": RRTStarPlanner,
}


def build_environment(cfg):
    return Environment(
        cfg.map_path,
        obstacle_threshold=cfg.obstacle_threshold,
        target_ore=cfg.target_ore,
        boundary_band=cfg.boundary_band,
        w_prospectivity=cfg.w_prospectivity,
        w_obstacle=cfg.w_obstacle,
        w_uncertainty=cfg.w_uncertainty,
    )


def run_inspection(env, cfg, planner, start, goal):
    """
    Cost-function-driven, shape-agnostic inspection loop.

    Each leg: ask the field itself for the single most interesting
    remaining ore-boundary point (highest uncertainty, closest to
    target_ore -- see planners/ore_waypoint.py), plan to it, reveal
    uncertainty around it on arrival, repeat.

    This makes NO assumption about the ore body's shape, size, or
    number of separate bodies -- it just keeps chasing whatever is
    still unexplored on the boundary until there's nothing left worth
    chasing (inspection_uncertainty_threshold) or a safety cap is hit
    (inspection_max_waypoints). Only then does it head to the actual
    goal. This is what "inspect the ore as much as possible before
    reaching goal" means when the shape is unknown: let the cost
    field decide the route, not a fixed geometric pattern.

    Returns the flat list of path points, or None if the goal couldn't
    be reached. Does NOT collect per-leg visited/tree search data --
    that data is only useful for debugging planner internals, not for
    the final plot, and collecting it across dozens of legs is real
    memory/render overhead for no visual benefit once you just want
    "map + final path".
    """
    full_path = []

    reveal_radius = (
        max(env.resolution_x, env.resolution_y) * cfg.inspection_reveal_radius_cells
    )

    current = start
    leg = 0

    while leg < cfg.inspection_max_waypoints:
        candidate = find_best_ore_point(env, cfg)

        if candidate is None:
            print("Inspection: no more ore-boundary candidates in the field, "
                  "heading to goal")
            break

        score, wp, uncertainty = candidate

        if uncertainty <= cfg.inspection_uncertainty_threshold:
            print(f"Inspection: best remaining uncertainty={uncertainty:.3f} "
                  f"<= threshold={cfg.inspection_uncertainty_threshold}, "
                  f"boundary sufficiently inspected, heading to goal")
            break

        print(f"Inspection leg {leg}: -> {wp} "
              f"(score={score:.3f}, uncertainty={uncertainty:.3f})")

        path, extra = planner.plan(current, wp)

        if path is None:
            print(f"  could not reach {wp}, marking revealed to avoid "
                  f"re-selecting an unreachable point, skipping")
            # Reveal it anyway so an unreachable point can't be picked
            # forever and stall the whole inspection loop.
            env.reveal_around(*wp, radius_m=reveal_radius)
            leg += 1
            continue

        full_path.extend(path)

        env.reveal_around(*wp, radius_m=reveal_radius)
        current = wp
        leg += 1

    print(f"Inspection complete after {leg} leg(s), planning final approach to goal")
    path, extra = planner.plan(current, goal)

    if path is None:
        print("Failed to reach goal from final inspection position")
        return None

    full_path.extend(path)

    return full_path


def main():
    cfg = parse_config()
    env = build_environment(cfg)
    print("Environment:", "3D" if env.is3d else "2D")

    planner = PLANNERS[cfg.planner](env, cfg)
    start = cfg.start(env.is3d)
    goal = cfg.goal(env.is3d)

    full_path = run_inspection(env, cfg, planner, start, goal)

    if full_path is None:
        return

    print("Path points:", len(full_path))

    np.save("planned_path.npy", np.asarray(full_path))
    print("Saved planned_path.npy for Unity")

    if env.is3d:
        Plotter3D(env).draw(full_path, start=start, goal=goal)
    else:
        Plotter(env).draw(full_path, start, goal, cfg.planner)


if __name__ == "__main__":
    main()