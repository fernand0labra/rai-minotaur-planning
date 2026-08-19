import numpy as np

from config import parse_config

from world.plot import Plotter3D
from world.env import Environment

from planner.astar import AStarPlanner
from planner.rrt import RRTPlanner

###

PLANNERS = {
    "astar": AStarPlanner,
    "rrt": RRTPlanner,
}

"""
Each steps: ask the field itself for the single most interesting remaining ore-boundary point (highest uncertainty, closest to
target_ore), plan to it, move robot forward, repeat.

Keeps chasing whatever is still unexplored on the boundary until there's nothing left worth chasing
or a safety cap is hit (inspection_max_waypoints). Only then does it head to the actual goal.

Returns the flat list of path points, or None if the goal couldn't be reached.
"""
def main():
    cfg = parse_config()

    env = Environment(
        cfg.map_path,
        target_susceptibility=cfg.target_susceptibility,
        w_prospectivity=cfg.w_prospectivity,
        w_obstacle=cfg.w_obstacle,
        w_uncertainty=cfg.w_uncertainty,
    )

    planner = PLANNERS[cfg.planner](env, cfg)
    start = cfg.start()
    goal = cfg.goal()

    steps = 0
    path = []
    current = start

    reveal_radius = (max(env.resolution_x, env.resolution_y, env.resolution_z) * cfg.inspection_reveal_radius_cells)
    while steps < cfg.inspection_max_waypoints:

        # Obtain candidate frontier based on current position
        frontier = env.get_frontier(cfg, current)
        if frontier is None:
            print("Inspection: no more ore-boundary candidates in the field, heading to goal")
            break

        # Compute path to frontier
        frontier_path, extra = planner.plan(current, frontier)

        if frontier_path is None:
            print("Failed to reach frontier")
            return None

        # Update full robot path with frontier path
        path.extend(frontier_path)

        # Move towards the frontier and obtain sensor readings
        env.robot_step(*frontier, radius_m=reveal_radius)
        current = frontier
        steps += 1

    print(f"Inspection complete after {steps} steps(s), planning final approach to goal")

    # Compute path to goal
    goal_path, extra = planner.plan(current, goal)
    if goal_path is None:
        print("Failed to reach goal from final inspection position")
        return None

    # Update full robot path with goal path
    path.extend(goal_path)
    print("Path points:", len(path))

    # Save planned path for Unity visualization
    np.savetxt("./data/planned_path.csv",  np.asarray(path), delimiter=",", header="x,y,z", comments="", fmt="%.6f")

    # Plot environment and computed path
    Plotter3D(env).draw(path, start=start, goal=goal)

###

if __name__ == "__main__":
    main()