import numpy as np

from config import parse_config

from world.plot import plot
from world.env import Environment

from planner.astar import AStarPlanner
from planner.rrt import RRTPlanner

###

PLANNERS = {
    "astar": AStarPlanner,
    "rrt": RRTPlanner,
}

def max_turn_angle(path):
    points = np.asarray(path, dtype=float)
    if len(points) < 3:  return 0.0
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = (lengths[:-1] > 1e-12) & (lengths[1:] > 1e-12)
    if not np.any(valid):  return 0.0
    cosine = np.sum(segments[:-1][valid] * segments[1:][valid], axis=1) / (lengths[:-1][valid] * lengths[1:][valid])
    return float(np.degrees(np.max(np.arccos(np.clip(cosine, -1.0, 1.0)))))

def backtrack_to_frontier(env, cfg, route_stack):
    reverse_path = []
    arrival_direction = None
    while len(route_stack) > 1:
        departed = np.asarray(route_stack.pop(), dtype=float)
        current = np.asarray(route_stack[-1], dtype=float)
        arrival_direction = current - departed
        reverse_path.append(tuple(current))
        frontier = env.get_frontier(cfg, current, arrival_direction)
        if frontier is not None:  return reverse_path, tuple(current), arrival_direction, frontier
    return reverse_path, tuple(route_stack[-1]), arrival_direction, None

"""
Each steps: ask the field itself for the single most interesting remaining ore-boundary point (highest uncertainty, well-spaced from
the current position), plan to it, move robot forward, repeat.

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
    route_stack = [start]
    backtrack_points = []
    current = start
    arrival_direction = None
    max_leg_turn_angle = 0.0

    reveal_radius = (max(env.resolution_x, env.resolution_y, env.resolution_z) * cfg.inspection_reveal_radius_cells)
    entry = env.get_entry_frontier(cfg, current)
    if entry is not None and cfg.inspection_max_waypoints > 0:
        staging, frontier = entry
        staging_path, extra = planner.plan(current, staging)
        frontier_path, extra = planner.plan(staging, frontier) if staging_path is not None else (None, {})
        if staging_path is None or frontier_path is None:
            print("Failed to reach gradient-contour entry")
            return None
        path.extend(staging_path)
        path.extend(frontier_path)
        route_stack.extend(tuple(point) for point in staging_path[1:])
        route_stack.extend(tuple(point) for point in frontier_path[1:])
        max_leg_turn_angle = max(max_leg_turn_angle, max_turn_angle(staging_path), max_turn_angle(frontier_path))
        arrival_direction = np.asarray(frontier_path[-1]) - np.asarray(frontier_path[-2])
        env.robot_step(*frontier, radius_m=reveal_radius)
        current = frontier
        steps = 1

    while steps < cfg.inspection_max_waypoints:

        # Obtain candidate frontier based on current position
        frontier = env.get_frontier(cfg, current, arrival_direction)
        if frontier is None:
            reverse_path, current, arrival_direction, frontier = backtrack_to_frontier(env, cfg, route_stack)
            path.extend(reverse_path)
            if frontier is None:
                print("Inspection: no more gradient-frontier candidates along the explored route, heading to goal")
                break
            backtrack_points.append(current)
            print(f"Inspection: backtracked {len(reverse_path)} path point(s) to resume exploration")

        # Compute path to frontier
        frontier_path, extra = planner.plan(current, frontier)

        if frontier_path is None:
            print("Failed to reach frontier")
            return None

        # Update full robot path with frontier path
        path.extend(frontier_path)
        route_stack.extend(tuple(point) for point in frontier_path[1:])
        max_leg_turn_angle = max(max_leg_turn_angle, max_turn_angle(frontier_path))
        if len(frontier_path) >= 2:  arrival_direction = np.asarray(frontier_path[-1]) - np.asarray(frontier_path[-2])

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
    max_leg_turn_angle = max(max_leg_turn_angle, max_turn_angle(goal_path))
    print("Path points:", len(path))
    print(f"Maximum moving turn angle: {max_leg_turn_angle:.2f} degrees")

    # Save planned path for Unity visualization
    np.savetxt("./maps/planned_path.csv",  np.asarray(path), delimiter=",", header="x,y,z", comments="", fmt="%.6f")
    np.savetxt("./maps/backtrack_points.csv", np.asarray(backtrack_points).reshape(-1, 3), delimiter=",", header="x,y,z", comments="", fmt="%.6f")

    # Plot environment and computed path
    plot()

###

if __name__ == "__main__":
    main()
