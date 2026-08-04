import numpy as np
from utils.priority_queue import PriorityQueue
from utils.cost_utils import edge_cost, segment_collision_free


class HybridAStarPlanner:
    """
    State = (position, orientation).
      - 2D (env.is3d == False): position=(x,y),   orientation = yaw (rad)
      - 3D (env.is3d == True):  position=(x,y,z), orientation = (yaw, pitch) (rad)

    Discretized to (grid_cell..., yaw_bin[, pitch_bin]) for the
    visited/closed set -- same idea as classic Hybrid A*, just one
    extra discretized axis when running in 3D.

    Motion primitives: cfg.hybrid_steer_options yaw deltas in
    [-max_steer_deg, +max_steer_deg]. In 3D, the same count of pitch
    deltas in [-max_pitch_deg, +max_pitch_deg] (falls back to
    max_steer_deg if cfg.max_pitch_deg isn't set) is added as a
    SEPARATE set of primitives (straight, pure-yaw turns, pure-pitch
    turns) rather than the full yaw x pitch grid, to keep branching
    factor linear instead of quadratic in n_steer.

    Assumes utils/cost_utils.py's edge_cost()/segment_collision_free()
    already work with N-dimensional point tuples (numpy-array-based),
    same as whatever generalization astar.py uses -- if those are
    still 2D-only (a[0]/a[1] indexing), this will break at the first
    3D call and they need the same generalization.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.is3d = bool(getattr(env, "is3d", False))
        self.step = cfg.hybrid_step_m
        self.heading_bins = 24  # 15 deg resolution, used for yaw and (3D) pitch bins

        max_steer = np.radians(cfg.max_steer_deg)
        n_steer = max(3, getattr(cfg, "hybrid_steer_options", 5))
        if n_steer % 2 == 0:
            n_steer += 1  # keep odd so 0.0 (straight) is included exactly
        yaw_deltas = list(np.linspace(-max_steer, max_steer, n_steer))

        if self.is3d:
            self.max_pitch = np.radians(getattr(cfg, "max_pitch_deg", cfg.max_steer_deg))
            pitch_deltas = list(np.linspace(-self.max_pitch, self.max_pitch, n_steer))
        else:
            self.max_pitch = None
            pitch_deltas = [0.0]

        # "+" pattern of primitives: straight, pure-yaw turns, and
        # (3D only) pure-pitch turns...
        primitives = {(0.0, 0.0)}
        for dyaw in yaw_deltas:
            primitives.add((dyaw, 0.0))
        if self.is3d:
            for dpitch in pitch_deltas:
                primitives.add((0.0, dpitch))

            # ...PLUS 4 diagonal primitives (both axes at once, each
            # still within its own max limit). Without these, closing
            # a large simultaneous bearing+depth gap requires
            # alternating pure-yaw/pure-pitch steps -- a spiral
            # instead of a direct approach -- which is what was
            # driving the iteration counts through the roof on long
            # 3D legs.
            max_yaw = yaw_deltas[-1]
            for dyaw in (max_yaw, -max_yaw):
                for dpitch in (self.max_pitch, -self.max_pitch):
                    primitives.add((dyaw, dpitch))

        self.motion_primitives = list(primitives)

    # -------------------------
    # State <-> key helpers
    # -------------------------
    def _heading_bin(self, angle):
        return int(((angle % (2 * np.pi)) / (2 * np.pi)) * self.heading_bins) % self.heading_bins

    def _state_key(self, pos, yaw, pitch):
        grid_idx = self.env.world_to_grid(*pos)
        if self.is3d:
            return (*grid_idx, self._heading_bin(yaw), self._heading_bin(pitch))
        return (*grid_idx, self._heading_bin(yaw))

    def _in_bounds(self, pos):
        if not (self.env.xmin <= pos[0] <= self.env.xmax and
                self.env.ymin <= pos[1] <= self.env.ymax):
            return False
        if self.is3d and not (self.env.zmin <= pos[2] <= self.env.zmax):
            return False
        return True

    def _step_position(self, pos, yaw, pitch):
        if self.is3d:
            dx = self.step * np.cos(pitch) * np.cos(yaw)
            dy = self.step * np.cos(pitch) * np.sin(yaw)
            dz = self.step * np.sin(pitch)
            return (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        dx = self.step * np.cos(yaw)
        dy = self.step * np.sin(yaw)
        return (pos[0] + dx, pos[1] + dy)

    def heuristic(self, pos, goal):
        return float(np.linalg.norm(np.asarray(goal) - np.asarray(pos)))

    # -------------------------
    # Main planning function
    # -------------------------
    def plan(self, start, goal, start_yaw=0.0, start_pitch=0.0):
        goal_tol = self.cfg.hybrid_goal_tol_m
        h_weight = self.cfg.hybrid_heuristic_weight

        start_pos = tuple(start)
        start_key = self._state_key(start_pos, start_yaw, start_pitch)

        open_set = PriorityQueue()
        came_from = {}
        g_cost = {start_key: 0.0}
        best_state = {start_key: (start_pos, start_yaw, start_pitch)}

        open_set.push(h_weight * self.heuristic(start_pos, goal), start_key)

        visited = []
        visited_set = set()

        max_iterations = 200_000
        iterations = 0
        progress_every = 20_000
        best_dist_to_goal = float("inf")

        while not open_set.empty():
            iterations += 1
            if iterations % progress_every == 0:
                print(f"Hybrid A* ({'3D' if self.is3d else '2D'}): {iterations} iterations, "
                      f"{len(visited)} visited, {len(open_set)} in open set, "
                      f"closest approach so far: {best_dist_to_goal:.2f}m")

            if iterations > max_iterations:
                print(f"Hybrid A*: iteration limit ({max_iterations}) hit, no path found "
                      f"(closest approach: {best_dist_to_goal:.2f}m, goal_tol={goal_tol})")
                return None, {"visited": visited}

            key = open_set.pop()
            if key in visited_set:
                continue
            visited_set.add(key)

            pos, yaw, pitch = best_state[key]
            visited.append(pos)

            dist = float(np.linalg.norm(np.asarray(goal) - np.asarray(pos)))
            if dist < best_dist_to_goal:
                best_dist_to_goal = dist

            if dist < goal_tol:
                return self._reconstruct(came_from, key, best_state), {"visited": visited}

            for dyaw, dpitch in self.motion_primitives:
                new_yaw = yaw + dyaw
                new_pitch = pitch + dpitch
                if self.is3d:
                    new_pitch = float(np.clip(new_pitch, -self.max_pitch, self.max_pitch))

                new_pos = self._step_position(pos, new_yaw, new_pitch)

                if not self._in_bounds(new_pos):
                    continue
                if self.env.is_obstacle(*new_pos):
                    continue
                if not segment_collision_free(self.env, pos, new_pos):
                    continue

                nkey = self._state_key(new_pos, new_yaw, new_pitch)
                turn_amount = abs(dyaw) + abs(dpitch)
                step_cost = edge_cost(self.env, pos, new_pos) + \
                    self.cfg.turn_penalty * turn_amount
                tentative_g = g_cost[key] + step_cost

                if nkey not in g_cost or tentative_g < g_cost[nkey]:
                    g_cost[nkey] = tentative_g
                    came_from[nkey] = key
                    best_state[nkey] = (new_pos, new_yaw, new_pitch)
                    f = tentative_g + h_weight * self.heuristic(new_pos, goal)
                    open_set.push(f, nkey)

        print(f"Hybrid A*: open set exhausted, no path found "
              f"(closest approach: {best_dist_to_goal:.2f}m, goal_tol={goal_tol}) "
              f"-- state space fully explored under current motion model")
        return None, {"visited": visited}

    def _reconstruct(self, came_from, key, best_state):
        states = [best_state[key]]
        while key in came_from:
            key = came_from[key]
            states.append(best_state[key])
        states.reverse()
        return [s[0] for s in states]  # position tuples only, same shape as AStarPlanner's path