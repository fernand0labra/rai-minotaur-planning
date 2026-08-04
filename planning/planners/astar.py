from utils.priority_queue import PriorityQueue
from utils.cost_utils import edge_cost
import numpy as np

class AStarPlanner:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def heuristic(self, a, b):
        pa = self.env.grid_to_world(*a)
        pb = self.env.grid_to_world(*b)
        return float(
            np.linalg.norm(
                np.asarray(pb) - np.asarray(pa)
            )
        )

    def plan(self, start, goal):
        start_g = self.env.world_to_grid(*start)
        goal_g = self.env.world_to_grid(*goal)
        open_set = PriorityQueue()
        came_from = {}
        g_cost = {
            start_g: 0.0
        }

        open_set.push(
            self.heuristic(start_g, goal_g),
            start_g
        )

        visited = []
        visited_set = set()

        iterations = 0
        max_iterations = (
            self.env.nx *
            self.env.ny *
            (self.env.nz if self.env.is3d else 1)
        )

        while not open_set.empty():
            iterations += 1
            if iterations > max_iterations:
                print("A*: iteration limit")
                return None, {"visited": visited}
            current = open_set.pop()

            if current in visited_set:
                continue

            visited_set.add(current)
            visited.append(self.env.grid_to_world(*current))

            if current == goal_g:
                return (
                    self.reconstruct(
                        came_from,
                        current
                    ),
                    {
                        "visited": visited
                    }
                )

            for n in self.env.get_neighbors(*current):
                if n in visited_set:
                    continue

                current_world = self.env.grid_to_world(*current)
                next_world = self.env.grid_to_world(*n)

                cost = edge_cost(self.env, current_world, next_world)

                tentative = (g_cost[current] + cost)

                if (
                    n not in g_cost
                    or tentative < g_cost[n]
                ):
                    came_from[n] = current
                    g_cost[n] = tentative

                    f = (tentative + self.cfg.astar_heuristic_weight * self.heuristic(n, goal_g))
                    open_set.push(f, n)

        print("A*: no path")

        return None, {
            "visited": visited
        }

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return [
            self.env.grid_to_world(*p)
            for p in path
        ]