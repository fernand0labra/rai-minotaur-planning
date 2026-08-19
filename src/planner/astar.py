import heapq
import itertools
import numpy as np

###

class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, next(self._counter), item))

    def pop(self):
        priority, _, item = heapq.heappop(self._heap)
        return item

    def empty(self):
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)

###

class AStarPlanner:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def distance(self, a, b):
        pa = self.env.voxel_to_world(*a)
        pb = self.env.voxel_to_world(*b)
        return float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))

    def plan(self, start, goal):
        visited_world = []
        visited_voxel = set()

        iterations = 0
        max_iterations = (self.env.nx * self.env.ny * self.env.nz)

        # Transform start and goal positions to voxel positions
        start_voxel = self.env.world_to_voxel(*start)
        goal_voxel = self.env.world_to_voxel(*goal)

        came_from = {}
        valid_neighbours = PriorityQueue()
        total_cost = {start_voxel: 0.0}

        # Initial placeholder using start voxel position
        valid_neighbours.push(self.distance(start_voxel, goal_voxel), start_voxel)  # (priority, item)

        # While there are valid neighbours and within the maximum iterations allowed
        while not valid_neighbours.empty() and iterations <= max_iterations:
            iterations += 1

            # Obtain valid non-visited neighbour
            current_voxel = valid_neighbours.pop()
            if current_voxel in visited_voxel:
                continue

            # Mark the location as visited
            visited_voxel.add(current_voxel)
            visited_world.append(self.env.voxel_to_world(*current_voxel))

            # If neighbour is goal position, return path and visitation
            if current_voxel == goal_voxel:
                path = [current_voxel]
        
                # Unravel the path from goal position
                while current_voxel in came_from:
                    current_voxel = came_from[current_voxel]
                    path.append(current_voxel)

                path.reverse()  # Reverse

                return [self.env.voxel_to_world(*p) for p in path], {"visited": visited_world}

            # Iterate over neighbours of current position
            for next_voxel in self.env.get_neighbors(*current_voxel):
                if next_voxel in visited_voxel:
                    continue

                # Transform current and neighbour positions to world frame 
                current_world = self.env.voxel_to_world(*current_voxel)
                next_world = self.env.voxel_to_world(*next_voxel)

                # Update cost with distance to neighbour and environmental factors
                cost = float(np.linalg.norm(np.asarray(next_world) - np.asarray(current_world)) + self.env.cost(*next_world))
                cost += total_cost[current_voxel]

                # If neighbour has not been visited or a "cheaper" path has been found
                if (next_voxel not in total_cost or cost < total_cost[next_voxel]):
                    came_from[next_voxel] = current_voxel
                    total_cost[next_voxel] = cost

                    # Update valid neighbours cost for given neighbour
                    valid_neighbours.push(cost + self.cfg.astar_distance_weight * self.distance(next_voxel, goal_voxel), next_voxel)

        print("A*: no path")
        return None, {"visited": visited_world}
