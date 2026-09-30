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
        radius = cfg.astar_motion_radius_cells
        self._scale = np.array([env.resolution_x, env.resolution_y, env.resolution_z])
        self._motion_primitives = [np.array((dx, dy, dz), dtype=int) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1) for dz in range(-radius, radius + 1) if radius - 0.5 <= np.linalg.norm((dx, dy, dz)) <= radius + 0.5]
        self._cos_max_turn = float(np.cos(np.radians(cfg.max_turn_angle_deg)))
        self._allowed_cache = {}

    def distance(self, a, b):
        pa = self.env.voxel_to_world(*a)
        pb = self.env.voxel_to_world(*b)
        return float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))

    def valid_directions(self, incoming):
        if incoming is None:  return self._motion_primitives
        key = tuple(incoming)
        if key not in self._allowed_cache:
            incoming_world = np.asarray(incoming) * self._scale
            incoming_norm = np.linalg.norm(incoming_world)
            self._allowed_cache[key] = [direction for direction in self._motion_primitives if np.dot(incoming_world, direction * self._scale) / (incoming_norm * np.linalg.norm(direction * self._scale)) >= self._cos_max_turn]
        return self._allowed_cache[key]

    def segment_is_free(self, start, end):
        delta = np.asarray(end) - np.asarray(start)
        samples = max(2, int(np.max(np.abs(delta))) * 2 + 1)
        line = np.rint(np.linspace(start, end, samples)).astype(int)
        return bool(np.all(self.env.occupancy[line[:, 0], line[:, 1], line[:, 2]] < self.env.target_susceptibility))

    def turn_is_valid(self, incoming, outgoing):
        if incoming is None:  return True
        incoming_world = np.asarray(incoming) * self._scale
        outgoing_world = np.asarray(outgoing) * self._scale
        cosine = np.dot(incoming_world, outgoing_world) / (np.linalg.norm(incoming_world) * np.linalg.norm(outgoing_world))
        return bool(cosine >= self._cos_max_turn)

    def plan(self, start, goal):
        visited_world = []
        visited_state = set()

        iterations = 0
        max_iterations = (self.env.nx * self.env.ny * self.env.nz)

        # Transform start and goal positions to voxel positions
        start_voxel = self.env.world_to_voxel(*start)
        goal_voxel = self.env.world_to_voxel(*goal)

        if self.segment_is_free(start_voxel, goal_voxel):
            return [self.env.voxel_to_world(*start_voxel), self.env.voxel_to_world(*goal_voxel)], {"visited": [self.env.voxel_to_world(*start_voxel), self.env.voxel_to_world(*goal_voxel)]}

        start_state = (start_voxel, None)
        came_from = {}
        valid_neighbours = PriorityQueue()
        total_cost = {start_state: 0.0}

        # Initial placeholder using start voxel position
        valid_neighbours.push(self.distance(start_voxel, goal_voxel), start_state)  # (priority, item)

        # While there are valid neighbours and within the maximum iterations allowed
        while not valid_neighbours.empty() and iterations <= max_iterations:
            # Obtain valid non-visited neighbour
            current_state = valid_neighbours.pop()
            if current_state in visited_state:
                continue
            current_voxel, incoming = current_state
            iterations += 1

            # Mark the location as visited
            visited_state.add(current_state)
            visited_world.append(self.env.voxel_to_world(*current_voxel))

            # If neighbour is goal position, return path and visitation
            if current_voxel == goal_voxel:
                path = [current_state]
        
                # Unravel the path from goal position
                while current_state in came_from:
                    current_state = came_from[current_state]
                    path.append(current_state)

                path.reverse()  # Reverse
                return [self.env.voxel_to_world(*state[0]) for state in path], {"visited": visited_world}

            # Iterate over neighbours of current position
            directions = self.valid_directions(incoming)
            goal_direction = np.asarray(goal_voxel) - np.asarray(current_voxel)
            if 0 < np.linalg.norm(goal_direction) <= self.cfg.astar_motion_radius_cells + 0.5 and self.turn_is_valid(incoming, goal_direction):
                directions = directions + [goal_direction]

            for direction in directions:
                next_voxel = tuple(np.asarray(current_voxel) + direction)
                if not (0 <= next_voxel[0] < self.env.nx and 0 <= next_voxel[1] < self.env.ny and 0 <= next_voxel[2] < self.env.nz):
                    continue
                next_direction = tuple(int(v) for v in direction)
                next_state = (next_voxel, next_direction)
                if next_state in visited_state or not self.segment_is_free(current_voxel, next_voxel):  continue

                # Update cost with distance to neighbour and environmental factors
                step_distance = float(np.linalg.norm(np.asarray(direction) * self._scale))
                cost = step_distance * (1.0 + self.env.cost_voxel(*next_voxel))
                cost += total_cost[current_state]

                # If neighbour has not been visited or a "cheaper" path has been found
                if (next_state not in total_cost or cost < total_cost[next_state]):
                    came_from[next_state] = current_state
                    total_cost[next_state] = cost

                    # Update valid neighbours cost for given neighbour
                    valid_neighbours.push(cost + self.cfg.astar_distance_weight * self.distance(next_voxel, goal_voxel), next_state)

        print("A*: no path")
        return None, {"visited": visited_world}
