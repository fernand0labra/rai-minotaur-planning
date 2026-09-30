import numpy as np

###

class Environment:

    def __init__(self, npz_path: str, target_susceptibility=0.8, w_prospectivity=1.0, w_obstacle=1000.0, w_uncertainty=0.2):

        data = np.load(npz_path)

        self.X = data["X"]
        self.Y = data["Y"]
        self.Z = data["Z"]

        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]

        # TODO Update from csv
        self.occupancy = np.asarray(data["occupancy"], dtype=np.float32)
        self.gradient = np.asarray(data["gradient"], dtype=np.float32)
        self.susceptibility = self.occupancy                                      # 0.0-1.0 values
        self.uncertainty = np.ones_like(self.susceptibility, dtype=np.float32)  # 0,1 values

        self.target_susceptibility = target_susceptibility

        self.w_prospectivity = w_prospectivity
        self.w_uncertainty = w_uncertainty
        self.w_obstacle = w_obstacle

        self.nx, self.ny, self.nz = self.susceptibility.shape

        self.xmin = float(self.x.min())
        self.xmax = float(self.x.max())

        self.ymin = float(self.y.min())
        self.ymax = float(self.y.max())

        self.zmin = float(self.z.min())
        self.zmax = float(self.z.max())

        self.resolution_x = (self.xmax - self.xmin) / (self.nx - 1)
        self.resolution_y = (self.ymax - self.ymin) / (self.ny - 1)
        self.resolution_z = (self.zmax - self.zmin) / (self.nz - 1)

    # ----------------------------------------------------
    # Robot simulation
    # ----------------------------------------------------

    def robot_step(self, *coords, radius_m):
        x, y, z = coords
        gx, gy, gz = self.world_to_voxel(x, y, z)

        # Compute scaled radius
        rgx = max(1, int(radius_m / self.resolution_x))
        rgy = max(1, int(radius_m / self.resolution_y))
        rgz = max(1, int(radius_m / self.resolution_z))

        # Obtain minimum/maximum axis bounds
        x0 = max(0, gx - rgx)
        x1 = min(self.nx, gx + rgx + 1)

        y0 = max(0, gy - rgy)
        y1 = min(self.ny, gy + rgy + 1)

        z0 = max(0, gz - rgz)
        z1 = min(self.nz, gz + rgz + 1)

        # Obtain position coordinates between axis bounds
        xx, yy, zz = np.ogrid[x0:x1, y0:y1, z0:z1]

        # Compute the squared distance from frontier to bounded positions
        distance2 = (
            ((xx - gx) * self.resolution_x) ** 2 + \
            ((yy - gy) * self.resolution_y) ** 2 + \
            ((zz - gz) * self.resolution_z) ** 2
        )

        # Consider points whose distance remains at a radius_m sphere from the frontier
        mask = distance2 <= radius_m**2

        # Set visible points to zero uncertainty
        self.uncertainty[x0:x1, y0:y1, z0:z1][mask] = 0

    # ----------------------------------------------------
    # Coordinate conversion
    # ----------------------------------------------------

    # TODO Update to voxel size not fixed size
    def world_to_voxel(self, *coords):
        x, y, z = coords

        gx = int(np.rint((x - self.xmin) / self.resolution_x))
        gy = int(np.rint((y - self.ymin) / self.resolution_y))
        gz = int(np.rint((z - self.zmin) / self.resolution_z))

        gx = max(0, min(self.nx - 1, gx))
        gy = max(0, min(self.ny - 1, gy))
        gz = max(0, min(self.nz - 1, gz))

        return gx, gy, gz

    def voxel_to_world(self, *idx):
        gx, gy, gz = idx

        return (
            self.xmin + gx * self.resolution_x,
            self.ymin + gy * self.resolution_y,
            self.zmin + gz * self.resolution_z,
        )

    # ----------------------------------------------------
    # Queries
    # ----------------------------------------------------

    def get_susceptibility(self, *coords):
        gx, gy, gz = self.world_to_voxel(*coords)
        return float(self.susceptibility[gx, gy, gz])

    def get_uncertainty(self, *coords):
        gx, gy, gz = self.world_to_voxel(*coords)
        return float(self.uncertainty[gx, gy, gz])

    """
    Scans the field for gradient-boundary candidates; free cells directly outside the target gradient contour. 
    Score is uncertainty minus distance-from-target-gradient (information gain minus travel cost).

    Returns the single most interesting remaining ore-boundary location, or None if nothing qualifies.
    """
    def get_frontier(self, cfg, robot_world, arrival_direction=None):

        # Identify indexes that are not obstacles and uncertain of their magnetic susceptibility
        cache_key = (cfg.target_susceptibility, cfg.target_gradient)
        if getattr(self, "_frontier_cache_key", None) != cache_key:
            self._frontier_cache_key = cache_key
            occupied = self.occupancy >= cfg.target_susceptibility
            tracked = self.gradient >= cfg.target_gradient
            adjacent = np.zeros_like(tracked)
            adjacent[1:, :, :] |= tracked[:-1, :, :]
            adjacent[:-1, :, :] |= tracked[1:, :, :]
            adjacent[:, 1:, :] |= tracked[:, :-1, :]
            adjacent[:, :-1, :] |= tracked[:, 1:, :]
            adjacent[:, :, 1:] |= tracked[:, :, :-1]
            adjacent[:, :, :-1] |= tracked[:, :, 1:]
            self._frontier_voxel = np.argwhere((~occupied) & (~tracked) & adjacent)

        frontier_voxel = self._frontier_voxel
        frontier_voxel = frontier_voxel[self.uncertainty[tuple(frontier_voxel.T)] > 0]
        if len(frontier_voxel) == 0:  return None

        # Add information gain to planning score
        tracking_gradient = self.gradient[tuple(frontier_voxel.T)]
        uncertainty = self.uncertainty[tuple(frontier_voxel.T)]
        scores = uncertainty - np.abs(tracking_gradient - cfg.target_gradient)

        frontier_world = np.empty_like(frontier_voxel, dtype=float)
        frontier_world[:, 0] = self.xmin + frontier_voxel[:, 0] * self.resolution_x
        frontier_world[:, 1] = self.ymin + frontier_voxel[:, 1] * self.resolution_y
        frontier_world[:, 2] = self.zmin + frontier_voxel[:, 2] * self.resolution_z

        # Compute distances between robot position and identified frontiers
        dists = np.linalg.norm(frontier_world - np.array(robot_world, dtype=float), axis=1)

        # Do not jump to a separate gradient island after the initial approach
        if arrival_direction is not None:
            local = dists <= cfg.inspection_max_frontier_distance_m
            frontier_voxel = frontier_voxel[local]
            frontier_world = frontier_world[local]
            scores = scores[local]
            dists = dists[local]
            if len(frontier_voxel) == 0:  return None

        # Add travel costs to planning score (nearby surface-following candidates)
        scores = scores - cfg.inspection_distance_weight * (dists / cfg.inspection_preferred_radius_m)
        if arrival_direction is not None:
            arrival_direction = np.asarray(arrival_direction, dtype=float)
            arrival_norm = np.linalg.norm(arrival_direction)
            candidate_directions = frontier_world - np.asarray(robot_world, dtype=float)
            candidate_norms = np.linalg.norm(candidate_directions, axis=1)
            valid = candidate_norms > 1e-12
            heading_alignment = np.zeros_like(scores)
            heading_alignment[valid] = np.sum(candidate_directions[valid] * arrival_direction, axis=1) / (candidate_norms[valid] * arrival_norm)
            within_turn = valid & (heading_alignment >= np.cos(np.radians(cfg.inspection_max_turn_angle_deg)))
            frontier_voxel = frontier_voxel[within_turn]
            frontier_world = frontier_world[within_turn]
            scores = scores[within_turn]
            dists = dists[within_turn]
            heading_alignment = heading_alignment[within_turn]
            if len(frontier_voxel) == 0:  return None
            scores = scores + cfg.inspection_heading_weight * heading_alignment

        # Choose high score frontier candidate
        for index in np.argsort(-scores):
            if self.line_is_free(robot_world, frontier_voxel[index]):  return tuple(frontier_world[index])
        return tuple(frontier_world[int(np.argmin(dists))])

    def get_entry_frontier(self, cfg, robot_world):
        self.get_frontier(cfg, robot_world)
        frontier_voxel = self._frontier_voxel
        frontier_voxel = frontier_voxel[self.uncertainty[tuple(frontier_voxel.T)] > 0]
        if len(frontier_voxel) < 2:  return None

        frontier_world = np.empty_like(frontier_voxel, dtype=float)
        frontier_world[:, 0] = self.xmin + frontier_voxel[:, 0] * self.resolution_x
        frontier_world[:, 1] = self.ymin + frontier_voxel[:, 1] * self.resolution_y
        frontier_world[:, 2] = self.zmin + frontier_voxel[:, 2] * self.resolution_z
        start_distances = np.linalg.norm(frontier_world - np.asarray(robot_world, dtype=float), axis=1)

        for entry_index in np.argsort(start_distances)[:512]:
            directions = frontier_world - frontier_world[entry_index]
            distances = np.linalg.norm(directions, axis=1)
            local = (distances > cfg.inspection_reveal_radius_cells * min(self.resolution_x, self.resolution_y, self.resolution_z)) & (distances <= cfg.inspection_max_frontier_distance_m)
            next_indices = np.flatnonzero(local)
            next_indices = next_indices[np.argsort(np.abs(distances[next_indices] - cfg.inspection_preferred_radius_m))]

            for next_index in next_indices[:64]:
                tracking_direction = directions[next_index] / distances[next_index]
                staging_world = frontier_world[entry_index] - cfg.inspection_entry_distance_m * tracking_direction
                if not (self.xmin <= staging_world[0] <= self.xmax and self.ymin <= staging_world[1] <= self.ymax and self.zmin <= staging_world[2] <= self.zmax):  continue
                staging_voxel = self.world_to_voxel(*staging_world)
                staging_world = np.asarray(self.voxel_to_world(*staging_voxel))
                arrival_direction = frontier_world[entry_index] - staging_world
                cosine = np.dot(arrival_direction, directions[next_index]) / (np.linalg.norm(arrival_direction) * distances[next_index])
                if cosine < np.cos(np.radians(cfg.inspection_max_turn_angle_deg)):  continue
                if self.occupancy[staging_voxel] >= self.target_susceptibility:  continue
                if not self.line_is_free(staging_world, frontier_voxel[entry_index]):  continue
                return tuple(staging_world), tuple(frontier_world[entry_index])
        return None

    def line_is_free(self, start, goal_voxel):
        start_voxel = np.rint((np.asarray(start) - np.array([self.xmin, self.ymin, self.zmin])) / np.array([self.resolution_x, self.resolution_y, self.resolution_z])).astype(int)
        delta = np.asarray(goal_voxel) - start_voxel
        samples = max(2, int(np.max(np.abs(delta))) * 2 + 1)
        line = np.rint(np.linspace(start_voxel, goal_voxel, samples)).astype(int)
        line[:, 0] = np.clip(line[:, 0], 0, self.nx - 1)
        line[:, 1] = np.clip(line[:, 1], 0, self.ny - 1)
        line[:, 2] = np.clip(line[:, 2], 0, self.nz - 1)
        return bool(np.all(self.occupancy[line[:, 0], line[:, 1], line[:, 2]] < self.target_susceptibility))

    def get_neighbors(self, *idx):
        neighbors = []
        gx, gy, gz = idx

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:

                    if dx == dy == dz == 0:  continue

                    nx = gx + dx
                    ny = gy + dy
                    nz = gz + dz

                    if (0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz):
                        if self.occupancy[nx, ny, nz] < self.target_susceptibility:
                            neighbors.append((nx, ny, nz))

        return neighbors

    # ----------------------------------------------------
    # Cost
    # ----------------------------------------------------

    def is_obstacle(self, *coords):
        gx, gy, gz = self.world_to_voxel(*coords)
        return self.occupancy[gx, gy, gz] >= self.target_susceptibility

    def cost(self, *coords):
        gx, gy, gz = self.world_to_voxel(*coords)
        susceptibility = float(self.occupancy[gx, gy, gz])
        uncertainty = self.get_uncertainty(*coords)

        if susceptibility >= self.target_susceptibility:  # Avoid drilling ore
            return self.w_obstacle
        else:
            return self.w_prospectivity * float(self.gradient[gx, gy, gz]) + \
                   self.w_uncertainty * uncertainty

    def cost_voxel(self, *idx):
        gx, gy, gz = idx
        if self.occupancy[gx, gy, gz] >= self.target_susceptibility:  return self.w_obstacle
        return self.w_prospectivity * float(self.gradient[gx, gy, gz]) + self.w_uncertainty * float(self.uncertainty[gx, gy, gz])
