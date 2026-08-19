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
        self.susceptibility = np.clip(data["field"], 0.0, 1.0)                  # 0.0-1.0 values
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

        gx = int((x - self.xmin) / self.resolution_x)
        gy = int((y - self.ymin) / self.resolution_y)
        gz = int((z - self.zmin) / self.resolution_z)

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
    Scans the field for ore-boundary candidates; cells within target_boundary of target_susceptibility that are not obstacles. 
    Score is uncertainty minus distance-from-target-ore (information gain minus travel cost).

    Returns (score, point, uncertainty) for the single most interesting remaining ore-boundary location, or None if nothing qualifies.
    """
    def get_frontier(self, cfg, robot_world):

        # Calculate frontier radius
        radius = int(cfg.inspection_preferred_radius_m / min(self.resolution_x, self.resolution_y, self.resolution_z))

        # Convert robot world position to grid position
        gx, gy, gz = self.world_to_voxel(*robot_world)

        # Define local region around robot
        x0 = max(0, gx - radius)
        x1 = min(self.nx, gx + radius + 1)

        y0 = max(0, gy - radius)
        y1 = min(self.ny, gy + radius + 1)

        z0 = max(0, gz - radius)
        z1 = min(self.nz, gz + radius + 1)

        # Select ore field section for frontier discovery 
        susceptibility = self.susceptibility[x0:x1, y0:y1, z0:z1]
        uncertainty = self.uncertainty[x0:x1, y0:y1, z0:z1]

        # Identify indexes that are not obstacles and uncertain of their magnetic susceptibility
        frontier_voxel = np.argwhere((uncertainty == 1) & (susceptibility <= cfg.target_susceptibility))
        if len(frontier_voxel) == 0:  return []

        # Add information gain to planning score
        scores = uncertainty[tuple(frontier_voxel.T)] - np.abs(susceptibility[tuple(frontier_voxel.T)] - cfg.target_susceptibility)

        candidates_voxel = frontier_voxel + np.array([x0, y0, z0])  # Map local indices back to full resolution grid indices
        frontier_world = [self.voxel_to_world(*row) for row in candidates_voxel]  

        # Compute distances between robot position and identified local frontiers
        dists = np.linalg.norm(np.array(frontier_world, dtype=float) - np.array(robot_world, dtype=float), axis=1)

        # Add travel costs to planning score (nearby candidates)
        scores = scores - cfg.inspection_distance_weight * (dists / cfg.inspection_preferred_radius_m)

        # Choose high score frontier candidate
        order = np.argsort(-scores)
        candidates_world = [frontier_world[k] for k in order]
        return None if not candidates_world else candidates_world[0]

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
                        p = self.voxel_to_world(nx, ny, nz)
                        if not self.is_obstacle(*p):
                                neighbors.append((nx, ny, nz))

        return neighbors

    # ----------------------------------------------------
    # Cost
    # ----------------------------------------------------

    def is_obstacle(self, *coords):
        return self.get_susceptibility(*coords) > self.target_susceptibility

    def cost(self, *coords):
        susceptibility = self.get_susceptibility(*coords)
        uncertainty = self.get_uncertainty(*coords)

        if susceptibility > self.target_susceptibility:  # Avoid drilling ore
            return self.w_obstacle
        else:
            return self.w_prospectivity * ((susceptibility - self.target_susceptibility) ** 2) + \
                   self.w_uncertainty * uncertainty