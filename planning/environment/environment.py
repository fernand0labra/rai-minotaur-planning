import numpy as np

class Environment:
    """
    Works for BOTH 2D and 3D environments.
    Automatically detects whether the loaded .npz contains a 2D scalar
    field (Z) or a 3D scalar field (field).
    """

    def __init__(
        self,
        npz_path: str,
        obstacle_threshold=1.8,
        target_ore=1.2,
        boundary_band=0.5,
        w_prospectivity=1.0,
        w_obstacle=1000.0,
        w_uncertainty=0.2,
    ):

        data = np.load(npz_path)
        self.is3d = "field" in data.files

        self.X = data["X"]
        self.Y = data["Y"]
        self.x = data["x"]
        self.y = data["y"]

        if self.is3d:
            self.Z = data["Z"]
            self.field = data["field"]
            self.z = data["z"]

            self.nx, self.ny, self.nz = self.field.shape

            self.zmin = float(self.z.min())
            self.zmax = float(self.z.max())

            self.resolution_z = (self.zmax - self.zmin) / (self.nz - 1)
        else:
            self.field = data["Z"]
            self.ny, self.nx = self.field.shape

        self.xmin = float(self.x.min())
        self.xmax = float(self.x.max())

        self.ymin = float(self.y.min())
        self.ymax = float(self.y.max())

        self.resolution_x = (self.xmax - self.xmin) / (self.nx - 1)
        self.resolution_y = (self.ymax - self.ymin) / (self.ny - 1)

        self.ore_min = float(self.field.min())
        self.ore_max = float(self.field.max())

        self.obstacle_threshold = obstacle_threshold
        self.target_ore = target_ore
        self.boundary_band = boundary_band

        self.w_prospectivity = w_prospectivity
        self.w_obstacle = w_obstacle
        self.w_uncertainty = w_uncertainty

        self.uncertainty = np.ones_like(self.field, dtype=np.float32)

    # ----------------------------------------------------
    # Coordinate conversion
    # ----------------------------------------------------

    def world_to_grid(self, *coords):
        if self.is3d:
            x, y, z = coords

            gx = int((x - self.xmin) / self.resolution_x)
            gy = int((y - self.ymin) / self.resolution_y)
            gz = int((z - self.zmin) / self.resolution_z)

            gx = max(0, min(self.nx - 1, gx))
            gy = max(0, min(self.ny - 1, gy))
            gz = max(0, min(self.nz - 1, gz))
            return gx, gy, gz
        else:
            x, y = coords

            gx = int((x - self.xmin) / self.resolution_x)
            gy = int((y - self.ymin) / self.resolution_y)

            gx = max(0, min(self.nx - 1, gx))
            gy = max(0, min(self.ny - 1, gy))

            return gx, gy

    def grid_to_world(self, *idx):
        if self.is3d:
            gx, gy, gz = idx

            return (
                self.xmin + gx * self.resolution_x,
                self.ymin + gy * self.resolution_y,
                self.zmin + gz * self.resolution_z,
            )
        else:
            gx, gy = idx

            return (
                self.xmin + gx * self.resolution_x,
                self.ymin + gy * self.resolution_y,
            )

    # ----------------------------------------------------
    # Queries
    # ----------------------------------------------------

    def get_ore(self, *coords):
        if self.is3d:
            gx, gy, gz = self.world_to_grid(*coords)
            return float(self.field[gx, gy, gz])
        else:
            gx, gy = self.world_to_grid(*coords)
            return float(self.field[gy, gx])

    def get_uncertainty(self, *coords):
        if self.is3d:
            gx, gy, gz = self.world_to_grid(*coords)
            return float(self.uncertainty[gx, gy, gz])
        else:
            gx, gy = self.world_to_grid(*coords)
            return float(self.uncertainty[gy, gx])

    # ----------------------------------------------------
    # Reveal
    # ----------------------------------------------------

    def reveal_around(self, *coords, radius_m):
        if self.is3d:
            x, y, z = coords
            gx, gy, gz = self.world_to_grid(x, y, z)

            rgx = max(1, int(radius_m / self.resolution_x))
            rgy = max(1, int(radius_m / self.resolution_y))
            rgz = max(1, int(radius_m / self.resolution_z))

            x0 = max(0, gx - rgx)
            x1 = min(self.nx, gx + rgx + 1)

            y0 = max(0, gy - rgy)
            y1 = min(self.ny, gy + rgy + 1)

            z0 = max(0, gz - rgz)
            z1 = min(self.nz, gz + rgz + 1)

            xx, yy, zz = np.ogrid[x0:x1, y0:y1, z0:z1]

            dist2 = (
                ((xx - gx) * self.resolution_x) ** 2
                + ((yy - gy) * self.resolution_y) ** 2
                + ((zz - gz) * self.resolution_z) ** 2
            )
            mask = dist2 <= radius_m**2

            self.uncertainty[x0:x1, y0:y1, z0:z1][mask] = 0
        else:
            x, y = coords

            gx, gy = self.world_to_grid(x, y)

            rgx = max(1, int(radius_m / self.resolution_x))
            rgy = max(1, int(radius_m / self.resolution_y))

            y0 = max(0, gy - rgy)
            y1 = min(self.ny, gy + rgy + 1)

            x0 = max(0, gx - rgx)
            x1 = min(self.nx, gx + rgx + 1)

            yy, xx = np.ogrid[y0:y1, x0:x1]

            dist2 = (
                ((yy - gy) * self.resolution_y) ** 2
                + ((xx - gx) * self.resolution_x) ** 2
            )

            mask = dist2 <= radius_m**2

            self.uncertainty[y0:y1, x0:x1][mask] = 0

    # ----------------------------------------------------
    # Cost
    # ----------------------------------------------------

    def is_obstacle(self, *coords):
        return self.get_ore(*coords) > self.obstacle_threshold

    def cost(self, *coords):

        ore = self.get_ore(*coords)
        uncertainty = self.get_uncertainty(*coords)

        obstacle_penalty = self.w_obstacle if ore > self.obstacle_threshold else 0

        prospectivity = self.w_prospectivity * (
            ((ore - self.target_ore) / max(self.boundary_band, 1e-6)) ** 2
        )

        uncertainty_cost = self.w_uncertainty * uncertainty

        return obstacle_penalty + prospectivity + uncertainty_cost

    # ----------------------------------------------------
    # Neighbours
    # ----------------------------------------------------
    def get_neighbors(self, *idx):
        neighbors = []
        if self.is3d:

            gx, gy, gz = idx

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:

                        if dx == dy == dz == 0:
                            continue

                        nx = gx + dx
                        ny = gy + dy
                        nz = gz + dz

                        if (
                            0 <= nx < self.nx
                            and 0 <= ny < self.ny
                            and 0 <= nz < self.nz
                        ):

                            p = self.grid_to_world(nx, ny, nz)

                            if not self.is_obstacle(*p):
                                neighbors.append((nx, ny, nz))
        else:
            gx, gy = idx

            moves = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

            for dx, dy in moves:

                nx = gx + dx
                ny = gy + dy

                if 0 <= nx < self.nx and 0 <= ny < self.ny:

                    p = self.grid_to_world(nx, ny)

                    if not self.is_obstacle(*p):
                        neighbors.append((nx, ny))

        return neighbors