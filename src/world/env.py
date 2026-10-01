import numpy as np


class Environment:
    """The original voxel environment with incremental sensor revelation."""

    def __init__(self, npz_path: str, target_susceptibility=0.5,
                 w_prospectivity=1.0, w_obstacle=1000.0, w_uncertainty=0.2):
        data = np.load(npz_path)
        self.X, self.Y, self.Z = data["X"], data["Y"], data["Z"]
        self.x, self.y, self.z = data["x"], data["y"], data["z"]
        self.occupancy = np.asarray(data["gradient"], dtype=np.float32)
        self.gradient = np.asarray(data["gradient"], dtype=np.float32)
        self.susceptibility = self.occupancy
        self.uncertainty = np.ones_like(self.susceptibility, dtype=np.float32)
        self.target_susceptibility = target_susceptibility
        self.w_prospectivity = w_prospectivity
        self.w_obstacle = w_obstacle
        self.w_uncertainty = w_uncertainty
        self.nx, self.ny, self.nz = self.susceptibility.shape
        self.xmin, self.xmax = float(self.x.min()), float(self.x.max())
        self.ymin, self.ymax = float(self.y.min()), float(self.y.max())
        self.zmin, self.zmax = float(self.z.min()), float(self.z.max())
        self.resolution_x = (self.xmax - self.xmin) / (self.nx - 1)
        self.resolution_y = (self.ymax - self.ymin) / (self.ny - 1)
        self.resolution_z = (self.zmax - self.zmin) / (self.nz - 1)

    def robot_step(self, *coords, radius_m):
        """Reveal a spherical sensor volume around the robot."""
        gx, gy, gz = self.world_to_voxel(*coords)
        radii = np.maximum(1, np.ceil(radius_m / np.array([
            self.resolution_x, self.resolution_y, self.resolution_z])).astype(int))
        x0, x1 = max(0, gx-radii[0]), min(self.nx, gx+radii[0]+1)
        y0, y1 = max(0, gy-radii[1]), min(self.ny, gy+radii[1]+1)
        z0, z1 = max(0, gz-radii[2]), min(self.nz, gz+radii[2]+1)
        xx, yy, zz = np.ogrid[x0:x1, y0:y1, z0:z1]
        distance2 = (((xx-gx)*self.resolution_x)**2 +
                     ((yy-gy)*self.resolution_y)**2 +
                     ((zz-gz)*self.resolution_z)**2)
        self.uncertainty[x0:x1, y0:y1, z0:z1][distance2 <= radius_m**2] = 0.0

    def reveal_borehole(self, endpoint, collar, radius_m):
        """Reveal measurements along a known borehole."""
        length = np.linalg.norm(np.asarray(collar) - np.asarray(endpoint))
        spacing = max(min(self.resolution_x, self.resolution_y, self.resolution_z), radius_m/2)
        for point in np.linspace(endpoint, collar, max(2, int(np.ceil(length/spacing))+1)):
            self.robot_step(*point, radius_m=radius_m)

    def world_to_voxel(self, *coords):
        point = np.asarray(coords, dtype=float)
        lower = np.array([self.xmin, self.ymin, self.zmin])
        resolution = np.array([self.resolution_x, self.resolution_y, self.resolution_z])
        index = np.rint((point-lower)/resolution).astype(int)
        index = np.clip(index, 0, np.array([self.nx, self.ny, self.nz])-1)
        return tuple(index)

    def voxel_to_world(self, *idx):
        return tuple(np.array([self.xmin, self.ymin, self.zmin]) +
                     np.asarray(idx)*np.array([self.resolution_x, self.resolution_y, self.resolution_z]))

    def get_susceptibility(self, *coords):
        return float(self.susceptibility[self.world_to_voxel(*coords)])

    def get_uncertainty(self, *coords):
        return float(self.uncertainty[self.world_to_voxel(*coords)])

    def susceptibility_gradient(self, *coords):
        """Central-difference vector gradient of magnetic susceptibility."""
        gx, gy, gz = self.world_to_voxel(*coords)
        index = (gx, gy, gz)
        result = np.zeros(3, dtype=float)
        resolutions = (self.resolution_x, self.resolution_y, self.resolution_z)
        limits = (self.nx, self.ny, self.nz)
        for axis in range(3):
            lo, hi = list(index), list(index)
            lo[axis] = max(0, lo[axis]-1)
            hi[axis] = min(limits[axis]-1, hi[axis]+1)
            span = (hi[axis]-lo[axis])*resolutions[axis]
            if span > 0:
                result[axis] = (float(self.susceptibility[tuple(hi)]) -
                                float(self.susceptibility[tuple(lo)])) / span
        return result

    def sample_core_gradient(self, start, end, radius_m, axial_samples=12, radial_samples=16):
        """Sample the surrounding ore field around one forward core segment."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        axis = end-start
        length = np.linalg.norm(axis)
        if length <= 1e-12:
            return np.empty((0, radial_samples), dtype=np.float32)
        axis /= length
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        values = np.empty((axial_samples, radial_samples), dtype=np.float32)
        for i, center in enumerate(np.linspace(start, end, axial_samples)):
            for j, angle in enumerate(np.linspace(0.0, 2*np.pi, radial_samples, endpoint=False)):
                point = self.clip_to_bounds(center+radius_m*(np.cos(angle)*u+np.sin(angle)*v))
                values[i, j] = self.gradient[self.world_to_voxel(*point)]
        return values

    def in_bounds(self, point):
        x, y, z = point
        return (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)

    def clip_to_bounds(self, point):
        return np.clip(point, [self.xmin, self.ymin, self.zmin],
                       [self.xmax, self.ymax, self.zmax])

    def is_obstacle(self, *coords):
        return self.get_susceptibility(*coords) >= self.target_susceptibility

    def get_neighbors(self, *idx):
        neighbors = []
        for delta in np.ndindex(3, 3, 3):
            offset = np.asarray(delta)-1
            if not np.any(offset):
                continue
            candidate = np.asarray(idx)+offset
            if np.all(candidate >= 0) and np.all(candidate < [self.nx, self.ny, self.nz]):
                if self.occupancy[tuple(candidate)] < self.target_susceptibility:
                    neighbors.append(tuple(candidate))
        return neighbors

    def cost(self, *coords):
        susceptibility = self.get_susceptibility(*coords)
        if susceptibility >= self.target_susceptibility:
            return self.w_obstacle
        return (self.w_prospectivity*float(self.gradient[self.world_to_voxel(*coords)]) +
                self.w_uncertainty*self.get_uncertainty(*coords))

    def cost_voxel(self, *idx):
        if self.occupancy[idx] >= self.target_susceptibility:
            return self.w_obstacle
        return (self.w_prospectivity*float(self.gradient[idx]) +
                self.w_uncertainty*float(self.uncertainty[idx]))
