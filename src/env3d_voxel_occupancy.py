import os
import numpy as np

from skimage import measure

###

def scalar_field_elliptical_3d(
    x, y, z,
    center=(0.0, 0.0, 0.0),
    R1=0.6, R2=1.2, R3=2.0,
    anisotropy=(1.5, 0.8, 1.2),
    warp_strength=0.3,
    irregularity=0.25,
    seed=42,
    smooth=0.05
):
    """
    Generate a smooth 3D scalar field composed of multiple concentric,
    anisotropic, and irregular elliptical shells.

    The resulting scalar field is the sum of three smooth step functions
    centered at radii R1, R2, and R3, with spatial warping and stochastic
    radial irregularity applied.

    Parameters
    ----------
    x, y, z : ndarray
        3D coordinate grids (e.g. from np.meshgrid) defining the sampling
        locations in space.

    center : tuple of float, optional
        (cx, cy, cz) center of the elliptical structure in world coordinates.

    R1, R2, R3 : float, optional
        Base radii of the three concentric shells.
        Larger values create larger layers.

    anisotropy : tuple of float, optional
        (ax, ay, az) scaling factors applied to each axis, producing
        ellipsoidal deformation instead of perfect spheres.

    warp_strength : float, optional
        Strength of sinusoidal spatial warping applied to coordinates,
        introducing smooth, directional distortions.

    irregularity : float, optional
        Amplitude of stochastic radial variation applied to the shell radii,
        controlling surface roughness.

    seed : int, optional
        Random seed for reproducible irregularity patterns.

    smooth : float, optional
        Controls the smoothness of shell boundaries.
        Smaller values create sharper transitions.

    Returns
    -------
    field : ndarray
        3D scalar field where higher values indicate deeper interior regions.
        Thresholding this field yields nested volumetric shells.
    """

    ## Random generator for reproducible irregularity
    rng = np.random.default_rng(seed)

    # Unpack center and anisotropy
    cx, cy, cz = center
    ax, ay, az = anisotropy

    ## Normalize coordinates and apply anisotropy
    dx = (x - cx) / ax
    dy = (y - cy) / ay
    dz = (z - cz) / az

    # Base elliptical radius (before warping)
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    ## Smooth spatial warping using trigonometric noise
    wx = warp_strength * (np.sin(2.5 * dx) + np.cos(1.5 * dy))
    wy = warp_strength * (np.sin(3.1 * dy) - np.cos(2.0 * dz))
    wz = warp_strength * (np.sin(2.0 * dz) + np.cos(1.0 * dx))

    # Warped radius field
    r_warp = np.sqrt(
        (dx + wx)**2 +
        (dy + wy)**2 +
        (dz + wz)**2
    )

    ## Radial irregularity using multi-frequency angular noise
    n_freqs = 5
    freqs = np.arange(1, n_freqs + 1)[:, None, None, None]

    # Random coefficients per frequency
    coeffs1 = rng.normal(0, 1, (n_freqs, 1, 1, 1))
    coeffs2 = rng.normal(0, 1, (n_freqs, 1, 1, 1))

    # Angle-like coordinate (similar to polar inclination)
    theta_like = np.arctan2(np.sqrt(dx**2 + dy**2), dz)

    # Angular noise field
    f = np.sum(
        coeffs1 * np.sin(freqs * theta_like) +
        coeffs2 * np.cos(freqs * theta_like),
        axis=0
    )

    # Normalize irregularity field
    f /= np.max(np.abs(f))

    ## Apply irregularity to shell radii
    R1_field = R1 + irregularity * f
    R2_field = R2 + irregularity * f
    R3_field = R3 + irregularity * f

    ## Smooth step functions defining the three shells
    t1 = 1.0 / (1.0 + np.exp((r_warp - R1_field) / smooth))
    t2 = 1.0 / (1.0 + np.exp((r_warp - R2_field) / smooth))
    t3 = 1.0 / (1.0 + np.exp((r_warp - R3_field) / smooth))

    # Combined scalar field (nested layers)
    return t1 + t2 + t3

###

# --------------------------------------
# Sample 3D grid and extract three non-intersecting meshes
# --------------------------------------

nx=85; ny=85; nz=85
bounds=(-100, 100)
outfile_prefix="env3d_voxel_occupancy"

# Grid
x = np.linspace(bounds[0], bounds[1], nx)
y = np.linspace(bounds[0], bounds[1], ny)
z = np.linspace(bounds[0], bounds[1], nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Area 1
# field = scalar_field_elliptical_3d(X, Y, Z,
#                 center=(10.0, 10.0, 0.0),
#                 R1=75.0, R2=50.0, R3=25.0,
#                 anisotropy=(1.0, 1.0, 1.2),
#                 warp_strength=0.0,
#                 irregularity=15.0,
#                 seed=13, smooth=1.0)

# Area 2
field = scalar_field_elliptical_3d(X, Y, Z,
                center=(5.0, 10.0, 0.0),
                R1=80.0, R2=65.0, R3=25.0,
                anisotropy=(1.0, 1.0, 1.2),
                warp_strength=0.0,
                irregularity=20.0,
                seed=7, smooth=1.0)

# Area 3
# field = scalar_field_elliptical_3d(X, Y, Z,
#                 center=(10.0, 0.0, 5.0),
#                 R1=50.0, R2=30.0, R3=25.0,
#                 anisotropy=(1.0, 2.0, 1.5),
#                 warp_strength=0.0,
#                 irregularity=20.0,
#                 seed=13, smooth=1.0)

# Area 4
# field = scalar_field_elliptical_3d(X, Y, Z,
#     center=(0.0, 10.0, 0.0),
#     R1=45.0, R2=30.0, R3=25.0,
#     anisotropy=(2.0, 1.0, 1.2),
#     warp_strength=0.0,
#     irregularity=5.0,
#     seed=13, smooth=1.0)

###

# --------------------------------------
# Create OBJ mesh with individual discretized voxels
# --------------------------------------

# --- Cube template ---
cube_verts = np.array([ 
    [0,0,0], [1,0,0], [1,1,0], [0,1,0], 
    [0,0,1], [1,0,1], [1,1,1], [0,1,1] 
])

cube_faces = np.array([
    [0,1,2], [0,2,3],   # bottom
    [4,5,6], [4,6,7],   # top
    [0,1,5], [0,5,4],   # front
    [2,3,7], [2,7,6],   # back
    [1,2,6], [1,6,5],   # right
    [3,0,4], [3,4,7]    # left
])

cube_size = 1

nx, ny, nz = len(x), len(y), len(z)
dx = (bounds[1] - bounds[0]) / nx
dy = (bounds[1] - bounds[0]) / ny
dz = (bounds[1] - bounds[0]) / nz

scale = np.array([dx, dy, dz])
scaled_cube = cube_verts * (scale * cube_size)

# --- Define voxel occupancy regions ---
region1 = field > 2.5
region2 = (field > 1.5) & (field <= 2.5)
region3 = (field > 0.5) & (field <= 1.5)
regions = [region1, region2, region3]

outfile = os.path.join(".", "env3d_voxel_occupancy.obj")
with open(outfile, "w") as f:

    vertex_offset = 1  # OBJ is 1-based
    cube_id = 0

    # ===============================
    # Regions
    # ===============================
    for rid, mask in enumerate(regions, start=1):
        print(f"Processing region {rid}...")

        for ix in range(0, nx, cube_size):
            for iy in range(0, ny, cube_size):
                for iz in range(0, nz, cube_size):

                    block = mask[ix:ix+cube_size, iy:iy+cube_size, iz:iz+cube_size]
                    if not np.any(block):
                        continue

                    origin = np.array([x[ix], y[iy], z[iz]])
                    verts = scaled_cube + origin

                    # Write vertices
                    for vx, vy, vz in verts:
                        f.write(f"v {vx} {vy} {vz}\n")

                    # Write group
                    f.write(f"o region_{rid}_cube_{cube_id}\n")

                    # Write faces
                    for a, b, c in cube_faces:
                        f.write(
                            f"f {a + vertex_offset} "
                            f"{b + vertex_offset} "
                            f"{c + vertex_offset}\n"
                        )

                    vertex_offset += 8
                    cube_id += 1

    # ===============================
    # Shell cubes
    # ===============================
    outer_mask = region1 | region2 | region3
    filled_indices = np.argwhere(outer_mask)

    # ======================================================
    # OCCUPANCY MATRIX (matches exported voxel cubes exactly)
    # ======================================================

    grid_x = len(range(0, nx, cube_size))
    grid_y = len(range(0, ny, cube_size))
    grid_z = len(range(0, nz, cube_size))

    occupancy = np.zeros(
        (grid_x, grid_y, grid_z),
        dtype=np.uint8
    )

    for gx, ix in enumerate(range(0, nx, cube_size)):
        for gy, iy in enumerate(range(0, ny, cube_size)):
            for gz, iz in enumerate(range(0, nz, cube_size)):

                block = outer_mask[
                    ix:ix+cube_size,
                    iy:iy+cube_size,
                    iz:iz+cube_size
                ]

                if np.any(block):
                    occupancy[gx, gy, gz] = 1

    print("\n====================================")
    print("Occupancy matrix shape:", occupancy.shape)
    print("Occupied voxels:", occupancy.sum())
    print("====================================\n")

    np.save("env3d_voxel_occupancy.npy", occupancy)
    print("Saved env3d_voxel_occupancy.npy")

    if filled_indices.size > 0:
        min_idx = filled_indices.min(axis=0)
        max_idx = filled_indices.max(axis=0)

        for ix in range(min_idx[0], max_idx[0]+1, cube_size):
            for iy in range(min_idx[1], max_idx[1]+1, cube_size):
                for iz in range(min_idx[2], max_idx[2]+1, cube_size):

                    block = outer_mask[ix:ix+cube_size, iy:iy+cube_size, iz:iz+cube_size]
                    if np.any(block):
                        continue

                    origin = np.array([x[ix], y[iy], z[iz]])
                    verts = scaled_cube + origin

                    for vx, vy, vz in verts:
                        f.write(f"v {vx} {vy} {vz}\n")

                    f.write(f"o shell_cube_{cube_id}\n")

                    for a, b, c in cube_faces:
                        f.write(
                            f"f {a + vertex_offset} "
                            f"{b + vertex_offset} "
                            f"{c + vertex_offset}\n"
                        )

                    vertex_offset += 8
                    cube_id += 1

print(f"Saved OBJ with per-cube groups {outfile}")

# ======================================================
# VISUALIZATION
# ======================================================

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")

ax.voxels(occupancy.astype(bool))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"Voxel Occupancy\n"
    f"Shape={occupancy.shape}\n"
    f"Occupied={occupancy.sum()}"
)

plt.tight_layout()
plt.show()
