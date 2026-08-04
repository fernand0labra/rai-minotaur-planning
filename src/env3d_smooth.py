import os
import numpy as np
from skimage import measure

# --------------------------------------
# 3D extension of the scalar field
# --------------------------------------

def scalar_field_elliptical_3d(x, y, z, center=(0.0, 0.0, 0.0),
                               R1=0.6, R2=1.2, R3=2.0,
                               anisotropy=(1.5, 0.8, 1.2),
                               warp_strength=0.3,
                               irregularity=0.25,
                               seed=42, smooth=0.05):
    rng = np.random.default_rng(seed)
    cx, cy, cz = center
    ax, ay, az = anisotropy

    dx = (x - cx) / ax
    dy = (y - cy) / ay
    dz = (z - cz) / az

    r = np.sqrt(dx**2 + dy**2 + dz**2)

    wx = warp_strength * np.sin(2.5 * dx) + warp_strength * np.cos(1.5 * dy)
    wy = warp_strength * np.sin(3.1 * dy) - warp_strength * np.cos(2.0 * dz)
    wz = warp_strength * np.sin(2.0 * dz) + warp_strength * np.cos(1.0 * dx)

    r_warp = np.sqrt((dx + wx)**2 + (dy + wy)**2 + (dz + wz)**2)

    n_freqs = 5
    freqs = np.arange(1, n_freqs + 1)[:, None, None, None]
    coeffs1 = rng.normal(0, 1, (n_freqs, 1, 1, 1))
    coeffs2 = rng.normal(0, 1, (n_freqs, 1, 1, 1))

    theta_like = np.arctan2(np.sqrt(dx**2 + dy**2), dz)

    f = np.sum(coeffs1 * np.sin(freqs * theta_like) +
               coeffs2 * np.cos(freqs * theta_like), axis=0)
    f /= np.max(np.abs(f))

    R1_field = R1 + irregularity * f
    R2_field = R2 + irregularity * f
    R3_field = R3 + irregularity * f

    t1 = 1.0 / (1.0 + np.exp((r_warp - R1_field) / smooth))
    t2 = 1.0 / (1.0 + np.exp((r_warp - R2_field) / smooth))
    t3 = 1.0 / (1.0 + np.exp((r_warp - R3_field) / smooth))

    return t1 + t2 + t3

# --------------------------------------
# Sample 3D grid and extract three non-intersecting meshes
# --------------------------------------

nx=200; ny=200; nz=200
bounds=(-100, 100)
outfile_prefix="layer"

# Grid
x = np.linspace(bounds[0], bounds[1], nx)
y = np.linspace(bounds[0], bounds[1], ny)
z = np.linspace(bounds[0], bounds[1], nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Area 1
field = scalar_field_elliptical_3d(X, Y, Z,
                center=(10.0, 10.0, 0.0),
                R1=75.0, R2=50.0, R3=25.0,
                anisotropy=(1.0, 1.0, 1.2),
                warp_strength=0.0,
                irregularity=15.0,
                seed=13, smooth=1.0)
os.makedirs("maps", exist_ok=True)

np.savez(
    "maps/ore_field_3d.npz",
    X=X,
    Y=Y,
    Z=Z,
    x=x,
    y=y,
    z=z,
    field=field
)

print("Saved planner environment")

# Area 2
# field = scalar_field_elliptical_3d(X, Y, Z,
#                 center=(5.0, 10.0, 0.0),
#                 R1=80.0, R2=65.0, R3=25.0,
#                 anisotropy=(1.0, 1.0, 1.2),
#                 warp_strength=0.0,
#                 irregularity=20.0,
#                 seed=7, smooth=1.0)

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

outfolder = "/home/rai-minotaur-environment/data/area1/voxel"

if True:  # Smooth environment

    # --- Define region masks: three non-overlapping shells ---
    region1 = field > 2.5      # inner core
    region2 = (field > 1.5) & (field <= 2.5)   # middle shell
    region3 = (field > 0.5) & (field <= 1.5)   # outer shell

    regions = [region1, region2, region3]

    for i, mask in enumerate(regions):
        # Marching cubes expects scalar field, so convert mask → float field
        float_field = mask.astype(float)

        verts, faces, normals, values = measure.marching_cubes(
            float_field, level=0.5,
            spacing=((bounds[1]-bounds[0])/nx,
                        (bounds[1]-bounds[0])/ny,
                        (bounds[1]-bounds[0])/nz))

        outfile = f"{outfile_prefix}_{i+1}.obj"
        with open(outfile, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Saved layer {i+1} → {outfile}")

if False:  # Voxel environment

    # --- Cube template ---
    cube_verts = np.array([
        [0,0,0],
        [1,0,0],
        [1,1,0],
        [0,1,0],
        [0,0,1],
        [1,0,1],
        [1,1,1],
        [0,1,1]
    ])

    cube_faces = np.array([
        [0,1,2], [0,2,3],   # bottom
        [4,5,6], [4,6,7],   # top
        [0,1,5], [0,5,4],   # front
        [2,3,7], [2,7,6],   # back
        [1,2,6], [1,6,5],   # right
        [3,0,4], [3,4,7]    # left
    ])

    cube_size = 5  # number of voxels per cube edge

    nx, ny, nz = len(x), len(y), len(z)
    dx = (bounds[1]-bounds[0])/nx
    dy = (bounds[1]-bounds[0])/ny
    dz = (bounds[1]-bounds[0])/nz
    scale = np.array([dx, dy, dz])
    scaled_cube = cube_verts * (scale * cube_size)

    # --- Define voxel occupancy regions ---
    region1 = field > 2.5
    region2 = (field > 1.5) & (field <= 2.5)
    region3 = (field > 0.5) & (field <= 1.5)
    regions = [region1, region2, region3]

    # --- Build voxel meshes ---
    for rid, mask in enumerate(regions, start=1):
        print(f"Processing region {rid}...")

        verts_out = []
        faces_out = []
        v_offset = 0

        for idx in range(0, nx, cube_size):
            for idy in range(0, ny, cube_size):
                for idz in range(0, nz, cube_size):
                    block = mask[idx:idx+cube_size, idy:idy+cube_size, idz:idz+cube_size]
                    if not np.any(block):
                        continue

                    origin = np.array([x[idx], y[idy], z[idz]])
                    v = scaled_cube + origin
                    verts_out.append(v)
                    faces_out.append(cube_faces + v_offset)
                    v_offset += 8

        if len(verts_out) == 0:
            print(f"Region {rid} empty, skipping.")
            continue

        verts_out = np.vstack(verts_out)
        faces_out = np.vstack(faces_out)

        outfile = os.path.join(outfolder, f"layer_{rid}.obj")
        with open(outfile, "w") as f:
            for vx, vy, vz in verts_out:
                f.write(f"v {vx} {vy} {vz}\n")
            for a,b,c in faces_out:
                f.write(f"f {a+1} {b+1} {c+1}\n")
        print(f"Saved voxel mesh → {outfile}")

    # --- Determine outer layer bounding box ---
    outer_mask = region1 | region2 | region3
    filled_indices = np.argwhere(outer_mask)
    if filled_indices.size == 0:
        print("No outer layer found, skipping shell.")
        exit(0)

    # min/max indices along each axis
    min_idx = filled_indices.min(axis=0)
    max_idx = filled_indices.max(axis=0)

    # --- Build enclosing shell around outer layer ---
    shell_verts = []
    shell_faces = []
    v_offset = 0

    for idx in range(min_idx[0], max_idx[0]+1, cube_size):
        for idy in range(min_idx[1], max_idx[1]+1, cube_size):
            for idz in range(min_idx[2], max_idx[2]+1, cube_size):
                # skip if inside the outer layer
                block = outer_mask[idx:idx+cube_size, idy:idy+cube_size, idz:idz+cube_size]
                if np.any(block):
                    continue

                origin = np.array([x[idx], y[idy], z[idz]])
                v = scaled_cube + origin
                shell_verts.append(v)
                shell_faces.append(cube_faces + v_offset)
                v_offset += 8

    if shell_verts:
        shell_verts = np.vstack(shell_verts)
        shell_faces = np.vstack(shell_faces)
        outfile = os.path.join(outfolder, "shell.obj")
        with open(outfile, "w") as f:
            for vx, vy, vz in shell_verts:
                f.write(f"v {vx} {vy} {vz}\n")
            for a,b,c in shell_faces:
                f.write(f"f {a+1} {b+1} {c+1}\n")
        print(f"Saved enclosing shell → {outfile}")