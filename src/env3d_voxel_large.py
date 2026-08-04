from pathlib import Path
import numpy as np

from env3d_voxel_texture import (
    load_function_from_python_file,
    build_deposit_masks,
    voxel_label_for_block,
    write_textured_cube,
    write_mtl,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# (WORLD_SIZE_METERS/GRID_N) * VOXEL_STEP

WORLD_SIZE_METERS = 500.0
GRID_N = 100
VOXEL_STEP = 2
SEED = 12

rng = np.random.default_rng(SEED)

N_AREAS = 30

AREAS = []

half = WORLD_SIZE_METERS / 2

for _ in range(N_AREAS):

    AREAS.append(
        dict(
            center=(
                rng.uniform(-0.85 * half, 0.85 * half),
                rng.uniform(-0.85 * half, 0.85 * half),
                rng.uniform(-0.85 * half, 0.85 * half),
            ),

            # Major axis
            R1=rng.uniform(30.0, 120.0),

            # Intermediate axis
            R2=rng.uniform(20.0, 80.0),

            # Minor axis
            R3=rng.uniform(10.0, 50.0),

            seed=int(rng.integers(1, 1_000_000)),

            irregularity=rng.uniform(2.0, 25.0),
        )
    )

# ---------------------------------------------------------
# GRID
# ---------------------------------------------------------

def create_grid():

    half = WORLD_SIZE_METERS / 2.0

    x = np.linspace(-half, half, GRID_N, dtype=np.float32)
    y = np.linspace(-half, half, GRID_N, dtype=np.float32)
    z = np.linspace(-half, half, GRID_N, dtype=np.float32)

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    return x, y, z, X, Y, Z


# ---------------------------------------------------------
# BUILD MULTIPLE ORE SYSTEMS
# ---------------------------------------------------------

def build_world(environment_file):

    scalar_field = load_function_from_python_file(
        environment_file,
        "scalar_field_elliptical_3d",
    )

    x, y, z, X, Y, Z = create_grid()

    world_mask = np.zeros(X.shape, dtype=bool)
    vein_mask = np.zeros(X.shape, dtype=bool)
    stock_mask = np.zeros(X.shape, dtype=bool)
    mag_mask = np.zeros(X.shape, dtype=bool)

    bounds = (
        -WORLD_SIZE_METERS / 2,
        WORLD_SIZE_METERS / 2,
    )

    for area in AREAS:

        field = scalar_field(
            X,
            Y,
            Z,
            center=area["center"],
            R1=area["R1"],
            R2=area["R2"],
            R3=area["R3"],
            anisotropy=(1.0, 1.0, 1.2),
            warp_strength=0.0,
            irregularity=area["irregularity"],
            seed=area["seed"],
            smooth=1.0,
        )

        local_rng = np.random.default_rng(area["seed"])

        area_world, area_veins, area_stock, area_mag = (
            build_deposit_masks(
                base_field=field,
                X=X,
                Y=Y,
                Z=Z,
                bounds=bounds,
                rng=local_rng,
            )
        )

        world_mask |= area_world
        vein_mask |= area_veins
        stock_mask |= area_stock
        mag_mask |= area_mag

    return (
        x,
        y,
        z,
        world_mask,
        vein_mask,
        stock_mask,
        mag_mask,
    )


# ---------------------------------------------------------
# BOUNDING BOX
# ---------------------------------------------------------

def append_bounding_box(
    obj,
    vertex_offset,
    vt_offset,
):

    half = WORLD_SIZE_METERS / 2.0

    return write_textured_cube(
        f=obj,
        origin=np.array(
            [-half, -half, -half],
            dtype=np.float32,
        ),
        cube_size=WORLD_SIZE_METERS,
        material="host_rock",
        vertex_offset=vertex_offset,
        vt_offset=vt_offset,
        object_name="host_box",
    )


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

def export_world():

    environment_file = Path("src/env3d_smooth.py")
    (
        x,
        y,
        z,
        world_mask,
        vein_mask,
        stock_mask,
        mag_mask,
    ) = build_world(environment_file)

    host_mask = np.ones(world_mask.shape, dtype=bool)

    host_mask &= ~mag_mask
    host_mask &= ~stock_mask
    host_mask &= ~vein_mask

    material_textures = {
        "host": Path("textures/base_host.png"),
        "veins": Path("textures/base_veins.png"),
        "stockwork": Path("textures/base_stockwork.png"),
        "magnesite": Path("textures/base_magnesite.png"),
    }

    write_mtl(
        Path("env3d_voxel_large.mtl"),
        material_textures,
    )

    with open("env3d_voxel_large.mtl", "a") as f:

        f.write("newmtl host_rock\n")
        f.write("Ka 1 1 1\n")
        f.write("Kd 1 1 1\n")
        f.write("Ks 0 0 0\n")

    cube_size = (
        VOXEL_STEP
        * WORLD_SIZE_METERS
        / GRID_N
    )

    with open("env3d_voxel_large.obj", "w") as obj:

        obj.write("mtllib env3d_voxel_large.mtl\n")

        vertex_offset = 1
        vt_offset = 1
        cube_id = 0

        for ix in range(0, GRID_N, VOXEL_STEP):
            for iy in range(0, GRID_N, VOXEL_STEP):
                for iz in range(0, GRID_N, VOXEL_STEP):

                    block_mag = np.any(
                        mag_mask[
                            ix:ix+VOXEL_STEP,
                            iy:iy+VOXEL_STEP,
                            iz:iz+VOXEL_STEP
                        ]
                    )

                    block_stock = np.any(
                        stock_mask[
                            ix:ix+VOXEL_STEP,
                            iy:iy+VOXEL_STEP,
                            iz:iz+VOXEL_STEP
                        ]
                    )

                    block_vein = np.any(
                        vein_mask[
                            ix:ix+VOXEL_STEP,
                            iy:iy+VOXEL_STEP,
                            iz:iz+VOXEL_STEP
                        ]
                    )

                    if block_mag:
                        label = "magnesite"

                    elif block_stock:
                        label = "stockwork"

                    elif block_vein:
                        label = "veins"

                    else:
                        label = "host"

                    origin = np.array(
                        [
                            x[ix],
                            y[iy],
                            z[iz],
                        ],
                        dtype=np.float32,
                    )

                    vertex_offset, vt_offset = write_textured_cube(
                        f=obj,
                        origin=origin,
                        cube_size=float(cube_size),
                        material=label,
                        vertex_offset=vertex_offset,
                        vt_offset=vt_offset,
                        object_name=f"{label}_{cube_id}",
                    )

                    cube_id += 1

        # vertex_offset, vt_offset = append_bounding_box(
        #     obj,
        #     vertex_offset,
        #     vt_offset,
        # )

    print("Saved env3d_voxel_large.obj")
    print("Saved env3d_voxel_large.mtl")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    export_world()