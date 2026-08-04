import numpy as np
import argparse
import importlib.util
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

VOXEL_SIZE = 10.0

NPY_FILE = "area1.npy"
GSLIB_FILE = "area1.gslib"

OBJ_FILE = "env3d_voxel_geomodel.obj"
MTL_FILE = "env3d_voxel_geomodel.mtl"

TEXTURE_DIR = Path("textures")

# =====================================================
# LOAD TEXTURE MODULE
# =====================================================

def load_texture_module(texture_file):

    spec = importlib.util.spec_from_file_location(
        "texture_module",
        texture_file
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# =====================================================
# GSLIB
# =====================================================

def load_gslib(path):

    data = np.loadtxt(
        path,
        skiprows=3
    )

    expected = 20 * 20 * 20

    if data.size != expected:
        raise RuntimeError(
            f"Expected {expected} values, got {data.size}"
        )

    return data.reshape(
        (20, 20, 20),
        order="F"
    )


# =====================================================
# GEOMETRY
# =====================================================

CUBE_VERTS = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1]
], dtype=float)

CUBE_QUADS = [
    (0,1,2,3),
    (4,5,6,7),
    (0,1,5,4),
    (2,3,7,6),
    (1,2,6,5),
    (3,0,4,7)
]

UV_QUAD = [
    (0,0),
    (1,0),
    (1,1),
    (0,1)
]


# =====================================================
# TEXTURES
# =====================================================

def generate_material_textures(texture_module):

    TEXTURE_DIR.mkdir(
        exist_ok=True,
        parents=True
    )

    host_file = TEXTURE_DIR / "host.png"
    mag_file = TEXTURE_DIR / "magnesite.png"

    cfg = texture_module.TextureConfig(
        width=512,
        height=512,
        seed=12,
        geoseed=1
    )

    texture_module.generate_texture(
        mode="host",
        cfg=cfg
    ).save(host_file)

    texture_module.generate_texture(
        mode="magnesite",
        cfg=cfg
    ).save(mag_file)

    return host_file, mag_file


# =====================================================
# MTL
# =====================================================

def write_mtl(path, host_tex, mag_tex):

    with open(path, "w") as f:

        f.write(
            f"""newmtl host
Ka 1 1 1
Kd 1 1 1
Ks 0 0 0
map_Kd {host_tex.as_posix()}

newmtl magnesite
Ka 1 1 1
Kd 1 1 1
Ks 0 0 0
map_Kd {mag_tex.as_posix()}

"""
        )


# =====================================================
# CUBE
# =====================================================

def write_cube(
    f,
    origin,
    material,
    vertex_offset,
    uv_offset,
    cube_id
):

    verts = (
        CUBE_VERTS * VOXEL_SIZE
        + origin
    )

    f.write(f"o cube_{cube_id}\n")
    f.write(f"usemtl {material}\n")

    for v in verts:
        f.write(
            f"v {v[0]} {v[1]} {v[2]}\n"
        )

    current_uv = uv_offset

    for quad in CUBE_QUADS:

        for u, v in UV_QUAD:
            f.write(
                f"vt {u} {v}\n"
            )

        a,b,c,d = quad

        v1 = vertex_offset + a
        v2 = vertex_offset + b
        v3 = vertex_offset + c
        v4 = vertex_offset + d

        t1 = current_uv
        t2 = current_uv + 1
        t3 = current_uv + 2
        t4 = current_uv + 3

        f.write(
            f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n"
        )

        f.write(
            f"f {v1}/{t1} {v3}/{t3} {v4}/{t4}\n"
        )

        current_uv += 4

    return (
        vertex_offset + 8,
        current_uv
    )


# =====================================================
# EXPORT
# =====================================================

def export_world(
    npy_file,
    gslib_file,
    texture_file
):

    occupancy = np.load(npy_file)

    if occupancy.shape != (20,20,20):
        raise RuntimeError(
            f"Expected (20,20,20), got {occupancy.shape}"
        )

    gslib = load_gslib(
        gslib_file
    )

    texture_module = (
        load_texture_module(
            texture_file
        )
    )

    host_tex, mag_tex = (
        generate_material_textures(
            texture_module
        )
    )

    write_mtl(
        MTL_FILE,
        host_tex,
        mag_tex
    )

    with open(OBJ_FILE, "w") as f:

        f.write(
            f"mtllib {MTL_FILE}\n"
        )

        vertex_offset = 1
        uv_offset = 1
        cube_id = 0

        for x in range(20):
            for y in range(20):
                for z in range(20):

                    if occupancy[x,y,z] == 0:
                        continue

                    value = gslib[x,y,z]

                    if value == -9999.99:
                        continue

                    material = (
                        "magnesite"
                        if value == 1
                        else "host"
                    )

                    origin = np.array([
                        x * VOXEL_SIZE,
                        y * VOXEL_SIZE,
                        z * VOXEL_SIZE
                    ])

                    (
                        vertex_offset,
                        uv_offset
                    ) = write_cube(
                        f,
                        origin,
                        material,
                        vertex_offset,
                        uv_offset,
                        cube_id
                    )

                    cube_id += 1

    print(
        f"Exported {cube_id} cubes"
    )

    print(
        f"OBJ -> {OBJ_FILE}"
    )

    print(
        f"MTL -> {MTL_FILE}"
    )


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npy-file",
        default=NPY_FILE
    )

    parser.add_argument(
        "--gslib-file",
        default=GSLIB_FILE
    )

    parser.add_argument(
        "--texture-file",
        default="texture.py"
    )

    args = parser.parse_args()

    export_world(
        args.npy_file,
        args.gslib_file,
        args.texture_file
    )