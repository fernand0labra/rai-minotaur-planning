"""Per-voxel textured voxel world exporter.

This version keeps the same 3D placement logic for magnesite, stockwork,
and veins, but generates a separate texture image and material for each
voxel cube that is exported.

To use one base image for all voxel textures:
    python world.py --texture-path /path/to/texture.png
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps


# -----------------------------------------------------------------------------
# Safe loader for environment.py
# -----------------------------------------------------------------------------


def load_function_from_python_file(file_path: Path, function_name: str):
    """Load one function from a .py file without executing top-level code."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    selected_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if not selected_nodes:
        raise ValueError(f"Function {function_name!r} not found in {file_path}")

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, filename=str(file_path), mode="exec")

    namespace = {"np": np, "math": math}
    exec(code, namespace)
    return namespace[function_name]


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

CUBE_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

# Six quad faces for a cube, each quad will be triangulated into two OBJ faces.
CUBE_QUADS = [
    (0, 1, 2, 3),  # bottom
    (4, 5, 6, 7),  # top
    (0, 1, 5, 4),  # front
    (2, 3, 7, 6),  # back
    (1, 2, 6, 5),  # right
    (3, 0, 4, 7),  # left
]

UV_QUAD = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def normalize01(a: np.ndarray) -> np.ndarray:
    amin = float(np.min(a))
    amax = float(np.max(a))
    if abs(amax - amin) < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def sample_centers_from_field(
    field: np.ndarray,
    n: int,
    rng: np.random.Generator,
    quantile_floor: float = 0.75,
) -> np.ndarray:
    """Pick candidate voxel indices from the high-value part of a field."""
    threshold = float(np.quantile(field, quantile_floor))
    candidates = np.argwhere(field >= threshold)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones_like(field, dtype=bool))

    weights = field[tuple(candidates.T)]
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0) ** 2
    s = float(weights.sum())
    if s <= 0:
        weights = np.ones(len(candidates), dtype=np.float64)
        s = float(weights.sum())

    if len(candidates) <= n:
        choice = rng.choice(len(candidates), size=n, replace=True, p=weights / s)
    else:
        choice = rng.choice(len(candidates), size=n, replace=False, p=weights / s)
    return candidates[choice]


def ellipsoidal_gaussian_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    centers: Sequence[Tuple[float, float, float]],
    radii: Sequence[Tuple[float, float, float]],
    amplitudes: Sequence[float],
) -> np.ndarray:
    field = np.zeros_like(X, dtype=np.float32)
    for (cx, cy, cz), (rx, ry, rz), amp in zip(centers, radii, amplitudes):
        rx = max(float(rx), 1e-6)
        ry = max(float(ry), 1e-6)
        rz = max(float(rz), 1e-6)
        dx = (X - cx) / rx
        dy = (Y - cy) / ry
        dz = (Z - cz) / rz
        field += float(amp) * np.exp(-0.5 * (dx * dx + dy * dy + dz * dz)).astype(np.float32)
    return field


def point_segment_distance_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> np.ndarray:
    """Distance from every grid point to a 3D line segment."""
    vx, vy, vz = (p1 - p0).astype(np.float32)
    wx = X - p0[0]
    wy = Y - p0[1]
    wz = Z - p0[2]
    vv = vx * vx + vy * vy + vz * vz
    if vv < 1e-12:
        return np.sqrt(wx * wx + wy * wy + wz * wz)
    t = (wx * vx + wy * vy + wz * vz) / vv
    t = np.clip(t, 0.0, 1.0)
    px = p0[0] + t * vx
    py = p0[1] + t * vy
    pz = p0[2] + t * vz
    dx = X - px
    dy = Y - py
    dz = Z - pz
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def build_vein_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    seed_points: np.ndarray,
    rng: np.random.Generator,
    bounds: Tuple[float, float],
    n_veins: int = 5,
) -> np.ndarray:
    """Create several 3D veins with decay and some that connect to other deposits."""
    lo, hi = bounds
    field = np.zeros_like(X, dtype=np.float32)
    n_seed = len(seed_points)

    for i in range(n_veins):
        if n_seed > 0 and (i % 2 == 0):
            s0 = seed_points[int(rng.integers(0, n_seed))]
            s1 = seed_points[int(rng.integers(0, n_seed))]
            start = np.array([s0[0], s0[1], s0[2]], dtype=np.float32)
            end = np.array([s1[0], s1[1], s1[2]], dtype=np.float32)
        else:
            start = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=np.float32)
            end = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=np.float32)

        n_ctrl = int(rng.integers(3, 6))
        controls = [start]
        for _ in range(n_ctrl - 2):
            mid = rng.uniform(lo, hi, size=3).astype(np.float32)
            controls.append(mid)
        controls.append(end)

        controls = np.asarray(controls, dtype=np.float32)
        controls[1:-1] += rng.normal(0.0, (hi - lo) * 0.04, size=controls[1:-1].shape)
        controls = np.clip(controls, lo, hi)

        thickness = float(rng.uniform((hi - lo) * 0.015, (hi - lo) * 0.035))
        spread = float(rng.uniform(0.85, 1.35))
        amp = float(rng.uniform(0.7, 1.2))

        for p0, p1 in zip(controls[:-1], controls[1:]):
            d = point_segment_distance_field(X, Y, Z, p0, p1)
            mid = 0.5 * (p0 + p1)
            seg_len = float(np.linalg.norm(p1 - p0)) + 1e-6
            axial = np.exp(-((np.sqrt((X - mid[0]) ** 2 + (Y - mid[1]) ** 2 + (Z - mid[2]) ** 2) / seg_len) ** 2))
            field += amp * np.exp(-((d / thickness) ** 2) * spread).astype(np.float32) * axial.astype(np.float32)

    return field


# -----------------------------------------------------------------------------
# Texture helpers
# -----------------------------------------------------------------------------


def load_texture_module(texture_file: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("texture_module", str(texture_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load texture module from {texture_file}")
    texture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(texture_module)
    return texture_module


def stable_voxel_hash(label: str, ix: int, iy: int, iz: int, seed: int) -> int:
    digest = hashlib.sha256(f"{label}:{ix}:{iy}:{iz}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def prepare_base_image(texture_module, texture_path: Optional[Path], mode: str, seed: int, geoseed: int) -> Image.Image:
    cfg_cls = texture_module.TextureConfig
    cfg = cfg_cls(width=512, height=512, seed=seed, geoseed=geoseed)
    return texture_module.generate_texture(mode=mode, cfg=cfg)

def make_voxel_texture(
    texture_module,
    texture_dir: Path,
    label: str,
    ix: int,
    iy: int,
    iz: int,
    seed: int,
    texture_path: Optional[Path] = None,
) -> Path:
    """Create a dedicated texture image for a single voxel."""
    voxel_dir = texture_dir / "voxels"
    voxel_dir.mkdir(parents=True, exist_ok=True)

    out_path = voxel_dir / f"{label}_{ix}_{iy}_{iz}.png"
    mode = label if label in {"veins", "stockwork", "magnesite"} else "host"
    img = prepare_base_image(texture_module, None, mode=mode, seed=12, geoseed=ix + iy + iz)
    img.save(out_path)

    return out_path


# -----------------------------------------------------------------------------
# Field building
# -----------------------------------------------------------------------------


def build_deposit_masks(
    base_field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world mask and the three deposit masks with spatial decay."""
    lo, hi = bounds
    base = normalize01(base_field)

    # Magnesite: several blob-like volumes.
    mag_seeds = sample_centers_from_field(base, n=5, rng=rng, quantile_floor=0.78)
    mag_centers = []
    mag_radii = []
    mag_amps = []
    for seed_pt in mag_seeds:
        i, j, k = (int(seed_pt[0]), int(seed_pt[1]), int(seed_pt[2]))
        i = int(np.clip(i, 0, X.shape[0] - 1))
        j = int(np.clip(j, 0, X.shape[1] - 1))
        k = int(np.clip(k, 0, X.shape[2] - 1))
        cx, cy, cz = float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])
        mag_centers.append((cx, cy, cz))
        rxy = float(rng.uniform((hi - lo) * 0.06, (hi - lo) * 0.14))
        rz = float(rng.uniform((hi - lo) * 0.04, (hi - lo) * 0.10))
        mag_radii.append((rxy, rxy * rng.uniform(0.75, 1.25), rz))
        mag_amps.append(float(rng.uniform(0.95, 1.35)))

    mag_field = ellipsoidal_gaussian_field(X, Y, Z, mag_centers, mag_radii, mag_amps)
    mag_field = np.clip(mag_field, 0.0, 1.0)

    # Stockwork: halos around magnesite plus some separate clusters.
    stock_centers = []
    stock_radii = []
    stock_amps = []
    for (cx, cy, cz), (rx, ry, rz) in zip(mag_centers, mag_radii):
        stock_centers.append((cx, cy, cz))
        stock_radii.append((rx * 1.8, ry * 1.8, rz * 1.8))
        stock_amps.append(0.55)

        for _ in range(int(rng.integers(2, 5))):
            offset = rng.normal(0.0, (hi - lo) * 0.05, size=3)
            stock_centers.append(
                (
                    float(np.clip(cx + offset[0], lo, hi)),
                    float(np.clip(cy + offset[1], lo, hi)),
                    float(np.clip(cz + offset[2], lo, hi)),
                )
            )
            stock_radii.append(
                (
                    rx * rng.uniform(0.35, 0.8),
                    ry * rng.uniform(0.35, 0.8),
                    rz * rng.uniform(0.35, 0.8),
                )
            )
            stock_amps.append(float(rng.uniform(0.35, 0.65)))

    for _ in range(3):
        cx, cy, cz = rng.uniform(lo, hi, size=3)
        stock_centers.append((float(cx), float(cy), float(cz)))
        scale = float(rng.uniform((hi - lo) * 0.04, (hi - lo) * 0.09))
        stock_radii.append((scale * 1.35, scale * 1.0, scale * 0.8))
        stock_amps.append(float(rng.uniform(0.25, 0.45)))

    stock_field = ellipsoidal_gaussian_field(X, Y, Z, stock_centers, stock_radii, stock_amps)
    stock_field = np.clip(stock_field, 0.0, 1.0)

    # Veins: thin paths that may connect to magnesite or stockwork.
    vein_field = build_vein_field(X, Y, Z, np.asarray(mag_centers, dtype=np.float32), rng, bounds, n_veins=6)
    vein_field = normalize01(vein_field)

    # World mask comes from the broad geological envelope from environment.py.
    world_field = normalize01(base_field)
    world_mask = world_field > 0.18

    # Decay at the edges so the regions blend instead of having hard cuts.
    mag_field = np.clip(mag_field * world_field, 0.0, 1.0)
    stock_field = np.clip(stock_field * (0.65 + 0.35 * world_field), 0.0, 1.0)
    vein_field = np.clip(vein_field * (0.55 + 0.45 * world_field), 0.0, 1.0)

    mag_q = float(np.quantile(mag_field[world_mask], 0.84)) if np.any(world_mask) else 0.75
    stock_q = float(np.quantile(stock_field[world_mask], 0.81)) if np.any(world_mask) else 0.70
    vein_q = float(np.quantile(vein_field[world_mask], 0.90)) if np.any(world_mask) else 0.82

    mag_mask = (mag_field >= mag_q) & world_mask
    stock_mask = (stock_field >= stock_q) & world_mask & (~mag_mask)
    vein_mask = (vein_field >= vein_q) & world_mask & (~mag_mask) & (~stock_mask)

    return world_mask, vein_mask, stock_mask, mag_mask


# -----------------------------------------------------------------------------
# OBJ/MTL export
# -----------------------------------------------------------------------------


def voxel_label_for_block(
    world_mask: np.ndarray,
    vein_mask: np.ndarray,
    stock_mask: np.ndarray,
    mag_mask: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    step: int,
) -> Optional[str]:
    block_world = world_mask[ix:ix + step, iy:iy + step, iz:iz + step]
    if not np.any(block_world):
        return None
    block_mag = mag_mask[ix:ix + step, iy:iy + step, iz:iz + step]
    block_stock = stock_mask[ix:ix + step, iy:iy + step, iz:iz + step]
    block_vein = vein_mask[ix:ix + step, iy:iy + step, iz:iz + step]

    # Priority: magnesite > stockwork > veins > host.
    if np.any(block_mag):
        return "magnesite"
    if np.any(block_stock):
        return "stockwork"
    if np.any(block_vein):
        return "veins"
    return "host"


def write_mtl(mtl_path: Path, material_to_texture: Dict[str, Path], transparency: float = 1.0) -> None:
    with mtl_path.open("w", encoding="utf-8") as f:
        for name in ["host", "veins", "stockwork", "magnesite"]:
            tex = material_to_texture[name]
            f.write(f"newmtl {name}\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write(f"d {float(transparency):.3f}\n")
            f.write(f"map_Kd {tex.as_posix()}\n\n")


def write_textured_cube(
    f,
    origin: np.ndarray,
    cube_size: float,
    material: str,
    vertex_offset: int,
    vt_offset: int,
    object_name: str,
) -> Tuple[int, int]:
    verts = CUBE_VERTS * cube_size + origin[None, :]
    f.write(f"o {object_name}\n")
    f.write(f"usemtl {material}\n")

    for vx, vy, vz in verts:
        f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

    current_vt = vt_offset
    for quad in CUBE_QUADS:
        for u, v in UV_QUAD:
            f.write(f"vt {u:.6f} {v:.6f}\n")

        a, b, c, d = quad
        v_ids = [vertex_offset + a, vertex_offset + b, vertex_offset + c, vertex_offset + d]
        vt_ids = [current_vt + 0, current_vt + 1, current_vt + 2, current_vt + 3]

        f.write(f"f {v_ids[0]}/{vt_ids[0]} {v_ids[1]}/{vt_ids[1]} {v_ids[2]}/{vt_ids[2]}\n")
        f.write(f"f {v_ids[0]}/{vt_ids[0]} {v_ids[2]}/{vt_ids[2]} {v_ids[3]}/{vt_ids[3]}\n")
        current_vt += 4

    return vertex_offset + 8, current_vt


def export_world(
    environment_file: Path,
    texture_file: Path,
    output_obj: Path,
    output_mtl: Path,
    texture_dir: Path,
    bounds: Tuple[float, float] = (-100.0, 100.0),
    grid_n: int = 96,
    voxel_size: int = 5,
    seed: int = 12,
    texture_path: Optional[Path] = None,
    export_mode: str = "all",
) -> None:
    rng = np.random.default_rng(seed)
    scalar_field_elliptical_3d = load_function_from_python_file(environment_file, "scalar_field_elliptical_3d")
    texture_module = load_texture_module(texture_file)

    lo, hi = bounds
    x = np.linspace(lo, hi, grid_n, dtype=np.float32)
    y = np.linspace(lo, hi, grid_n, dtype=np.float32)
    z = np.linspace(lo, hi, grid_n, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Use the environment field as the host geology.

    # Area 1
    # base_field = scalar_field_elliptical_3d(X, Y, Z,
    #                 center=(10.0, 10.0, 0.0),
    #                 R1=75.0, R2=50.0, R3=25.0,
    #                 anisotropy=(1.0, 1.0, 1.2),
    #                 warp_strength=0.0,
    #                 irregularity=15.0,
    #                 seed=13, smooth=1.0)

    # Area 2
    base_field = scalar_field_elliptical_3d(X, Y, Z,
                    center=(5.0, 10.0, 0.0),
                    R1=80.0, R2=65.0, R3=25.0,
                    anisotropy=(1.0, 1.0, 1.2),
                    warp_strength=0.0,
                    irregularity=20.0,
                    seed=7, smooth=1.0)

    # Area 3
    # base_field = scalar_field_elliptical_3d(X, Y, Z,
    #                 center=(10.0, 0.0, 5.0),
    #                 R1=50.0, R2=30.0, R3=25.0,
    #                 anisotropy=(1.0, 2.0, 1.5),
    #                 warp_strength=0.0,
    #                 irregularity=20.0,
    #                 seed=13, smooth=1.0)

    # Area 4
    # base_field = scalar_field_elliptical_3d(X, Y, Z,
    #     center=(0.0, 10.0, 0.0),
    #     R1=45.0, R2=30.0, R3=25.0,
    #     anisotropy=(2.0, 1.0, 1.2),
    #     warp_strength=0.0,
    #     irregularity=5.0,
    #     seed=13, smooth=1.0)

    world_mask, vein_mask, stock_mask, mag_mask = build_deposit_masks(
        base_field=base_field,
        X=X,
        Y=Y,
        Z=Z,
        bounds=bounds,
        rng=rng,
    )

    # Use the same cube size concept as environment.py, but allow a configurable value.
    cube_world = voxel_size * ((hi - lo) / float(grid_n))

    texture_dir.mkdir(parents=True, exist_ok=True)
    material_to_texture: Dict[str, Path] = {}
    material_source_mode = {
        "host": "host",
        "veins": "veins",
        "stockwork": "stockwork",
        "magnesite": "magnesite",
    }

    # Prebuild one base texture per material, then make a unique per-voxel copy later.
    # This makes the voxel textures visibly related while still giving each voxel its own image.
    for mat, mode in material_source_mode.items():
        cfg = texture_module.TextureConfig(width=512, height=512, seed=seed, geoseed=1)
        base_img = texture_module.generate_texture(mode=mode, cfg=cfg).convert("RGB")
        mat_path = texture_dir / f"base_{mat}.png"
        base_img.save(mat_path)
        material_to_texture[mat] = mat_path

    # write_mtl(output_mtl, material_to_texture)

    voxel_texture_cache: Dict[Tuple[str, int, int, int], Path] = {}

    with output_obj.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {output_mtl.name}\n")
        vertex_offset = 1
        vt_offset = 1
        cube_id = 0

        for ix in range(0, grid_n, voxel_size):
            for iy in range(0, grid_n, voxel_size):
                for iz in range(0, grid_n, voxel_size):
                    label = voxel_label_for_block(world_mask, vein_mask, stock_mask, mag_mask, ix, iy, iz, voxel_size)
                    if label is None:
                        continue

                    if not should_export(label, export_mode):
                        continue

                    ## COMMENT
                    cache_key = (label, ix, iy, iz)
                    if cache_key not in voxel_texture_cache:
                        voxel_texture_cache[cache_key] = make_voxel_texture(
                            texture_module=texture_module,
                            texture_dir=texture_dir,
                            label=label,
                            ix=ix,
                            iy=iy,
                            iz=iz,
                            seed=seed,
                            texture_path=material_to_texture[label] if texture_path is not None else None,
                        )

                    ## COMMENT
                    ## We keep the per-material MTL, but each voxel uses its own material entry.
                    voxel_material = f"{label}_{ix}_{iy}_{iz}"
                    voxel_tex = voxel_texture_cache[cache_key]
                    with output_mtl.open("a", encoding="utf-8") as mtl:
                        mtl.write(f"newmtl {voxel_material}\n")
                        mtl.write("Ka 1.000 1.000 1.000\n")
                        mtl.write("Kd 1.000 1.000 1.000\n")
                        mtl.write("Ks 0.000 0.000 0.000\n")
                        if label == "host": mtl.write("d 0.000\n")
                        else:               mtl.write("d 1.000\n")
                        
                        mtl.write(f"map_Kd {voxel_tex.as_posix()}\n\n")

                    # voxel_material = f"{label}"

                    origin = np.array([x[ix], y[iy], z[iz]], dtype=np.float32)
                    object_name = f"{label}_cube_{cube_id}"
                    vertex_offset, vt_offset = write_textured_cube(
                        f=f,
                        origin=origin,
                        cube_size=float(cube_world),
                        material=voxel_material,
                        vertex_offset=vertex_offset,
                        vt_offset=vt_offset,
                        object_name=object_name,
                    )
                    cube_id += 1

    # Prepend the material library reference now that the MTL is complete.
    # If the OBJ file already has the line at the top, leave it as-is.
    print(f"Saved OBJ: {output_obj}")
    print(f"Saved MTL: {output_mtl}")
    print(f"Saved voxel textures in: {texture_dir / 'voxels'}")


def should_export(label: str, mode: str) -> bool:
    return (
        mode == "all"
        or (mode == "no_host" and label != "host")
        or (mode == "veins" and label == "veins")
        or (mode == "stockwork" and label == "stockwork")
        or (mode == "magnesite" and label == "magnesite")
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a per-voxel textured world from environment.py and texture.py")
    parser.add_argument("--environment-file", type=Path, default=Path("src/env3d_smooth.py"), help="Path to environment.py")
    parser.add_argument("--texture-file", type=Path, default=Path("src/utils/texture.py"), help="Path to texture.py")
    parser.add_argument("--output-obj", type=Path, default=Path("env3d_voxel_texture.obj"), help="Output OBJ path")
    parser.add_argument("--output-mtl", type=Path, default=Path("env3d_voxel_texture.mtl"), help="Output MTL path")
    parser.add_argument("--texture-dir", type=Path, default=Path("textures"), help="Directory for generated textures")
    parser.add_argument("--texture-path", type=Path, default=None, help="Optional single source texture to duplicate per voxel")
    parser.add_argument("--grid-n", type=int, default=85, help="Grid resolution along each axis")
    parser.add_argument("--voxel-size", type=int, default=1, help="Voxel step when exporting")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--bounds", type=float, nargs=2, default=(-100.0, 100.0), help="World bounds as two floats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modes = [
        # ("world", "all"),
        ("world", "no_host"),
        # ("world_veins", "veins"),
        # ("world_stockwork", "stockwork"),
        # ("world_magnesite", "magnesite"),
    ]

    for name, mode in modes:
        export_world(
            environment_file=args.environment_file,
            texture_file=args.texture_file,
            output_obj=Path(f"{args.output_obj}"),
            output_mtl=Path(f"{args.output_mtl}"),
            texture_dir=args.texture_dir,
            bounds=(float(args.bounds[0]), float(args.bounds[1])),
            grid_n=int(args.grid_n),
            voxel_size=int(args.voxel_size),
            seed=int(args.seed),
            texture_path=args.texture_path,
            export_mode=mode,
        )


if __name__ == "__main__":
    main()
