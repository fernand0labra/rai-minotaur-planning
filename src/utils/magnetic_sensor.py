#!/usr/bin/env python3
"""
Compute and plot magnetic susceptibility for voxel textures.

This script expects the voxel textures produced by the voxel exporter, for example:
    textures/voxels/magnesite_12_24_36.png
    textures/voxels/stockwork_8_16_20.png

It reads each voxel texture, estimates how much magnesite is present from the
high color channels, applies a label-aware density model for:
    - veins
    - stockwork
    - magnesite
    - host

and then renders a 3D gradient-style plot of magnetic susceptibility.

Usage:
    python magnetic_sensor.py \
        --texture-dir textures/voxels \
        --bounds -100 100 \
        --grid-n 96 \
        --voxel-size 5 \
        --output-csv susceptibility.csv \
        --output-plot susceptibility_3d.png
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Registers the 3D projection.


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Higher values mean stronger response to the bright/high-channel content.
# These are intentionally ordered by density/purity of the ore style.
LABEL_DENSITY = {
    "host": 0.10,
    "veins": 0.25,
    "stockwork": 0.55,
    "magnesite": 0.90,
}

# A simple susceptibility model in normalized units.
# The result is scaled by both density and bright-channel magnesite content.
CHI_BASE = 0.02
CHI_PEAK = 1.00
GAMMA = 1.45  # sharpens contrast for richer magnesite textures

VOXEL_NAME_RE = re.compile(
    r"^(?P<label>host|veins|stockwork|magnesite)_(?P<ix>\d+)_(?P<iy>\d+)_(?P<iz>\d+)\.png$"
)


@dataclass(frozen=True)
class VoxelSample:
    label: str
    ix: int
    iy: int
    iz: int
    center: Tuple[float, float, float]
    magnesite_fraction: float
    susceptibility: float
    texture_path: Path


# -----------------------------------------------------------------------------
# Texture reading and susceptibility model
# -----------------------------------------------------------------------------

def load_rgb_image(texture_path: Path) -> np.ndarray:
    """Return the image as float RGB in [0, 1]."""
    img = Image.open(texture_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def estimate_magnesite_fraction(rgb: np.ndarray) -> float:
    """
    Estimate magnesite abundance from image brightness and high color channels.

    The idea here is:
    - use the brightest channel per pixel as the main signal,
    - reinforce pixels where several channels are simultaneously high,
    - summarize using a high percentile and mean, so bright veins/ore pockets
      have a stronger impact than dark background pixels.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image array with shape (H, W, 3).")

    channel_max = np.max(rgb, axis=-1)
    channel_mean = np.mean(rgb, axis=-1)
    channel_min = np.min(rgb, axis=-1)

    bright_tail = np.quantile(channel_max, 0.90)
    bright_pixels = channel_max >= bright_tail

    if np.any(bright_pixels):
        high_mean = float(channel_max[bright_pixels].mean())
        high_mix = float(channel_mean[bright_pixels].mean())
    else:
        high_mean = float(channel_max.mean())
        high_mix = float(channel_mean.mean())

    # Encourage pixels where all channels are elevated, which often indicates
    # the more magnesite-rich parts of the synthetic textures.
    rgb_balance = float(np.clip((channel_mean - channel_min).mean(), 0.0, 1.0))
    score = 0.60 * high_mean + 0.30 * high_mix + 0.10 * (1.0 - rgb_balance)

    return float(np.clip(score, 0.0, 1.0))


def susceptibility_from_texture(rgb: np.ndarray, label: str) -> Tuple[float, float]:
    """
    Convert a voxel texture into a susceptibility value.

    The result blends:
    - a label-specific density factor,
    - the amount of magnesite inferred from the texture,
    - a non-linear boost for richer ore textures.

    Returns:
        (magnesite_fraction, susceptibility)
    """
    if label not in LABEL_DENSITY:
        raise ValueError(f"Unknown label: {label!r}")

    magnesite_fraction = estimate_magnesite_fraction(rgb)
    density = LABEL_DENSITY[label]

    # Nonlinear model:
    #   1) density controls the local ore type
    #   2) magnesite fraction controls how bright/rich the voxel looks
    #   3) gamma sharpens the difference between weak and strong ore
    susceptibility = CHI_BASE + (CHI_PEAK - CHI_BASE) * density * (magnesite_fraction ** GAMMA)

    return magnesite_fraction, float(susceptibility)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def voxel_center_from_indices(
    ix: int,
    iy: int,
    iz: int,
    bounds: Tuple[float, float],
    grid_n: int,
    voxel_size: int,
) -> Tuple[float, float, float]:
    """
    Reconstruct the voxel center using the same spacing logic as the exporter.
    """
    lo, hi = bounds
    coords = np.linspace(lo, hi, grid_n, dtype=np.float32)
    cell_size = (hi - lo) / float(grid_n)
    cube_size = voxel_size * cell_size

    origin = np.array([coords[ix], coords[iy], coords[iz]], dtype=np.float32)
    center = origin + cube_size * 0.5
    return float(center[0]), float(center[1]), float(center[2])


def parse_voxel_texture_path(path: Path) -> Tuple[str, int, int, int]:
    m = VOXEL_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Could not parse voxel texture filename: {path.name}")
    return (
        m.group("label"),
        int(m.group("ix")),
        int(m.group("iy")),
        int(m.group("iz")),
    )


def collect_voxel_samples(
    texture_dir: Path,
    bounds: Tuple[float, float],
    grid_n: int,
    voxel_size: int,
) -> List[VoxelSample]:
    samples: List[VoxelSample] = []
    texture_paths = sorted(texture_dir.glob("*.png"))

    if not texture_paths:
        raise FileNotFoundError(f"No PNG voxel textures found in {texture_dir}")

    for texture_path in texture_paths:
        label, ix, iy, iz = parse_voxel_texture_path(texture_path)
        rgb = load_rgb_image(texture_path)
        magnesite_fraction, susceptibility = susceptibility_from_texture(rgb, label)
        center = voxel_center_from_indices(ix, iy, iz, bounds, grid_n, voxel_size)

        samples.append(
            VoxelSample(
                label=label,
                ix=ix,
                iy=iy,
                iz=iz,
                center=center,
                magnesite_fraction=magnesite_fraction,
                susceptibility=susceptibility,
                texture_path=texture_path,
            )
        )

    return samples


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def write_csv(samples: Iterable[VoxelSample], out_csv: Path) -> None:
    rows = [
        "label,ix,iy,iz,x,y,z,magnesite_fraction,susceptibility,texture_path"
    ]
    for s in samples:
        x, y, z = s.center
        rows.append(
            f"{s.label},{s.ix},{s.iy},{s.iz},"
            f"{x:.6f},{y:.6f},{z:.6f},"
            f"{s.magnesite_fraction:.6f},{s.susceptibility:.6f},"
            f"{s.texture_path.as_posix()}"
        )
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def label_marker(label: str) -> str:
    return {
        "host": "o",
        "veins": "^",
        "stockwork": "s",
        "magnesite": "o",
    }.get(label, "o")


def label_size(label: str) -> float:
    return {
        "host": 12.0,
        "veins": 18.0,
        "stockwork": 24.0,
        "magnesite": 30.0,
    }.get(label, 12.0)


def plot_3d_susceptibility(
    samples: List[VoxelSample],
    output_plot: Path | None = None,
    show: bool = True,
) -> None:
    xs = np.array([s.center[0] for s in samples], dtype=np.float32)
    ys = np.array([s.center[1] for s in samples], dtype=np.float32)
    zs = np.array([s.center[2] for s in samples], dtype=np.float32)
    chis = np.array([s.susceptibility for s in samples], dtype=np.float32)
    mag = np.array([s.magnesite_fraction for s in samples], dtype=np.float32)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Sort to improve the look of the gradient in depth.
    order = np.argsort(chis)
    xs, ys, zs, chis, mag = xs[order], ys[order], zs[order], chis[order], mag[order]
    samples_sorted = [samples[i] for i in order]

    sc = ax.scatter(
        xs,
        ys,
        zs,
        c=chis,
        cmap="plasma",
        s=np.array([label_size(s.label) for s in samples_sorted], dtype=np.float32),
        alpha=0.90,
        depthshade=True,
        marker="o",
        linewidths=0.0,
    )

    # Add a faint overlay that emphasizes magnesite-rich voxels.
    rich = mag >= np.quantile(mag, 0.80)
    if np.any(rich):
        ax.scatter(
            xs[rich],
            ys[rich],
            zs[rich],
            c=chis[rich],
            cmap="viridis",
            s=36,
            alpha=0.35,
            depthshade=False,
            marker="o",
            linewidths=0.0,
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Magnetic susceptibility (normalized units)")

    ax.set_title("Voxel magnetic susceptibility (gradient style)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Keep the axes visually balanced.
    x_range = float(xs.max() - xs.min()) if len(xs) else 1.0
    y_range = float(ys.max() - ys.min()) if len(ys) else 1.0
    z_range = float(zs.max() - zs.min()) if len(zs) else 1.0
    ax.set_box_aspect((max(x_range, 1e-6), max(y_range, 1e-6), max(z_range, 1e-6)))

    # Improve visibility of the color gradient.
    ax.view_init(elev=24, azim=40)
    plt.tight_layout()

    if output_plot is not None:
        fig.savefig(output_plot, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read voxel textures, compute magnetic susceptibility, and plot the 3D result."
    )
    parser.add_argument(
        "--texture-dir",
        type=Path,
        default=Path("textures/voxels"),
        help="Directory containing per-voxel PNG textures.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        default=(-100.0, 100.0),
        help="World bounds used when the voxel exporter created the scene.",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=96,
        help="Grid resolution along each axis used in the exporter.",
    )
    parser.add_argument(
        "--voxel-size",
        type=int,
        default=5,
        help="Voxel step used in the exporter.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("susceptibility.csv"),
        help="Where to write the per-voxel susceptibility table.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("susceptibility_3d.png"),
        help="Where to save the 3D plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = collect_voxel_samples(
        texture_dir=args.texture_dir,
        bounds=(float(args.bounds[0]), float(args.bounds[1])),
        grid_n=int(args.grid_n),
        voxel_size=int(args.voxel_size),
    )

    write_csv(samples, args.output_csv)
    plot_3d_susceptibility(samples, output_plot=args.output_plot, show=not args.no_show)

    print(f"Loaded {len(samples)} voxel textures from {args.texture_dir}")
    print(f"Wrote susceptibility table to {args.output_csv}")
    print(f"Saved 3D plot to {args.output_plot}")


if __name__ == "__main__":
    main()
