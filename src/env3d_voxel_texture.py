"""Procedural drill-core textures driven by a normalized ore gradient.

The public API intentionally consists of the four material generators and
``patch_core_textures``.  All images are deterministic for a given seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _noise(shape: tuple[int, int], rng: np.random.Generator, scales=(3, 10, 35)) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for scale, weight in zip(scales, (0.50, 0.32, 0.18)):
        out += weight * ndimage.gaussian_filter(rng.normal(size=shape), scale, mode="wrap")
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def _rgb(base: tuple[int, int, int], variation: np.ndarray, amount: float) -> np.ndarray:
    color = np.asarray(base, dtype=np.float32)[None, None, :]
    return np.clip(color * (1.0 + amount * (variation[..., None] - 0.5)), 0, 255)


def _cracks(shape: tuple[int, int], rng: np.random.Generator, count: int, width: float) -> np.ndarray:
    """Return a fine, branching fracture mask."""
    h, w = shape
    ink = np.zeros(shape, dtype=bool)
    for _ in range(count):
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        angle = rng.uniform(-np.pi, np.pi)
        for _ in range(int(rng.uniform(25, 110))):
            yi, xi = int(round(y)), int(round(x))
            if not (0 <= yi < h and 0 <= xi < w):
                break
            ink[yi, xi] = True
            angle += rng.normal(0, 0.18)
            y += np.sin(angle) * rng.uniform(0.7, 1.5)
            x += np.cos(angle) * rng.uniform(0.7, 1.5)
    return ndimage.gaussian_filter(ink.astype(np.float32), width)


def host(shape: tuple[int, int] = (512, 512), seed: int = 12) -> Image.Image:
    """Dark brown/gray, fine-grained host rock like the reference core."""
    rng = np.random.default_rng(seed)
    fine = _noise(shape, rng, (2, 12, 55))
    bands = ndimage.gaussian_filter(rng.normal(size=shape), (22, 4))
    bands = (bands - bands.min()) / max(np.ptp(bands), 1e-6)
    image = _rgb((70, 66, 55), 0.72 * fine + 0.28 * bands, 0.72)
    hairlines = _cracks(shape, rng, max(8, shape[0] // 30), 0.45)
    image = image * (1 - 0.28 * hairlines[..., None])
    return Image.fromarray(image.astype(np.uint8), "RGB")


def stockwork(shape: tuple[int, int] = (512, 512), seed: int = 12) -> Image.Image:
    """Host rock cut by a dense, pale, intersecting fracture network."""
    rng = np.random.default_rng(seed)
    image = np.asarray(host(shape, seed + 101), dtype=np.float32)
    network = _cracks(shape, rng, max(35, shape[0] // 7), 1.05)
    network += 0.55 * _cracks(shape, rng, max(20, shape[0] // 12), 2.2)
    network = np.clip(network * 2.8, 0, 1)[..., None]
    pale = np.array([190, 193, 184], dtype=np.float32)
    image = image * (1 - 0.78 * network) + pale * (0.78 * network)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def veins(shape: tuple[int, int] = (512, 512), seed: int = 12) -> Image.Image:
    """Host rock with sparse, continuous white mineral veins."""
    rng = np.random.default_rng(seed)
    image = np.asarray(host(shape, seed + 211), dtype=np.float32)
    mask = _cracks(shape, rng, max(10, shape[0] // 32), 2.8)
    mask = np.clip(mask * 5.5, 0, 1)[..., None]
    mineral = np.array([226, 229, 221], dtype=np.float32)
    image = image * (1 - mask) + mineral * mask
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def magnesite(shape: tuple[int, int] = (512, 512), seed: int = 12) -> Image.Image:
    """Bright massive magnesite with faint gray marbling and fractures."""
    rng = np.random.default_rng(seed)
    cloud = _noise(shape, rng, (5, 24, 70))
    image = _rgb((224, 229, 225), cloud, 0.22)
    fractures = np.clip(_cracks(shape, rng, max(12, shape[0] // 24), 0.65) * 2.5, 0, 1)
    image *= 1 - 0.35 * fractures[..., None]
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def patch_core_textures(
    ore_gradient: np.ndarray,
    seed: int = 12,
    thresholds: tuple[float, float, float] = (0.5, 0.7, 0.9),
    slice_axis: int = 1,
    slice_index: int | None = None,
) -> Image.Image:
    """Patch the four textures into one core image using ore-gradient ranges.

    A 3-D scalar field is converted to a 2-D section through its centre. Use
    ``slice_axis`` and ``slice_index`` to choose another section.

    Values are normalized to [0, 1] when needed.  Ranges are host [0,.5),
    stockwork [.5,.7), veins [.7,.9), and magnesite [.9,1].  A mildly
    distorted classification field prevents artificial ruler-straight contacts.
    """
    field = np.asarray(ore_gradient, dtype=np.float32)
    if field.ndim == 3:
        axis = int(slice_axis) % 3
        index = field.shape[axis] // 2 if slice_index is None else int(slice_index)
        if not 0 <= index < field.shape[axis]:
            raise IndexError(
                f"slice_index {index} is outside axis {axis} with size {field.shape[axis]}"
            )
        field = np.take(field, index, axis=axis)
    if field.ndim == 1:
        field = np.repeat(field[:, None], 256, axis=1)
    if field.ndim != 2:
        raise ValueError("ore_gradient must be a 1-D log, 2-D slice, or 3-D scalar field")
    finite = np.isfinite(field)
    if not finite.any():
        raise ValueError("ore_gradient contains no finite values")
    lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    if lo < 0 or hi > 1:
        field = (field - lo) / max(hi - lo, 1e-6)
    field = np.nan_to_num(field, nan=0.0).clip(0, 1)

    rng = np.random.default_rng(seed)
    contact_noise = _noise(field.shape, rng, (4, 18, 55)) - 0.5
    classified = np.clip(field + 0.045 * contact_noise, 0, 1)
    textures = [
        np.asarray(host(field.shape, seed + 1)),
        np.asarray(stockwork(field.shape, seed + 2)),
        np.asarray(veins(field.shape, seed + 3)),
        np.asarray(magnesite(field.shape, seed + 4)),
    ]
    labels = np.digitize(classified, thresholds)
    core = np.empty((*field.shape, 3), dtype=np.uint8)
    for label, texture in enumerate(textures):
        core[labels == label] = texture[labels == label]

    # Cylindrical illumination and darkened rim, matching a photographed core.
    x = np.linspace(-1, 1, field.shape[1], dtype=np.float32)
    light = np.clip(1.03 - 0.22 * x * x, 0.74, 1.03)[None, :, None]
    core = np.clip(core.astype(np.float32) * light, 0, 255).astype(np.uint8)
    return Image.fromarray(core, "RGB")


def _demo_gradient(height: int, width: int) -> np.ndarray:
    """Reference-like log: host-dominant with two broad magnesite intervals."""
    y = np.linspace(0, 1, height)[:, None]
    g = 0.28 + 0.16 * np.sin(10 * np.pi * y)
    for center, spread, peak in ((0.17, 0.055, 0.95), (0.53, 0.045, 1.0), (0.80, 0.025, 0.78)):
        g += peak * np.exp(-0.5 * ((y - center) / spread) ** 2)
    return np.repeat(np.clip(g, 0, 1), width, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate four geological textures and a patched core preview")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_core"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    square = (512, 512)
    for name, generator in (("host", host), ("stockwork", stockwork), ("veins", veins), ("magnesite", magnesite)):
        generator(square, args.seed).save(args.output_dir / f"{name}.png")
    gradient = _demo_gradient(args.height, args.width)
    patch_core_textures(gradient, args.seed).save(args.output_dir / "core_model.png")
    np.save(args.output_dir / "ore_gradient.npy", gradient)


# if __name__ == "__main__":
#     main()
