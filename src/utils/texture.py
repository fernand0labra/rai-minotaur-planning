import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RGB = Tuple[int, int, int]
ColorRange = Tuple[RGB, RGB]


@dataclass
class TextureConfig:
    width: int = 512
    height: int = 512
    seed: int = 7
    geoseed: int = 13

    # Approximate ranges inspired by the reference images
    host_rock_ranges: List[ColorRange] = field(default_factory=lambda: [
        ((18, 16, 16), (42, 38, 34)),
        ((28, 24, 20), (72, 62, 54)),
        ((40, 34, 28), (92, 80, 66)),
        ((54, 48, 42), (118, 106, 92)),
    ])

    earthy_ranges: List[ColorRange] = field(default_factory=lambda: [
        ((78, 56, 38), (132, 102, 72)),
        ((96, 72, 46), (154, 120, 84)),
        ((116, 84, 56), (186, 142, 102)),
        ((92, 82, 66), (150, 136, 118)),
    ])

    magnesite_ranges: List[ColorRange] = field(default_factory=lambda: [
        ((214, 214, 214), (255, 255, 255)),
        ((194, 192, 188), (239, 237, 231)),
        ((226, 222, 214), (252, 248, 242)),
        ((205, 206, 208), (242, 244, 246)),
    ])

    # Background / host material
    host_noise_scale: int = 96
    host_octaves: int = 4

    # Veins: distorted edge-to-edge flowing paths
    vein_count: int = 8
    vein_thickness: Tuple[int, int] = (5, 15)
    vein_wiggle: float = 0.5
    vein_blur: float = 0.5
    vein_secondary_weight: float = 0.5
    vein_opacity: float = 1.0

    # Stockwork: rocky host with small irregular closed magnesite spots
    stockwork_spot_count: int = 350
    stockwork_spot_size: Tuple[int, int] = (5, 25)
    stockwork_spot_irregularity: float = 0.17
    stockwork_spot_blur: float = 0.5
    stockwork_spot_secondary_weight: float = 0.5
    stockwork_opacity: float = 1.0

    # Magnesite: whole image is magnesite with shaded noise
    magnesite_noise_scale: int = 96
    magnesite_octaves: int = 4
    magnesite_secondary_weight: float = 0.5
    magnesite_opacity: float = 1.0


# -----------------------------
# Color utilities
# -----------------------------
def sample_color_from_ranges(rng: np.random.Generator, ranges: Sequence[ColorRange]) -> np.ndarray:
    low, high = ranges[int(rng.integers(0, len(ranges)))]
    return np.array([rng.integers(low[i], high[i] + 1) for i in range(3)], dtype=np.float32)


def build_palette(rng: np.random.Generator, ranges: Sequence[ColorRange], n: int = 8) -> np.ndarray:
    """
    Easy substitution point for palette logic.
    Replace this function to change the color distribution model.
    """
    return np.stack([sample_color_from_ranges(rng, ranges) for _ in range(n)], axis=0)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a * (1.0 - t) + b * t


# -----------------------------
# Noise helpers
# -----------------------------
def smooth_noise(width: int, height: int, cell_size: int, rng: np.random.Generator) -> np.ndarray:
    grid_w = max(2, width // cell_size + 2)
    grid_h = max(2, height // cell_size + 2)
    grid = rng.random((grid_h, grid_w), dtype=np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    img = img.resize((width, height), resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def fractal_noise(width: int, height: int, base_scale: int, octaves: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    total = 0.0
    amp = 1.0
    scale = max(4, base_scale)

    for _ in range(octaves):
        out += amp * smooth_noise(width, height, scale, rng)
        total += amp
        amp *= 0.5
        scale = max(4, scale // 2)

    return out / max(total, 1e-6)


# -----------------------------
# Backgrounds
# -----------------------------
def create_host_background(cfg: TextureConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Dark rocky host material for veins and stockwork.
    """
    n1 = fractal_noise(cfg.width, cfg.height, cfg.host_noise_scale, cfg.host_octaves, rng)
    n2 = fractal_noise(cfg.width, cfg.height, max(8, cfg.host_noise_scale // 3), 3, rng)

    host_palette = build_palette(rng, cfg.host_rock_ranges, n=10)
    earthy_palette = build_palette(rng, cfg.earthy_ranges, n=8)

    a = host_palette[int(rng.integers(0, len(host_palette)))]
    b = host_palette[int(rng.integers(0, len(host_palette)))]
    base = lerp(a[None, None, :], b[None, None, :], n1[..., None])

    earth = earthy_palette[int(rng.integers(0, len(earthy_palette)))]
    tint = np.clip((n2 - 0.45) * 0.45, 0.0, 0.30)[..., None]
    base = lerp(base, earth[None, None, :], tint)

    # A subtle cylindrical shading so the core feels like a drill core photo.
    xx = np.linspace(-1.0, 1.0, cfg.width, dtype=np.float32)
    shading = (1.0 - (xx ** 2))[None, :] * 18.0
    edge_dark = (np.abs(xx) ** 1.7)[None, :] * 24.0

    base = np.clip(base + shading[..., None] - edge_dark[..., None], 0, 255)
    return base.astype(np.float32)


def create_magnesite_background(cfg: TextureConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Full-image magnesite with shade variation and noise.
    """
    n1 = fractal_noise(cfg.width, cfg.height, cfg.magnesite_noise_scale, cfg.magnesite_octaves, rng)
    n2 = fractal_noise(cfg.width, cfg.height, max(8, cfg.magnesite_noise_scale // 3), 3, rng)

    mag_palette = build_palette(rng, cfg.magnesite_ranges, n=12)
    earthy_palette = build_palette(rng, cfg.earthy_ranges, n=6)

    a = mag_palette[int(rng.integers(0, len(mag_palette)))]
    b = mag_palette[int(rng.integers(0, len(mag_palette)))]
    base = lerp(a[None, None, :], b[None, None, :], n1[..., None])

    earth = earthy_palette[int(rng.integers(0, len(earthy_palette)))]
    tint = np.clip((n2 - 0.62) * 0.18, 0.0, 0.12)[..., None]
    base = lerp(base, earth[None, None, :], tint)

    # Soft cloudy texture
    cloud = (fractal_noise(cfg.width, cfg.height, 28, 3, rng) - 0.5)[..., None] * 18.0
    base = np.clip(base + cloud, 0, 255)
    return base.astype(np.float32)


# -----------------------------
# Veins: edge-to-edge flowing paths
# -----------------------------
def edge_to_edge_path(
    width: int,
    height: int,
    rng: np.random.Generator,
    orientation: str = "horizontal",
    wiggle: float = 0.2,
    n_samples: int = 120,
    phase: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Produces a distorted line that starts on one side of the image and exits another side.
    The path expands and compresses like a river.
    """
    if phase is None:
        phase = float(rng.uniform(0, 2 * math.pi))

    pts = []
    if orientation == "horizontal":
        y0 = float(rng.uniform(height * 0.08, height * 0.92))
        amp = float(rng.uniform(height * 0.05, height * 0.18))
        freq = float(rng.uniform(1.0, 1.5))
        drift = float(rng.uniform(-height * 0.06, height * 0.06))
        for i, x in enumerate(np.linspace(0, width - 1, n_samples)):
            t = i / max(n_samples - 1, 1)
            center = y0 + drift * (t - 0.5)
            y = center + amp * math.sin((t * freq * 2 * math.pi) + phase)
            y += rng.normal(0.0, wiggle * height * 0.02)
            pts.append((float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))))
    else:
        x0 = float(rng.uniform(width * 0.08, width * 0.92))
        amp = float(rng.uniform(width * 0.05, width * 0.18))
        freq = float(rng.uniform(1.0, 1.5))
        drift = float(rng.uniform(-width * 0.06, width * 0.06))
        for i, y in enumerate(np.linspace(0, height - 1, n_samples)):
            t = i / max(n_samples - 1, 1)
            center = x0 + drift * (t - 0.5)
            x = center + amp * math.sin((t * freq * 2 * math.pi) + phase)
            x += rng.normal(0.0, wiggle * width * 0.02)
            pts.append((float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))))

    return pts


def create_vein_mask(cfg: TextureConfig, rng: np.random.Generator) -> Image.Image:
    """
    Main substitution point for vein geometry.
    Creates flowing, distorted, edge-to-edge bands with variable width.
    """
    mask = Image.new("L", (cfg.width, cfg.height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(cfg.vein_count):
        orientation = "horizontal" if rng.random() < 0.5 else "vertical"
        path = edge_to_edge_path(
            cfg.width,
            cfg.height,
            rng,
            orientation=orientation,
            wiggle=cfg.vein_wiggle,
            n_samples=int(rng.integers(90, 160)),
        )

        amp = float(rng.uniform(0.5, 0.75))
        base_w = float(rng.integers(cfg.vein_thickness[0], cfg.vein_thickness[1] + 1))

        # Draw the vein as many short segments with varying width, like a river that expands/compresses.
        for i in range(len(path) - 1):
            t = i / max(len(path) - 2, 1)
            flow = 0.25 + 0.25 * math.sin(t * 2 * math.pi * rng.uniform(0.8, 2.2) + rng.uniform(0, 2 * math.pi))
            local_w = max(1, int(base_w * (0.40 + 2.0 * flow * amp)))

            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            draw.line((x0, y0, x1, y1), fill=255, width=local_w)

        # Add larger blobs along the path to create the "compressed" sections.
        # for _ in range(int(rng.integers(3, 8))):
        #     idx = int(rng.integers(0, len(path)))
        #     x, y = path[idx]
        #     r = int(rng.integers(max(2, base_w // 3), max(3, base_w + 3)))
        #     draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    if cfg.vein_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(cfg.vein_blur))
    return mask


# -----------------------------
# Stockwork: rocky host + irregular closed magnesite spots
# -----------------------------
def irregular_blob_points(
    cx: float,
    cy: float,
    radius: float,
    rng: np.random.Generator,
    n_vertices: int = 14,
    irregularity: float = 0.35,
) -> List[Tuple[float, float]]:
    pts = []
    phase = float(rng.uniform(0, 2 * math.pi))
    for i in range(n_vertices):
        a = 2 * math.pi * i / n_vertices + phase
        r = radius * (1.0 + rng.normal(0.0, irregularity))
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r
        pts.append((x, y))
    return pts


def create_stockwork_mask(cfg: TextureConfig, rng: np.random.Generator) -> Image.Image:
    """
    Stockwork = dark rocky material with small, irregular, closed magnesite spots.
    """
    mask = Image.new("L", (cfg.width, cfg.height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(cfg.stockwork_spot_count):
        cx = float(rng.uniform(0, cfg.width - 1))
        cy = float(rng.uniform(0, cfg.height - 1))
        r = float(rng.integers(cfg.stockwork_spot_size[0], cfg.stockwork_spot_size[1] + 1))
        n_vertices = int(rng.integers(6, 16))
        irregularity = float(rng.uniform(cfg.stockwork_spot_irregularity * 0.6, cfg.stockwork_spot_irregularity * 1.3))

        pts = irregular_blob_points(cx, cy, r, rng, n_vertices=n_vertices, irregularity=irregularity)
        draw.polygon(pts, fill=255)

        # A few spots get a second smaller inner shape to mimic nested magnesite patches.
        if rng.random() < 0.25:
            inner_r = max(2.0, r * rng.uniform(0.35, 0.65))
            inner_pts = irregular_blob_points(cx + rng.normal(0, r * 0.1), cy + rng.normal(0, r * 0.1),
                                              inner_r, rng, n_vertices=max(5, n_vertices - 2),
                                              irregularity=irregularity * 0.7)
            draw.polygon(inner_pts, fill=255)

    if cfg.stockwork_spot_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(cfg.stockwork_spot_blur))
    return mask


# -----------------------------
# Magnesite: whole-image texture
# -----------------------------
def create_magnesite_mask(cfg: TextureConfig, rng: np.random.Generator) -> Image.Image:
    """
    Whole image is magnesite, so this returns a full mask.
    """
    return Image.new("L", (cfg.width, cfg.height), 255)


# -----------------------------
# Colorization / composition
# -----------------------------
def make_color_layer(
    mask: Image.Image,
    rng: np.random.Generator,
    primary_ranges: Sequence[ColorRange],
    secondary_ranges: Sequence[ColorRange],
    secondary_weight: float,
    detail_noise_scale: int,
) -> np.ndarray:
    """
    Main substitution point for pixel distribution behavior.
    """
    m = np.asarray(mask, dtype=np.float32) / 255.0
    h, w = m.shape

    noise = fractal_noise(w, h, detail_noise_scale, 3, rng)
    primary = build_palette(rng, primary_ranges, n=10)
    secondary = build_palette(rng, secondary_ranges, n=10)

    idx_p = rng.integers(0, len(primary), size=(h, w))
    idx_s = rng.integers(0, len(secondary), size=(h, w))

    c1 = primary[idx_p]
    c2 = secondary[idx_s]

    blend = np.clip(secondary_weight + (noise - 0.5) * 0.34, 0.0, 1.0)[..., None]
    layer = lerp(c1, c2, blend)

    jitter = (rng.random((h, w, 3), dtype=np.float32) - 0.5) * 8.0
    layer = np.clip(layer + jitter, 0, 255)

    return layer * m[..., None]


def overlay(base: np.ndarray, layer: np.ndarray, mask: Image.Image, opacity: float) -> np.ndarray:
    a = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None] * opacity
    return np.clip(base * (1.0 - a) + layer * a, 0, 255)


# -----------------------------
# Main generator
# -----------------------------
def generate_texture(mode: str = "veins", cfg: Optional[TextureConfig] = None) -> Image.Image:
    if cfg is None:
        cfg = TextureConfig()

    rng = np.random.default_rng(cfg.seed)
    georng = np.random.default_rng(cfg.geoseed)
    mode = mode.lower().strip()

    if mode == "magnesite":
        base = create_host_background(cfg, rng)
        mask = create_magnesite_mask(cfg, rng)
        layer = make_color_layer(
            mask=mask,
            rng=rng,
            primary_ranges=cfg.magnesite_ranges,
            secondary_ranges=cfg.earthy_ranges,
            secondary_weight=cfg.magnesite_secondary_weight,
            detail_noise_scale=24,
        )
        composed = overlay(base, layer, mask, cfg.magnesite_opacity)

        # Extra subtle shading across the full image
        x = np.linspace(-1.0, 1.0, cfg.width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        vignette = (xx**2 + yy**2)[..., None] * 8.0
        composed = np.clip(composed + 10.0 - vignette, 0, 255)

    else:
        base = create_host_background(cfg, rng)

        if mode == "host":
            cfg.vein_count = 0
            mask = create_vein_mask(cfg, georng)
            layer = make_color_layer(
                mask=mask,
                rng=rng,
                primary_ranges=cfg.magnesite_ranges,
                secondary_ranges=cfg.earthy_ranges,
                secondary_weight=cfg.vein_secondary_weight,
                detail_noise_scale=20,
            )
            composed = overlay(base, layer, mask, cfg.vein_opacity)

        elif mode == "veins":
            mask = create_vein_mask(cfg, georng)
            layer = make_color_layer(
                mask=mask,
                rng=rng,
                primary_ranges=cfg.magnesite_ranges,
                secondary_ranges=cfg.earthy_ranges,
                secondary_weight=cfg.vein_secondary_weight,
                detail_noise_scale=20,
            )
            composed = overlay(base, layer, mask, cfg.vein_opacity)

        elif mode == "stockwork":
            mask = create_stockwork_mask(cfg, georng)
            layer = make_color_layer(
                mask=mask,
                rng=rng,
                primary_ranges=cfg.magnesite_ranges,
                secondary_ranges=cfg.earthy_ranges,
                secondary_weight=cfg.stockwork_spot_secondary_weight,
                detail_noise_scale=18,
            )
            composed = overlay(base, layer, mask, cfg.stockwork_opacity)

            # Keep most of the image rocky and let magnesite stay in the closed spots.
            dark_noise = fractal_noise(cfg.width, cfg.height, 56, 3, rng)
            composed = np.clip(composed * (0.92 + 0.08 * dark_noise[..., None]), 0, 255)

        else:
            raise ValueError("mode must be one of: 'veins', 'stockwork', 'magnesite'")

    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    for i in range(1, 2):
        cfg = TextureConfig(width=512, height=512, seed=12, geoseed=i)

        # "veins", "stockwork", "magnesite"
        mode = "veins"
        img = generate_texture(mode=mode, cfg=cfg)
        img.save(f"texture{i}.png")