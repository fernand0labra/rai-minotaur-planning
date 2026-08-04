import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

###

def scalar_field_circular(x, y, center=(0.0, 0.0),
                 R1=0.6, R2=1.2,
                 irregularity=0.3, seed=42,
                 smooth=0.05):
    """
    Scalar field with three nested *irregular* regions.
    Each boundary is generated as:
        r(theta) = R + irregularity * f(theta)
    where f(theta) is a random smooth Fourier-like perturbation.
    """
    rng = np.random.default_rng(seed)
    cx, cy = center

    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # --- Helper to make a smooth random perturbation ---
    def random_shape(theta, base_radius):
        n_freqs = 6
        freqs = np.arange(1, n_freqs + 1)[:, None, None]  # shape (6,1,1)
        coeffs1 = rng.normal(0, 1, (n_freqs, 1, 1))
        coeffs2 = rng.normal(0, 1, (n_freqs, 1, 1))
        # Broadcast works now: (6,1,1) × (400,400) → (6,400,400)
        f = np.sum(coeffs1 * np.sin(freqs * theta) + coeffs2 * np.cos(freqs * theta), axis=0)
        f = f / np.max(np.abs(f))  # normalize
        return base_radius + irregularity * f

    # Irregular radii for each boundary
    R1_irreg = random_shape(theta, R1)
    R2_irreg = random_shape(theta, R2)

    # Smooth transitions (sigmoid)
    t1 = 1.0 / (1.0 + np.exp((r - R1_irreg) / smooth))
    t2 = 1.0 / (1.0 + np.exp((r - R2_irreg) / smooth))

    value = t1 + t2
    return value

def scalar_field_elliptical(x, y, center=(0.0, 0.0),
                 R1=0.6, R2=1.2, R3=2.0,
                 anisotropy=(1.5, 0.8),
                 warp_strength=0.3,
                 irregularity=0.25,
                 seed=42, smooth=0.05):
    """
    Scalar field with three nested *non-circular irregular* regions.
    
    The base shape is an ellipse, optionally warped and perturbed.
    """
    rng = np.random.default_rng(seed)
    cx, cy = center
    ax, ay = anisotropy

    # Shift and scale to introduce anisotropy (stretching)
    dx = (x - cx) / ax
    dy = (y - cy) / ay

    # Base elliptical 'radius'
    r = np.sqrt(dx**2 + dy**2)

    # Non-circular warp using sinusoidal distortions in x/y directions
    wx = warp_strength * np.sin(2.5 * dx) + warp_strength * np.cos(1.5 * dy)
    wy = warp_strength * np.sin(3.1 * dy) - warp_strength * np.cos(2.0 * dx)
    r_warp = np.sqrt((dx + wx)**2 + (dy + wy)**2)

    # Random smooth perturbation (low-frequency noise)
    n_freqs = 5
    freqs = np.arange(1, n_freqs + 1)[:, None, None]
    coeffs1 = rng.normal(0, 1, (n_freqs, 1, 1))
    coeffs2 = rng.normal(0, 1, (n_freqs, 1, 1))
    theta_like = np.arctan2(dy, dx)
    f = np.sum(coeffs1 * np.sin(freqs * theta_like) +
               coeffs2 * np.cos(freqs * theta_like), axis=0)
    f /= np.max(np.abs(f))

    # Irregular, noncircular nested boundaries
    R1_field = R1 + irregularity * f
    R2_field = R2 + irregularity * f
    R3_field = R3 + irregularity * f

    # Smooth region transitions
    t1 = 1.0 / (1.0 + np.exp((r_warp - R1_field) / smooth))
    t2 = 1.0 / (1.0 + np.exp((r_warp - R2_field) / smooth))
    t3 = 1.0 / (1.0 + np.exp((r_warp - R3_field) / smooth))

    value = t1 + t2 + t3
    return value

if True:

    nx, ny = 2000, 2000
    x = np.linspace(-100, 100, nx)
    y = np.linspace(-100, 100, ny)
    X, Y = np.meshgrid(x, y)

    Z = scalar_field_circular(X, Y, center=(20.0, 10.0),
                    R1= 55.0, R2=45.0,
                    irregularity=15.0, seed=13, smooth=1.0)
    os.makedirs("maps", exist_ok=True)

    # Save the environment
    np.savez(
        "maps/ore_field_2d.npz",
        X=X,
        Y=Y,
        Z=Z,
        x=x,
        y=y
    )

    print("Environment saved to maps/ore_field_2d.npz")

    plt.figure(figsize=(7, 6))
    plt.imshow(Z, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),
            interpolation='bilinear', aspect='equal')
    plt.xlabel("x"); plt.ylabel("y")
    plt.colorbar(label="scalar value")
    plt.contour(X, Y, Z, levels=[0.5, 1.5], colors='k', linewidths=1)
    plt.show()


if False:

    nx, ny = 2000, 2000
    x = np.linspace(-100, 100, nx)
    y = np.linspace(-100, 100, ny)
    X, Y = np.meshgrid(x, y)

    # Z = scalar_field_elliptical(X, Y,
    #                 center=(10.0, 10.0),
    #                 R1=75.0, R2=50.0, R3=25.0,
    #                 anisotropy=(1.0, 1.0),
    #                 warp_strength=0.0,
    #                 irregularity=15.0,
    #                 seed=13, smooth=1.0)

    Z = scalar_field_elliptical(X, Y,
                    center=(5.0, 10.0),
                    R1=80.0, R2=65.0, R3=25.0,
                    anisotropy=(1.0, 1.0),
                    warp_strength=0.0,
                    irregularity=20.0,
                    seed=7, smooth=1.0)

    # Z = scalar_field_elliptical(X, Y,
    #                 center=(10.0, 0.0),
    #                 R1=50.0, R2=30.0, R3=25.0,
    #                 anisotropy=(1.0, 2.0),
    #                 warp_strength=0.0,
    #                 irregularity=20.0,
    #                 seed=13, smooth=1.0)

    # Z = scalar_field_elliptical(X, Y,
    #                 center=(0.0, 10.0),
    #                 R1=45.0, R2=30.0, R3=25.0,
    #                 anisotropy=(2.0, 1.0),
    #                 warp_strength=0.0,
    #                 irregularity=5.0,
    #                 seed=13, smooth=1.0)

    plt.figure(figsize=(7,6))
    plt.imshow(Z, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()),
            interpolation='bilinear', aspect='equal')
    plt.colorbar(label="scalar value")
    plt.contour(X, Y, Z, levels=[0.5, 1.5, 2.5], colors='k', linewidths=1)

    if True:

        ax = plt.gca()
        bx, by = 5, 5  # block size
        bx_grid = np.arange(-100, 100, bx)
        by_grid = np.arange(-100, 100, by)

        for xi in bx_grid:
            for yi in by_grid:
                # Sample scalar field at block center
                xc = xi + bx/2
                yc = yi + by/2  
                # np.array([[xc]]), np.array([[yc]]),
                val = scalar_field_elliptical(np.array([[xc]]), np.array([[yc]]),
                                        center=(5.0, 10.0),
                                        R1=80.0, R2=65.0, R3=25.0,
                                        anisotropy=(1.0, 1.0),
                                        warp_strength=0.0,
                                        irregularity=20.0,
                                        seed=7, smooth=1.0)
                # Normalize to [0,1] for colormap
                norm_val = (val - Z.min()) / (Z.max() - Z.min())
                color = plt.cm.viridis(norm_val[0,0])
                rect = Rectangle((xi, yi), bx, by, color=color)
                ax.add_patch(rect)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("env2d.png")