import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- import needed as a
# side effect to register the '3d' projection with matplotlib. Without this,
# fig.add_subplot(projection="3d") raises ValueError: Unknown projection '3d'
# on some environments even though the package is installed.
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    from skimage.measure import marching_cubes
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


class Plotter3D:
    """
    Draws the ore body (as an isosurface, or a fallback point cloud
    if scikit-image isn't installed) plus the planned path -- colored
    as a gradient from start to end -- and start/goal markers.

    No search-tree/visited overlay -- same reason as the 2D plotter:
    it's planner-internal debugging data, not something the final
    route plot needs, and it's the main render cost for large
    multi-leg runs.

    Rendering is optimized relative to the naive version:
      - isosurface uses Poly3DCollection (built once from the
        marching_cubes triangles) instead of plot_trisurf, which
        re-triangulates internally and is much slower for large meshes.
      - both the isosurface and voxel scatter are capped at a max
        number of primitives (faces / points) and decimated if the
        raw data exceeds it, so render time stays roughly constant
        regardless of grid resolution.
      - the voxel-scatter mask is computed on a *downsampled* copy of
        the field instead of the full array, so we're not scanning
        every cell just to throw most of them away via stride slicing.
    """

    def __init__(self, env):
        self.env = env
        self.fig = None

    def draw(self, path=None, start=None, goal=None,
              title="Planned Inspection Route", voxel_stride=None,
              max_faces=20000, max_points=20000):
        fig = plt.figure(figsize=(10, 8))
        self.fig = fig
        ax = fig.add_subplot(111, projection="3d")

        bounds = []  # collect (mins, maxs) pairs to set axis limits at the end

        if _HAVE_SKIMAGE:
            b = self._draw_isosurface(ax, max_faces)
        else:
            print("Plotter3D: scikit-image not found, falling back to voxel "
                  "scatter (pip install scikit-image for a proper isosurface, "
                  "much faster to render).")
            b = self._draw_voxel_scatter(ax, voxel_stride, max_points)
        if b is not None:
            bounds.append(b)

        if path:
            b = self._draw_path_gradient(ax, path)
            if b is not None:
                bounds.append(b)

        if start:
            ax.scatter(*start, c="cyan", s=100, edgecolors="black",
                       linewidths=1, label="start", zorder=6)
            bounds.append((np.array(start), np.array(start)))
        if goal:
            ax.scatter(*goal, c="magenta", s=100, edgecolors="black",
                       linewidths=1, label="goal", zorder=6)
            bounds.append((np.array(goal), np.array(goal)))

        if bounds:
            mins = np.min([b[0] for b in bounds], axis=0)
            maxs = np.max([b[1] for b in bounds], axis=0)
            pad = np.maximum((maxs - mins) * 0.02, 1e-6)
            ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
            try:
                ax.set_box_aspect(np.maximum(maxs - mins, 1e-6))
            except AttributeError:
                pass  # older matplotlib without set_box_aspect

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    def _draw_path_gradient(self, ax, path):
        """
        Draws the path as a Line3DCollection colored by *cumulative
        arc-length progress* (0 at start, 1 at end) rather than by
        raw point index -- so if waypoints are unevenly spaced the
        color still reflects how far along the actual route you are.
        A colorbar makes the start->end direction explicit.
        """
        p = np.asarray(path, dtype=float)
        if len(p) < 2:
            if len(p) == 1:
                ax.scatter(*p[0], c="red", s=50, zorder=5)
            return None

        seg_len = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1] if cum[-1] > 0 else 1.0
        t = cum / total

        segments = np.stack([p[:-1], p[1:]], axis=1)  # (N-1, 2, 3)
        seg_t = (t[:-1] + t[1:]) / 2.0  # color each segment by its midpoint progress

        lc = Line3DCollection(segments, cmap="plasma", linewidth=3, zorder=5)
        lc.set_array(seg_t)
        lc.set_clim(0.0, 1.0)
        ax.add_collection3d(lc)

        cbar = self.fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("path progress")
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["start", "end"])

        return p.min(axis=0), p.max(axis=0)

    def _draw_isosurface(self, ax, max_faces):
        env = self.env
        level = env.target_ore
        field_max, field_min = env.field.max(), env.field.min()
        if not (field_min < level < field_max):
            print(f"Plotter3D: target_ore={level} not within field range "
                  f"[{field_min:.2f}, {field_max:.2f}], using voxel scatter")
            return self._draw_voxel_scatter(ax, None, max_faces)

        verts, faces, _, _ = marching_cubes(env.field, level=level)

        if len(faces) > max_faces:
            stride = int(np.ceil(len(faces) / max_faces))
            faces = faces[::stride]

        verts_world = np.empty_like(verts)
        verts_world[:, 0] = env.xmin + verts[:, 0] * env.resolution_x
        verts_world[:, 1] = env.ymin + verts[:, 1] * env.resolution_y
        verts_world[:, 2] = env.zmin + verts[:, 2] * env.resolution_z

        mesh = verts_world[faces]  # (F, 3, 3) triangle vertex coords
        poly = Poly3DCollection(
            mesh, facecolor="goldenrod", edgecolor="none",
            alpha=0.35, rasterized=True,
        )
        ax.add_collection3d(poly)

        used_verts = verts_world[np.unique(faces)]
        return used_verts.min(axis=0), used_verts.max(axis=0)

    def _draw_voxel_scatter(self, ax, voxel_stride, max_points):
        env = self.env
        if voxel_stride is None:
            voxel_stride = max(1, env.nx // 40)

        # Downsample the field *before* masking -- avoids scanning every
        # cell in a potentially huge array just to discard most of them.
        field_strided = env.field[::voxel_stride, ::voxel_stride, ::voxel_stride]
        mask = np.abs(field_strided - env.target_ore) < (env.boundary_band * 2)
        idx = np.argwhere(mask)

        if len(idx) == 0:
            print("Plotter3D: no cells near target_ore to display")
            return None

        if len(idx) > max_points:
            choice = np.random.choice(len(idx), max_points, replace=False)
            idx = idx[choice]

        vals = field_strided[tuple(idx.T)]
        grid_idx = idx * voxel_stride  # map back to original grid coordinates
        pts = np.array([env.grid_to_world(*i) for i in grid_idx])

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="viridis",
                   s=4, alpha=0.3, linewidths=0, rasterized=True)

        return pts.min(axis=0), pts.max(axis=0)