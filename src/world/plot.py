import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None


class Plotter3D:
    def __init__(self, env):
        self.env = env

    def draw(self, path=None, start=None, goal=None, title="Planned Inspection Route"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")

        self._ore(ax)
        if path is not None and len(path):
            self._path(fig, ax, path)
        if start is not None:
            ax.scatter(*start, s=80, label="start")
        if goal is not None:
            ax.scatter(*goal, s=80, label="goal")

        ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def _ore(self, ax):
        env = self.env
        level = env.target_susceptibility

        if marching_cubes is None:
            idx = np.argwhere(np.abs(env.susceptibility - level) <= env.boundary_band)
            if len(idx):
                pts = np.array([env.voxel_to_world(*p) for p in idx[::max(1, len(idx) // 10000)]])
                ax.scatter(
                    df["x"],
                    df["y"],
                    df["z"],
                    c=df["susceptibility"],
                    cmap="viridis",
                )
            return

        verts, faces, _, _ = marching_cubes(env.susceptibility, level=level)
        verts[:, 0] = env.xmin + verts[:, 0] * env.resolution_x
        verts[:, 1] = env.ymin + verts[:, 1] * env.resolution_y
        verts[:, 2] = env.zmin + verts[:, 2] * env.resolution_z
        ax.plot_trisurf(*verts.T, triangles=faces, alpha=.25, linewidth=0)

    @staticmethod
    def _path(fig, ax, path):
        p = np.asarray(path, dtype=float)
        if len(p) == 1:
            ax.scatter(*p[0], s=40)
            return

        segments = np.stack((p[:-1], p[1:]), axis=1)
        line = Line3DCollection(segments, cmap="plasma", linewidth=3)
        line.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection3d(line)
        fig.colorbar(line, ax=ax, shrink=.6, pad=.1, label="path progress")