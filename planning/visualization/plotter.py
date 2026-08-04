import matplotlib.pyplot as plt


class Plotter:
    """
    Draws exactly two things on top of the ore field: the planned
    path and the start/goal markers. No search-tree/visited overlay --
    that data is planner-internal debugging info, not something the
    final route plot needs, and rendering tens of thousands of extra
    points/lines across a multi-leg inspection run is real, avoidable
    render cost for no benefit here.
    """

    def __init__(self, environment):
        self.env = environment

    def draw(self, path=None, start=None, goal=None,
              title="Planned Inspection Route",
              path_linewidth=2.5, marker_size=140):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(
            self.env.field, origin="lower", cmap="viridis",
            extent=[self.env.xmin, self.env.xmax, self.env.ymin, self.env.ymax],
        )
        plt.colorbar(ax.images[0], ax=ax, label="Ore density")

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, c="red", linewidth=path_linewidth,
                    solid_capstyle="round", label="planned path", zorder=4)

        if start:
            ax.scatter(*start, c="cyan", s=marker_size, edgecolors="black",
                       linewidths=1.2, label="start", zorder=5)
        if goal:
            ax.scatter(*goal, c="magenta", s=marker_size, edgecolors="black",
                       linewidths=1.2, label="goal", zorder=5)

        ax.legend(loc="upper left", framealpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(self.env.xmin, self.env.xmax)
        ax.set_ylim(self.env.ymin, self.env.ymax)
        fig.tight_layout()
        plt.show()