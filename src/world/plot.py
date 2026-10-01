import argparse

import numpy as np
import plotly.graph_objects as go


def add_borehole(fig, collar, endpoint, name, color):
    points = np.vstack((collar, endpoint))
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="lines+markers", line=dict(color=color, width=10),
        marker=dict(size=4, color=color), name=name))
    fig.add_trace(go.Scatter3d(
        x=[endpoint[0]], y=[endpoint[1]], z=[endpoint[2]], mode="markers",
        marker=dict(size=8, color=color, symbol="diamond",
                    line=dict(color="black", width=1)),
        name=f"{name} end", showlegend=False))


def add_route(fig, run):
    path = run["path"]
    progress = np.linspace(0.0, 1.0, len(path))
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers",
        line=dict(color=progress, colorscale="Turbo", width=7),
        marker=dict(size=3, color=progress, colorscale="Turbo"),
        name="Robot path (10 cm steps)",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"))


def add_direction_vectors(fig, run):
    desired = run["desired_directions"]
    gradients = run["gradients"].copy()
    if not len(desired):
        return
    stride = max(1, len(desired)//20)
    origins = run["path"][:-1:stride]
    desired, gradients = desired[::stride], gradients[::stride]
    norms = np.linalg.norm(gradients, axis=1)
    valid = norms > 1e-12
    gradients[valid] /= norms[valid, None]
    scale = float(run["step_length_m"])*1.5
    fig.add_trace(go.Cone(
        x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
        u=gradients[:, 0], v=gradients[:, 1], w=gradients[:, 2],
        sizemode="absolute", sizeref=scale, anchor="tail",
        colorscale=[[0, "lime"], [1, "lime"]], showscale=False,
        name="Susceptibility-gradient direction"))
    fig.add_trace(go.Cone(
        x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
        u=desired[:, 0], v=desired[:, 1], w=desired[:, 2],
        sizemode="absolute", sizeref=scale, anchor="tail",
        colorscale=[[0, "cyan"], [1, "cyan"]], showscale=False,
        name="Magnetic tracking command"))


def plot(map_path="./maps/env3d_smooth.npz",
         run_path="./maps/magnetic_navigation_run.npz",
         output="magnetic_borehole_navigation.html", stride=2):
    field = np.load(map_path)
    run_data = np.load(run_path)
    run = {key: run_data[key] for key in run_data.files}
    stride = max(1, int(stride))

    X = field["X"][::stride, ::stride, ::stride]
    Y = field["Y"][::stride, ::stride, ::stride]
    Z = field["Z"][::stride, ::stride, ::stride]
    susceptibility = field["occupancy"][::stride, ::stride, ::stride]
    uncertainty = run["uncertainty"][::stride, ::stride, ::stride]
    gradient = field["gradient"][::stride, ::stride, ::stride]

    fig = go.Figure()

    # The ore body is the selected susceptibility contour.
    fig.add_trace(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=gradient.ravel(),
        isomin=0.1, isomax=0.9, surface_count=5, opacity=0.25,
        colorscale="Viridis", showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name=f"Ore susceptibility contour"))

    # Only voxels measured by a borehole or the robot are shown. Their color
    # is the gradient magnitude, as requested, rather than a uniform reveal color.
    revealed = np.argwhere(uncertainty < 0.5)
    if len(revealed):
        revealed = revealed[::max(1, len(revealed)//18000)]
        ix, iy, iz = revealed.T
        revealed_gradient = gradient[ix, iy, iz]
        fig.add_trace(go.Scatter3d(
            x=X[ix, iy, iz], y=Y[ix, iy, iz], z=Z[ix, iy, iz],
            mode="markers", customdata=susceptibility[ix, iy, iz],
            marker=dict(size=2.2, color=revealed_gradient, colorscale="Plasma",
                        opacity=0.42, colorbar=dict(title="Gradient")),
            name="Revealed area (gradient colored)",
            hovertemplate=("gradient=%{marker.color:.4f}<br>"
                           "susceptibility=%{customdata:.4f}<extra></extra>")))

    add_borehole(fig, run["start_collar"], run["start"], "Borehole 1", "#4056ff")
    add_borehole(fig, run["goal_collar"], run["goal"], "Borehole 2", "#ff713d")
    add_route(fig, run)
    add_direction_vectors(fig, run)

    fig.update_layout(
        title=dict(text="Source Seeking Algorithm", x=0.5),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            yaxis=dict(title="y [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            zaxis=dict(title="z [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15))),
        margin=dict(l=10, r=110, t=80, b=10),
        legend=dict(x=0.5, y=-0.04, xanchor="center", orientation="h"))
    fig.write_html(output, auto_open=False)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="./maps/env3d_smooth.npz")
    parser.add_argument("--run", default="./maps/magnetic_navigation_run.npz")
    parser.add_argument("--output", default="magnetic_borehole_navigation.html")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    plot(args.map, args.run, args.output, args.stride)
