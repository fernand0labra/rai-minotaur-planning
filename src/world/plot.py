import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def add_drill_cores(fig, run, visible=True, radius_scale=1.0):
    """Wrap each sampled texture around its corresponding drilled segment."""
    if "core_textures" not in run or not len(run["core_textures"]):
        return None
    radius = float(run["core_radius_m"])*float(radius_scale)
    vertices, faces, colors = [], [], []
    for start, end, texture in zip(run["core_starts"], run["core_ends"], run["core_textures"]):
        axis = end-start
        length = np.linalg.norm(axis)
        if length <= 1e-12:
            continue
        axis /= length
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        rows, columns = texture.shape[:2]
        offset = len(vertices)
        for i, center in enumerate(np.linspace(start, end, rows)):
            for j, angle in enumerate(np.linspace(0.0, 2*np.pi, columns, endpoint=False)):
                vertices.append(center+radius*(np.cos(angle)*u+np.sin(angle)*v))
                colors.append(f"rgb({texture[i, j, 0]},{texture[i, j, 1]},{texture[i, j, 2]})")
        for i in range(rows-1):
            for j in range(columns):
                a = offset+i*columns+j
                b = offset+i*columns+(j+1)%columns
                c = offset+(i+1)*columns+(j+1)%columns
                d = offset+(i+1)*columns+j
                faces.extend(((a, b, c), (a, c, d)))
    if vertices:
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        trace_index = len(fig.data)
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                               vertexcolor=colors, flatshading=False, name="0.1 m drill cores",
                               hoverinfo="name", visible=True if visible else "legendonly",
                               legendgroup="cores", lighting=dict(ambient=0.65, diffuse=0.8)))
        return trace_index
    return None


def add_core_textures_2d(fig, run, visible=True):
    """Add the unwrapped textures beneath the 3-D scene."""
    if "core_textures" not in run or not len(run["core_textures"]):
        return None
    textures = [np.swapaxes(texture, 0, 1) for texture in run["core_textures"]]
    image = np.concatenate(textures, axis=1)
    section_width = textures[0].shape[1]
    lengths = np.linalg.norm(run["core_ends"]-run["core_starts"], axis=1)
    distance = np.concatenate(([0.0], np.cumsum(lengths)))
    trace_index = len(fig.data)
    fig.add_trace(go.Image(z=image, visible=visible), row=2, col=1)
    for i in range(len(textures)+1):
        x = i*section_width-0.5
        fig.add_trace(go.Scatter(x=[x, x], y=[0, image.shape[0]-0.5], mode="lines",
                                 line=dict(width=1, color="rgba(255,255,255,0.65)"),
                                 hoverinfo="skip", showlegend=False), row=2, col=1)
    return trace_index


def add_sensor_log(fig, field, run, field_name, row, title, x_title, color):
    if field_name not in field or "core_ends" not in run or not len(run["core_ends"]):
        return
    points = np.vstack((run["core_starts"][0], run["core_ends"]))
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    ix = np.abs(field["x"][:, None]-points[:, 0]).argmin(axis=0)
    iy = np.abs(field["y"][:, None]-points[:, 1]).argmin(axis=0)
    iz = np.abs(field["z"][:, None]-points[:, 2]).argmin(axis=0)
    values = field[field_name][ix, iy, iz] if field_name != "magnetic_susceptibility" else 1.0-field[field_name][ix, iy, iz]
    fig.add_trace(go.Scatter(x=distance, y=values, mode="lines+markers",
                             line=dict(color=color, width=2), marker=dict(size=4),
                             name=title, showlegend=False,
                             hovertemplate=f"{title}=%{{x:.4f}}<br>distance=%{{y:.2f}} m<extra></extra>"), row=row, col=1)
    fig.update_xaxes(title_text="Drilled distance [m]", gridcolor="rgba(150,150,150,0.25)", row=row, col=1)
    fig.update_yaxes(title_text=x_title, gridcolor="rgba(150,150,150,0.25)", row=row, col=1)


def plot(map_path="./maps/env3d_smooth.npz",
         run_path="./maps/magnetic_navigation_run.npz",
         output="magnetic_borehole_navigation.html", stride=2, show_gradient=True,
         show_core_cylinders=True, show_core_textures_2d=True, core_radius_scale=1.0):
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

    scene_fig = go.Figure()

    # The ore body is the selected susceptibility contour.
    gradient_visibility = True if show_gradient else "legendonly"
    scene_fig.add_trace(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=gradient.ravel(),
        isomin=0.1, isomax=0.9, surface_count=5, opacity=0.25,
        colorscale="Viridis", showscale=False, visible=gradient_visibility,
        caps=dict(x_show=False, y_show=False, z_show=False), legendgroup="gradient",
        legendgrouptitle_text="Ore gradient", name=f"Ore susceptibility contour"))

    # Only voxels measured by a borehole or the robot are shown. Their color
    # is the gradient magnitude, as requested, rather than a uniform reveal color.
    revealed = np.argwhere(uncertainty < 0.5)
    if len(revealed):
        revealed = revealed[::max(1, len(revealed)//18000)]
        ix, iy, iz = revealed.T
        revealed_gradient = gradient[ix, iy, iz]
        scene_fig.add_trace(go.Scatter3d(
            x=X[ix, iy, iz], y=Y[ix, iy, iz], z=Z[ix, iy, iz],
            mode="markers", customdata=susceptibility[ix, iy, iz],
            marker=dict(size=2.2, color=revealed_gradient, colorscale="Plasma",
                        opacity=0.42),
            name="Revealed area (gradient colored)", visible=gradient_visibility,
            legendgroup="gradient",
            hovertemplate=("gradient=%{marker.color:.4f}<br>"
                           "susceptibility=%{customdata:.4f}<extra></extra>")))

    add_borehole(scene_fig, run["start_collar"], run["start"], "Borehole 1", "#4056ff")
    add_borehole(scene_fig, run["goal_collar"], run["goal"], "Borehole 2", "#ff713d")
    add_route(scene_fig, run)
    cylinder_trace = add_drill_cores(scene_fig, run, show_core_cylinders, core_radius_scale)
    add_direction_vectors(scene_fig, run)

    fig = make_subplots(rows=5, cols=1,
                        specs=[[{"type": "scene"}], [{"type": "xy"}], [{"type": "xy"}],
                               [{"type": "xy"}], [{"type": "xy"}]],
                        row_heights=[0.48, 0.12, 0.12, 0.12, 0.12], vertical_spacing=0.025)

    for trace in scene_fig.data:
        fig.add_trace(trace, row=1, col=1)
    add_core_textures_2d(fig, run, show_core_textures_2d)
    add_sensor_log(fig, field, run, "ultrasonic_density", 3, "Ultrasonic density", "Ultrasonic Density [g/cm³]", "#1f77b4")
    add_sensor_log(fig, field, run, "infrared_magnesite", 4, "Infrared magnesite", "IR Magnesite Content [%]", "#d62728")
    add_sensor_log(fig, field, run, "magnetic_susceptibility", 5, "Magnetic susceptibility", "Magnetic Susceptibility", "#9467bd")

    fig.update_layout(
        height=2200,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            yaxis=dict(title="y [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            zaxis=dict(title="z [m]", backgroundcolor="rgb(232,238,247)", gridcolor="white"),
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15))),
        xaxis=dict(visible=False), yaxis=dict(title="Core circumference", autorange="reversed"),
        margin=dict(l=80, r=110, t=0, b=0),
        legend=dict(x=0.5, y=0.975, xanchor="center", orientation="h", groupclick="togglegroup"))
    if cylinder_trace is not None:
        fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0.5, xanchor="center", y=1,
                                            buttons=[dict(label="Show Drill Core", method="restyle",
                                                          args=[{"visible": True}, [cylinder_trace]]),
                                                     dict(label="Hide Drill Core", method="restyle",
                                                          args=[{"visible": False}, [cylinder_trace]])])])
    fig.write_html(output, auto_open=False)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="./maps/env3d_smooth.npz")
    parser.add_argument("--run", default="./maps/magnetic_navigation_run.npz")
    parser.add_argument("--output", default="magnetic_borehole_navigation.html")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--hide-gradient", action="store_true")
    parser.add_argument("--hide-core-cylinders", action="store_true")
    parser.add_argument("--hide-core-textures-2d", action="store_true")
    parser.add_argument("--core-radius-scale", type=float, default=1.0)
    args = parser.parse_args()
    plot(args.map, args.run, args.output, args.stride, not args.hide_gradient,
         not args.hide_core_cylinders, not args.hide_core_textures_2d, args.core_radius_scale)
