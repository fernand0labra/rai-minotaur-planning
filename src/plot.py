import json
import base64
import argparse
import numpy as np
import plotly.graph_objects as go

from skimage import measure
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

###

def _plain_json(value):
    if isinstance(value, np.ndarray):     return value.tolist()
    if isinstance(value, np.generic):     return value.item()
    if isinstance(value, dict):           return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):  return [_plain_json(item) for item in value]

    return value


def _load_final_state(map_path, run_path):
    field = np.load(map_path)

    run_data = np.load(run_path, allow_pickle=True)
    run = {key: run_data[key] for key in run_data.files}
    
    return field, run, list(run["simulation_steps"])[-1]

###

def _add_borehole(fig, collar, endpoint, name, color):
    points = np.vstack((collar, endpoint))

    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines+markers", line=dict(color=color, width=10), marker=dict(size=4, color=color), name=name, legendgroup="boreholes"))
    fig.add_trace(go.Scatter3d(x=[endpoint[0]], y=[endpoint[1]], z=[endpoint[2]], mode="markers", marker=dict(size=8, color=color, symbol="diamond", line=dict(color="black", width=1)), name=f"{name} end", showlegend=False, legendgroup="boreholes"))


def _add_route(fig, run):
    path = run["path"]
    progress = np.linspace(0.0, 1.0, len(path))
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers", line=dict(color=progress, colorscale="Turbo", width=7), marker=dict(size=3, color=progress, colorscale="Turbo"), name="Robot path (10 cm steps)", legendgroup="route", hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"))


def _add_direction_vectors(fig, run):
    desired = run["desired_directions"]
    gradients = run["gradients"].copy()

    if not len(desired):  return

    stride = max(1, len(desired)//20)
    origins = run["path"][:-1:stride]

    desired, gradients = desired[::stride], gradients[::stride]

    norms = np.linalg.norm(gradients, axis=1)
    valid = norms > 1e-12
    scale = float(run["step_length_m"])*1.5

    gradients[valid] /= norms[valid, None]

    fig.add_trace(go.Cone(x=origins[:, 0], y=origins[:, 1], z=origins[:, 2], u=gradients[:, 0], v=gradients[:, 1], w=gradients[:, 2], sizemode="absolute", sizeref=scale, anchor="tail", colorscale=[[0, "lime"], [1, "lime"]], showscale=False, name="Susceptibility-gradient direction", legendgroup="decisions"))
    fig.add_trace(go.Cone(x=origins[:, 0], y=origins[:, 1], z=origins[:, 2], u=desired[:, 0], v=desired[:, 1], w=desired[:, 2], sizemode="absolute", sizeref=scale, anchor="tail", colorscale=[[0, "cyan"], [1, "cyan"]], showscale=False, name="Magnetic tracking command", legendgroup="decisions"))


def _add_drill_cores(fig, run, visible=True, radius_scale=1.0):

    if "core_textures" not in run or not len(run["core_textures"]):
        return None

    radius = float(run["core_radius_m"])*float(radius_scale)
    vertices, faces, colors = [], [], []

    for start, end, texture in zip(run["core_starts"], run["core_ends"], run["core_textures"]):
        axis = end-start

        length = np.linalg.norm(axis)
        if length <= 1e-12:  continue
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
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], vertexcolor=colors, flatshading=False, name="0.1 m drill cores", hoverinfo="name", visible=True if visible else "legendonly", legendgroup="cores", lighting=dict(ambient=0.65, diffuse=0.8)))

        return trace_index

    return None


def _add_core_textures_2d(fig, run, visible=True):

    if "core_textures" not in run or not len(run["core_textures"]):
        return None

    textures = [np.swapaxes(texture, 0, 1) for texture in run["core_textures"]]
    image = np.concatenate(textures, axis=1)

    section_width = textures[0].shape[1]
    trace_index = len(fig.data)

    fig.add_trace(go.Image(z=image, visible=visible), row=2, col=1)

    for i in range(len(textures)+1):
        x = i*section_width-0.5
        fig.add_trace(go.Scatter(x=[x, x], y=[0, image.shape[0]-0.5], mode="lines", line=dict(width=1, color="rgba(255,255,255,0.65)"), hoverinfo="skip", showlegend=False), row=2, col=1)

    return trace_index


def _add_sensor_log(fig, field, run, field_name, row, title, x_title, color):

    if field_name not in field or "core_ends" not in run or not len(run["core_ends"]):
        return

    points = np.vstack((run["core_starts"][0], run["core_ends"]))
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))

    ix = np.abs(field["x"][:, None]-points[:, 0]).argmin(axis=0)
    iy = np.abs(field["y"][:, None]-points[:, 1]).argmin(axis=0)
    iz = np.abs(field["z"][:, None]-points[:, 2]).argmin(axis=0)

    values = field[field_name][ix, iy, iz] if field_name != "magnetic_susceptibility" else 1.0-field[field_name][ix, iy, iz]

    fig.add_trace(go.Scatter(x=distance, y=values, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=4), name=title, showlegend=False, hovertemplate=f"{title}=%{{y:.4f}}<br>distance=%{{x:.2f}} m<extra></extra>"), row=row, col=1)
    fig.update_xaxes(title_text="Drilled distance [m]", gridcolor="rgba(150,150,150,0.25)", row=row, col=1)
    fig.update_yaxes(title_text=x_title, gridcolor="rgba(150,150,150,0.25)", row=row, col=1)

###

def plot_geomodel(map_path, run_path, output, stride=2, voxel_size=6.0):
    field, run, final = _load_final_state(map_path, run_path)

    stride = max(1, int(stride))
    
    X, Y, Z = field["X"][::stride, ::stride, ::stride], field["Y"][::stride, ::stride, ::stride], field["Z"][::stride, ::stride, ::stride]
    gradient = field["gradient"][::stride, ::stride, ::stride]
    
    coords, confidence, values = np.asarray(final["voxel_coords"]), np.asarray(final["voxel_confidence"]), np.asarray(final["voxel_gradient"])
    
    mask = confidence >= 0.6
    
    fig = go.Figure()
    fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=gradient.ravel(), isomin=0.1, isomax=0.9, surface_count=5, opacity=0.25, colorscale="Viridis", showscale=False, caps=dict(x_show=False, y_show=False, z_show=False), name="Ore gradient"))
    fig.add_trace(go.Scatter3d(x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2], mode="markers", customdata=confidence[mask], marker=dict(size=voxel_size, symbol="square", color=values[mask], colorscale="Plasma", cmin=float(np.nanmin(field["gradient"])), cmax=float(np.nanmax(field["gradient"])), opacity=0.1, colorbar=dict(title="Estimated gradient")), name="Estimated gradient voxels", hovertemplate="gradient=%{marker.color:.4f}<br>confidence=%{customdata:.3f}<extra></extra>"))
    fig.update_layout(height=900, scene=dict(aspectmode="cube", xaxis=dict(title="x [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), yaxis=dict(title="y [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), zaxis=dict(title="z [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), camera=dict(eye=dict(x=1.45, y=1.45, z=1.15))), margin=dict(l=40, r=100, t=20, b=20), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0))
    
    fig_json = json.dumps(_plain_json(fig.to_dict()))
    voxel_json = json.dumps(dict(coords=coords.tolist(), confidence=confidence.tolist(), gradient=values.tolist()))
    
    plotly_js = get_plotlyjs()
    html = '<!DOCTYPE html><html><head><meta charset="utf-8"><script>'+plotly_js+'</script><style>body{font-family:Arial,sans-serif;margin:0}.controls{display:flex;gap:18px;align-items:center;padding:12px 18px;border-bottom:1px solid #ccc}input[type=range]{width:320px}</style></head><body><div class="controls"><b>Confidence</b><input id="confidence" type="range" min="0" max="1" value="0.6" step="0.01"><span id="confidenceValue">0.5</span><label><input id="gradientToggle" type="checkbox" checked> Ore gradient</label><label><input id="voxelToggle" type="checkbox" checked> Voxels</label></div><div id="plot"></div><script>const fig='+fig_json+';const voxels='+voxel_json+';Plotly.newPlot("plot",fig.data,fig.layout,{responsive:true});function update(){const c=+document.getElementById("confidence").value;document.getElementById("confidenceValue").textContent=c.toFixed(2);const keep=[];for(let i=0;i<voxels.confidence.length;i++)if(voxels.confidence[i]>=c)keep.push(i);Plotly.restyle("plot",{visible:document.getElementById("gradientToggle").checked},[0]);Plotly.restyle("plot",{x:[keep.map(i=>voxels.coords[i][0])],y:[keep.map(i=>voxels.coords[i][1])],z:[keep.map(i=>voxels.coords[i][2])],"marker.color":[keep.map(i=>voxels.gradient[i])],customdata:[keep.map(i=>voxels.confidence[i])],visible:document.getElementById("voxelToggle").checked},[1])}let pending=false;function schedule(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;update()})}document.getElementById("confidence").addEventListener("input",schedule);document.getElementById("gradientToggle").addEventListener("change",update);document.getElementById("voxelToggle").addEventListener("change",update);update();</script></body></html>'
    
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)


def plot_navigation(map_path, run_path, output, stride=2, core_radius_scale=1.0, show_gradient=True, show_core_cylinders=True, show_core_textures_2d=True):
    field, run, final = _load_final_state(map_path, run_path)
    
    stride = max(1, int(stride))
    
    X, Y, Z = field["X"][::stride, ::stride, ::stride], field["Y"][::stride, ::stride, ::stride], field["Z"][::stride, ::stride, ::stride]
    gradient = field["gradient"][::stride, ::stride, ::stride]
    scene_fig = go.Figure()
    
    gradient_visibility = True if show_gradient else "legendonly"
    scene_fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=gradient.ravel(), isomin=0.1, isomax=0.9, surface_count=5, opacity=0.25, colorscale="Viridis", showscale=False, visible=gradient_visibility, caps=dict(x_show=False, y_show=False, z_show=False), legendgroup="gradient", name="Ore gradient"))

    _add_borehole(scene_fig, run["start_collar"], run["start"], "Borehole 1", "#4056ff")
    _add_borehole(scene_fig, run["goal_collar"], run["goal"], "Borehole 2", "#ff713d")
    
    _add_route(scene_fig, run)
    _add_direction_vectors(scene_fig, run)
    cylinder_trace = _add_drill_cores(scene_fig, run, show_core_cylinders, core_radius_scale)
    
    
    fig = make_subplots(rows=5, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]], row_heights=[0.56, 0.11, 0.11, 0.11, 0.11], vertical_spacing=0.025)
    for trace in scene_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    _add_core_textures_2d(fig, run, show_core_textures_2d)
    _add_sensor_log(fig, field, run, "ultrasonic_density", 3, "Ultrasonic density", "Density [g/cm³]", "#1f77b4")
    _add_sensor_log(fig, field, run, "infrared_magnesite", 4, "Infrared magnesite", "IR Magnesite [%]", "#d62728")
    _add_sensor_log(fig, field, run, "magnetic_susceptibility", 5, "Magnetic susceptibility", "Susceptibility [-]", "#9467bd")
    
    fig.update_layout(height=1500, scene=dict(aspectmode="cube", xaxis=dict(title="x [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), yaxis=dict(title="y [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), zaxis=dict(title="z [m]", range=[-1, 1], backgroundcolor="rgb(232,238,247)"), camera=dict(eye=dict(x=1.45, y=1.45, z=1.15))), margin=dict(l=70, r=100, t=20, b=20), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0))
    
    # if cylinder_trace is not None:
    #     fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0.6, xanchor="center", y=1, buttons=[dict(label="Show Drill Core", method="restyle", args=[{"visible": True}, [cylinder_trace]]), dict(label="Hide Drill Core", method="restyle", args=[{"visible": False}, [cylinder_trace]])])])
    
    fig.write_html(output, auto_open=False, include_plotlyjs=True)


def plot_orebody(map_path, run_path, output, gradient_threshold=0.5, min_confidence=0.1, smooth_sigma=0.0, voxel_stride=2):
    field, _, final = _load_final_state(map_path, run_path)
    x, y, z = (np.asarray(field[key]) for key in ("x", "y", "z"))

    coords = np.asarray(final["voxel_coords"], dtype=np.float32)
    confidence = np.asarray(final["voxel_confidence"], dtype=np.float32)
    gradient = np.asarray(final["voxel_gradient"], dtype=np.float32)

    if len(coords) == 0:
        raise ValueError("No geomodel voxels found in the run file.")
    if not (len(coords) == len(confidence) == len(gradient)):
        raise ValueError("Final voxel coordinates, confidence, and gradient arrays must have equal lengths.")

    indices = np.column_stack((
        np.abs(x[:, None] - coords[:, 0]).argmin(axis=0),
        np.abs(y[:, None] - coords[:, 1]).argmin(axis=0),
        np.abs(z[:, None] - coords[:, 2]).argmin(axis=0),
    )).astype(int)

    shape = (len(x), len(y), len(z))
    confidence_volume = np.zeros(shape, dtype=np.float32)
    gradient_volume = np.zeros(shape, dtype=np.float32)
    index_tuple = tuple(indices.T)

    confidence_volume[index_tuple] = confidence
    gradient_volume[index_tuple] = gradient

    selected = ((confidence_volume >= min_confidence) & (gradient_volume >= gradient_threshold))
    if not np.any(selected):
        raise ValueError("No voxels pass both thresholds. Try lowering gradient_threshold or min_confidence.")

    body = selected.astype(np.float32)
    if smooth_sigma > 0:
        body = ndimage.gaussian_filter(body, sigma=smooth_sigma)

    dx, dy, dz = (float(np.mean(np.diff(axis))) for axis in (x, y, z))
    spacing = (dx, dy, dz)

    body_padded = np.pad(body, 1)
    vertices, faces, _, _ = measure.marching_cubes(body_padded, level=0.5, spacing=spacing)
    vertices += np.array([x[0] - dx, y[0] - dy, z[0] - dz])

    confidence_padded = np.pad(confidence_volume, 1)
    vertex_indices = np.rint(np.column_stack(((vertices[:, 0] - (x[0] - dx)) / dx, (vertices[:, 1] - (y[0] - dy)) / dy, (vertices[:, 2] - (z[0] - dz)) / dz))).astype(int)

    vertex_confidence = confidence_padded[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2]]

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=vertex_confidence, intensitymode="vertex", colorscale="Viridis", cmin=min_confidence, cmax=1.0, colorbar=dict(title="Confidence"), opacity=0.65, name=f"Gradient ≥ {gradient_threshold:g}", hovertemplate="Confidence: %{intensity:.3f}<extra></extra>"))

    fig.write_html(output, include_plotlyjs=True, auto_open=False)
    
###

if __name__ == "__main__":
    stride = 2

    map_path = "env/data/env.npz"
    run_path = "plan/data/planner.npz"

    core_radius_scale = 0.5
    navigation_output = "plan/data/planner.html"
    # plot_navigation(map_path, run_path, navigation_output, stride, core_radius_scale)

    voxel_size = 6.0
    geomodel_output = "state/data/geomodel.html"
    # plot_geomodel(map_path, run_path, geomodel_output, stride, voxel_size)

    gradient_threshold = 0.6
    min_confidence = 0.75
    orebody_output="state/data/orebody.html"
    # plot_orebody(map_path, run_path, orebody_output, gradient_threshold, min_confidence)