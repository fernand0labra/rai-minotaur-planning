import argparse
import os
import numpy as np
import plotly.graph_objects as go

###

def add_path(fig, path):
    progress = np.linspace(0.0, 1.0, len(path))
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", line=dict(color=progress, colorscale="Plasma", width=6, colorbar=dict(title="Path progress", tickvals=[0, 1], ticktext=["start", "end"])), name="Inspection route"))
    fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers", marker=dict(size=7, color="cyan", line=dict(color="black", width=1)), name="Start"))
    fig.add_trace(go.Scatter3d(x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]], mode="markers", marker=dict(size=7, color="magenta", line=dict(color="black", width=1)), name="Goal"))

def add_backtracking(fig, backtracks):
    if len(backtracks) == 0:  return
    fig.add_trace(go.Scatter3d(x=backtracks[:, 0], y=backtracks[:, 1], z=backtracks[:, 2], mode="markers", marker=dict(size=5, color="red", symbol="diamond", line=dict(color="black", width=1)), name="Backtrack and resume"))

def plot_field(X, Y, Z, values, path, backtracks, title, output, isomin, isomax, surface_count, opacity, colorscale):
    fig = go.Figure(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=values.ravel(), isomin=isomin, isomax=isomax, surface_count=surface_count, opacity=opacity, colorscale=colorscale, caps=dict(x_show=False, y_show=False, z_show=False)))
    add_path(fig, path)
    add_backtracking(fig, backtracks)
    fig.update_layout(title=title, scene=dict(aspectmode="cube", xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    fig.write_html(output, auto_open=False)

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="./maps/env3d_smooth.npz")
    parser.add_argument("--path", default="./maps/planned_path.csv")
    parser.add_argument("--backtracks", default="./maps/backtrack_points.csv")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    data = np.load(args.map)
    stride = max(1, args.stride)
    X = data["X"][::stride, ::stride, ::stride]
    Y = data["Y"][::stride, ::stride, ::stride]
    Z = data["Z"][::stride, ::stride, ::stride]
    occupancy = data["occupancy"][::stride, ::stride, ::stride]
    gradient = data["gradient"][::stride, ::stride, ::stride]
    path = np.atleast_2d(np.loadtxt(args.path, delimiter=",", skiprows=1))
    backtracks = np.atleast_2d(np.loadtxt(args.backtracks, delimiter=",", skiprows=1)) if os.path.exists(args.backtracks) and os.path.getsize(args.backtracks) > 6 else np.empty((0, 3))

    plot_field(X, Y, Z, occupancy, path, backtracks, "Occupancy and planned path", "occupancy_path.html", 0.5, 0.5, 1, 0.45, "Viridis")
    plot_field(X, Y, Z, gradient, path, backtracks, "Gradient and planned path", "gradient_path.html", 0.1, 0.9, 5, 0.18, "Plasma")

###

if __name__ == "__main__":
    plot()
