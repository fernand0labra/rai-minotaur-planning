"""
Creates a coarse copy of the ore field for fast planner development.
Keep the full-resolution npz for Unity/final visualization; plan on
this one while iterating on cost weights and thresholds.

Usage:
    python3 utils/downsample_map.py maps/ore_field_2d.npz maps/ore_field_2d_lowres.npz 10
    (last arg = stride, e.g. 10 turns 2000x2000 into 200x200)
"""
import sys
import numpy as np


def downsample(src_path, dst_path, stride):
    data = np.load(src_path)

    Z = data["Z"][::stride, ::stride]
    x = data["x"][::stride]
    y = data["y"][::stride]
    X, Y = np.meshgrid(x, y)

    np.savez(dst_path, X=X, Y=Y, Z=Z, x=x, y=y)
    print(f"{src_path} ({data['Z'].shape}) -> {dst_path} ({Z.shape})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 downsample_map.py <src.npz> <dst.npz> <stride>")
        sys.exit(1)

    downsample(sys.argv[1], sys.argv[2], int(sys.argv[3]))