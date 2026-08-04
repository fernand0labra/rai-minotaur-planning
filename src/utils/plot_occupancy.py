import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# LOAD OCCUPANCY MATRIX
# ======================================================

occupancy = np.load("data/area2/env3d_voxel_occupancy.npy")

# ======================================================
# INFO
# ======================================================

print("\n====================================")
print("Occupancy matrix shape:", occupancy.shape)
print("Occupied voxels:", int(occupancy.sum()))
print("Occupancy percentage:",
      100 * occupancy.sum() / occupancy.size)
print("====================================\n")

# ======================================================
# VISUALIZATION
# ======================================================

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")

ax.voxels(occupancy.astype(bool))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"Voxel Occupancy\n"
    f"Shape={occupancy.shape}\n"
    f"Occupied={int(occupancy.sum())}"
)

plt.tight_layout()
plt.show()