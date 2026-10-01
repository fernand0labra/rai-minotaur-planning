"""Damped least-squares geometric controller."""

import numpy as np

from kinematics.FK_steering_tip import FK_steering_tip


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def geometricController(q, nd, alpha):
    """Calculate joint rates that turn the tip toward ``nd``."""
    omega1 = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    omega2_local = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])

    T1, _, T3, _ = FK_steering_tip(q, alpha)
    n = T3[:3, :3] @ np.array([1.0, 0.0, 0.0])
    omega2 = T1[:3, :3] @ omega2_local
    Jw = np.column_stack((omega1, omega2))

    nd = np.asarray(nd, dtype=float).reshape(3)
    nd_norm = np.linalg.norm(nd)
    if nd_norm == 0.0:
        return np.zeros(2)
    nd /= nd_norm

    damping = 1e-2
    Jn = -_skew(n) @ Jw
    e_tan = nd - np.dot(n, nd) * n
    qdot = np.linalg.solve(
        Jn.T @ Jn + damping**2 * np.eye(2), Jn.T @ e_tan
    )

    max_velocity = 1.0
    speed = np.linalg.norm(qdot)
    if speed > max_velocity:
        qdot *= max_velocity / speed
    return qdot


geometric_controller = geometricController
