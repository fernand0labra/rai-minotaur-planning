"""Forward kinematics for the two-joint steering tip."""

import numpy as np
from scipy.linalg import expm


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _twist_hat(twist: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4))
    result[:3, :3] = _skew(twist[:3])
    result[:3, 3] = twist[3:]
    return result


def FK_steering_tip(q, alpha):
    """Return the four homogeneous transforms ``T1, T2, T3, T4``.

    Parameters
    ----------
    q : array-like, shape (2,)
        Joint angles in radians.
    alpha : float
        Steering-axis angle in radians.
    """
    q1, q2 = np.asarray(q, dtype=float).reshape(2)

    d1, d2 = 0.231, 0.159
    d3 = 0.121 + 0.339 + 0.12
    d4 = 0.161

    omega1 = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    omega2 = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])
    origin = np.zeros(3)
    B1 = np.r_[omega1, np.cross(-omega1, origin)]
    B2 = np.r_[omega2, np.cross(-omega2, origin)]

    def home(distance):
        transform = np.eye(4)
        transform[:3, 3] = [distance, 0.0, 0.0]
        return transform

    T1 = home(d1) @ expm(_twist_hat(B1) * q1)
    T2 = T1 @ home(d2) @ expm(_twist_hat(B2) * q2)
    T3 = T2 @ home(d3)
    T4 = T3 @ home(d4)
    return T1, T2, T3, T4


# Python-style alias.
fk_steering_tip = FK_steering_tip
