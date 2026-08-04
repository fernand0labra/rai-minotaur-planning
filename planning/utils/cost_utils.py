import numpy as np

def edge_cost(env, a, b):
    """
    Shared edge cost for 2D and 3D.
    a,b can be:
        2D: (x,y)
        3D: (x,y,z)
    Same cost function is used by all planners.
    """
    distance = np.linalg.norm(
        np.asarray(b) - np.asarray(a)
    )
    return float(distance + env.cost(*b))

def segment_collision_free(env, a, b, n_checks=10):
    """
    Collision check for both 2D and 3D.
    Samples the line between two points.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    for i in range(1, n_checks + 1):
        t = i / n_checks
        point = a + t * (b - a)
        if env.is_obstacle(*point):
            return False
    return True