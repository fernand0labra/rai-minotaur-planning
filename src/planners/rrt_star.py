import numpy as np

from utils.cost_utils import segment_collision_free
from planners.rrt import RRTNode

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


class RRTStarPlanner:

    def __init__(self, env, cfg):

        self.env = env
        self.cfg = cfg

        self.rebuild_interval = max(
            1,
            cfg.rrt_star_rebuild_interval,
        )

        if not _HAVE_SCIPY:
            print(
                "RRT*: WARNING - scipy not installed. "
                "Using brute-force nearest neighbour search."
            )

    def _sample(self, goal):

        if np.random.rand() < self.cfg.rrt_goal_bias:
            return goal

        if self.env.is3d:

            return (
                np.random.uniform(self.env.xmin, self.env.xmax),
                np.random.uniform(self.env.ymin, self.env.ymax),
                np.random.uniform(self.env.zmin, self.env.zmax),
            )

        return (
            np.random.uniform(self.env.xmin, self.env.xmax),
            np.random.uniform(self.env.ymin, self.env.ymax),
        )

    def _steer(self, from_node, to_pt):

        start = np.asarray(from_node.state)
        goal = np.asarray(to_pt)

        direction = goal - start
        dist = np.linalg.norm(direction)

        if dist < 1e-9:
            return None

        step = min(self.cfg.rrt_step, dist)

        direction /= dist

        return tuple(start + direction * step)

    def _edge_cost(self, a, b):

        return (
            np.linalg.norm(
                np.asarray(b) - np.asarray(a)
            )
            + self.env.cost(*b)
        )

    def plan(self, start, goal):

        goal_tol = self.cfg.rrt_step
        radius = self.cfg.rrt_star_radius

        root = RRTNode(start)

        nodes = [root]

        coords = [list(root.state)]

        edges = []

        best_goal_node = None

        tree = None

        pending = [0]

        def nearest(pt):

            best_node = None
            best_dist = float("inf")

            if tree is not None:

                d, idx = tree.query(pt)

                best_node = nodes[idx]
                best_dist = d

            for i in pending:

                n = nodes[i]

                d = np.linalg.norm(
                    np.asarray(n.state)
                    - np.asarray(pt)
                )

                if d < best_dist:

                    best_dist = d
                    best_node = n

            return best_node

        def near(pt, r):

            result = []

            if tree is not None:

                idxs = tree.query_ball_point(pt, r)

                result.extend(
                    nodes[i]
                    for i in idxs
                )

            for i in pending:

                n = nodes[i]

                if (
                    np.linalg.norm(
                        np.asarray(n.state)
                        - np.asarray(pt)
                    )
                    <= r
                ):
                    result.append(n)

            return result

        progress_every = 2000

        for i in range(self.cfg.rrt_max_iters):

            if (i + 1) % progress_every == 0:

                print(
                    f"RRT*: {i+1}/{self.cfg.rrt_max_iters} "
                    f"iterations, "
                    f"{len(nodes)} nodes"
                )

            sample = self._sample(goal)

            nearest_node = nearest(sample)

            new_state = self._steer(
                nearest_node,
                sample,
            )

            if new_state is None:
                continue

            if self.env.is_obstacle(*new_state):
                continue

            if not segment_collision_free(
                self.env,
                nearest_node.state,
                new_state,
            ):
                continue

            neighborhood = near(
                new_state,
                radius,
            )

            best_parent = nearest_node

            best_cost = (
                nearest_node.cost
                + self._edge_cost(
                    nearest_node.state,
                    new_state,
                )
            )

            for cand in neighborhood:

                if not segment_collision_free(
                    self.env,
                    cand.state,
                    new_state,
                ):
                    continue

                c = (
                    cand.cost
                    + self._edge_cost(
                        cand.state,
                        new_state,
                    )
                )

                if c < best_cost:

                    best_cost = c
                    best_parent = cand

            new_node = RRTNode(
                new_state,
                parent=best_parent,
                cost=best_cost,
            )

            nodes.append(new_node)

            coords.append(
                list(new_node.state)
            )

            edges.append(
                (
                    best_parent.state,
                    new_state,
                )
            )

            pending.append(
                len(nodes) - 1
            )

            for cand in neighborhood:

                if cand is best_parent:
                    continue

                if not segment_collision_free(
                    self.env,
                    new_state,
                    cand.state,
                ):
                    continue

                c = (
                    new_node.cost
                    + self._edge_cost(
                        new_state,
                        cand.state,
                    )
                )

                if c < cand.cost:

                    cand.parent = new_node
                    cand.cost = c

            if (
                np.linalg.norm(
                    np.asarray(new_state)
                    - np.asarray(goal)
                )
                < goal_tol
            ):

                if (
                    best_goal_node is None
                    or new_node.cost < best_goal_node.cost
                ):
                    best_goal_node = new_node

            if (
                _HAVE_SCIPY
                and len(pending) >= self.rebuild_interval
            ):

                tree = cKDTree(
                    np.asarray(coords)
                )

                pending = []

        if best_goal_node is None:

            print(
                "RRT*: max iterations reached"
            )

            return (
                None,
                {
                    "tree": edges
                },
            )

        return (
            self._reconstruct(best_goal_node),
            {
                "tree": edges
            },
        )

    def _reconstruct(self, node):

        path = []

        while node is not None:

            path.append(node.state)

            node = node.parent

        path.reverse()

        return path