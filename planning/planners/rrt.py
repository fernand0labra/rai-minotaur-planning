import numpy as np

from utils.cost_utils import segment_collision_free


class RRTNode:
    __slots__ = ("state", "parent", "cost")

    def __init__(self, state, parent=None, cost=0.0):
        self.state = tuple(state)
        self.parent = parent
        self.cost = cost


class RRTPlanner:

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

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

    def _nearest(self, nodes, pt):

        dists = [
            np.linalg.norm(
                np.asarray(n.state) - np.asarray(pt)
            )
            for n in nodes
        ]

        return nodes[int(np.argmin(dists))]

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

    def plan(self, start, goal):

        goal_tol = self.cfg.rrt_step

        root = RRTNode(start)

        nodes = [root]

        edges = []

        for _ in range(self.cfg.rrt_max_iters):

            sample = self._sample(goal)

            nearest = self._nearest(nodes, sample)

            new_state = self._steer(nearest, sample)

            if new_state is None:
                continue

            if self.env.is_obstacle(*new_state):
                continue

            if not segment_collision_free(
                self.env,
                nearest.state,
                new_state,
            ):
                continue

            step_cost = (
                np.linalg.norm(
                    np.asarray(new_state)
                    - np.asarray(nearest.state)
                )
                + self.env.cost(*new_state)
            )

            new_node = RRTNode(
                new_state,
                parent=nearest,
                cost=nearest.cost + step_cost,
            )

            nodes.append(new_node)

            edges.append(
                (
                    nearest.state,
                    new_state,
                )
            )

            if (
                np.linalg.norm(
                    np.asarray(new_state)
                    - np.asarray(goal)
                )
                < goal_tol
            ):

                return (
                    self._reconstruct(new_node),
                    {
                        "tree": edges
                    },
                )

        print("RRT: max iterations reached, no path found")

        return (
            None,
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