import numpy as np


class Planner:

    def __init__(self, env, geo, core, controller, tracker, \
                ore_threshold=0.8, w_prospectivity=1.0, w_obstacle=1000.0, w_uncertainty=0.2, \
                step_length=0.08, reveal_radius=0.05, goal_tolerance=0.1, max_steps=1000):

        self.env = env
        self.log_gradient = []

        self.geo = geo
        self.log_uncertainty, self.log_estimated = [], []

        self.core = core
        self.log_core_start, self.log_core_end, self.log_core_texture, self.log_core_step = [], [], [], []

        self.controller = controller
        self.log_desired, self.log_heading, self.log_q = [], [], []

        self.tracker = tracker
        self.log_path = []

        self.ore_threshold = ore_threshold
        self.w_prospectivity = w_prospectivity
        self.w_obstacle = w_obstacle
        self.w_uncertainty = w_uncertainty

        self.step_length = step_length
        self.reveal_radius = reveal_radius
        self.goal_tolerance = goal_tolerance
        self.max_steps = max_steps

###

    def rotation_from_x(self, direction):
        x_axis = np.asarray(direction, dtype=float)
        x_axis /= np.linalg.norm(x_axis)

        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(x_axis, helper)) > 0.99:
            helper = np.array([0.0, 1.0, 0.0])

        y_axis = np.cross(helper, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        return np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))

    ###

    def step(self, *coords):
        gx, gy, gz = self.env._world_to_voxel(*coords)
        if self.geo.measured[gx, gy, gz]:  return

        radii = np.full(3, max(1, int(np.ceil(self.geo.decay_radius / self.env.spacing))), dtype=int)
        
        x0, x1 = max(0, gx-radii[0]), min(self.env.resolution, gx+radii[0]+1)
        y0, y1 = max(0, gy-radii[1]), min(self.env.resolution, gy+radii[1]+1)
        z0, z1 = max(0, gz-radii[2]), min(self.env.resolution, gz+radii[2]+1)

        self.geo.update((gx, gy, gz), (x0, y0, z0), (x1, y1, z1), self.env.gradient)

        self.log_uncertainty.append(self.geo.uncertainty.astype(np.float32).copy())
        self.log_estimated.append(self.geo.estimated.astype(np.float32).copy())


    def plan(self, start, goal):

        position = start.copy()
        self.log_path.append(position.copy())

        base_rotation = self.rotation_from_x(goal - start)

        q = np.zeros(2)
        alpha = np.deg2rad(self.controller.alpha_deg)

        start_borehole = np.array([start[0], start[1], self.env.bounds[1]])
        goal_borehole = np.array([goal[0], goal[1], self.env.bounds[1]])
        
        self.reveal_borehole(start, start_borehole)
        self.reveal_borehole(goal, goal_borehole)
        
        for step in range(self.max_steps):
            self.step(*position)
            
            distance_to_goal = np.linalg.norm(goal-position)
            if distance_to_goal <= self.goal_tolerance:  break
        
            gradient = self.env._get_gradient(*position)
            gradient_df = self.env._get_gradient_df(*position)

            desired = self.tracker._track(position, goal, gradient, gradient_df)
            heading, q, _ = self.controller.steer(base_rotation, desired, q, alpha)

            step_length = min(self.step_length, distance_to_goal)

            # First candidate step using the actual achievable robot heading.
            candidate = self.env._clip_to_bounds(position + step_length * heading)
            candidate_gradient = self.env._get_gradient(*candidate)

            # Reject a step that would enter the ore.
            if candidate_gradient >= self.ore_threshold:
                gradient_norm = np.linalg.norm(gradient_df)

                if gradient_norm > self.tracker.gradient_epsilon:
                    normal = gradient_df / gradient_norm

                    # Remove the component pointing through the susceptibility contour.
                    safe_heading = (heading - np.dot(heading, normal) * normal)
                    safe_norm = np.linalg.norm(safe_heading)

                    if safe_norm > self.tracker.gradient_epsilon:
                        safe_heading /= safe_norm

                        # Choose the tangent sign that progresses toward the goal.
                        goal_direction = goal - position

                        if np.dot(safe_heading, goal_direction) < 0.0:
                            safe_heading = -safe_heading

                        candidate = self.env._clip_to_bounds(position + step_length * safe_heading)
                        candidate_gradient = (self.env._get_gradient(*candidate))

            if candidate_gradient < self.ore_threshold:  next_position = candidate        # Only accept the candidate if it remains outside the ore
            else:                                        next_position = position.copy()  # No safe step was found. Stay in place and obtain another measurement

            self.log_gradient.append(gradient_df)

            self.log_desired.append(desired)
            self.log_heading.append(heading)
            self.log_q.append(q.copy())
            
            if np.linalg.norm(next_position - position) > self.tracker.gradient_epsilon:
                local_gradient = self.env.rgb._sample(position, next_position)

                self.log_core_start.append(position.copy())
                self.log_core_end.append(next_position.copy())

                self.log_core_texture.append(np.asarray(self.core.patch_core_textures(local_gradient, seed=111+step)))
                
                self.log_core_step.append(step+1)
                
            position = next_position

            self.log_path.append(position.copy())

            base_rotation = self.rotation_from_x(heading)
            q[:] = 0.0
        else:
            print(f"Warning: goal not reached within {self.max_steps} steps")

        # Reveal the final sensor footprint and store everything used by the plot.
        self.step(*position)

        self.log_gradient = np.asarray(self.log_gradient, dtype=np.float32)

        self.log_uncertainty = np.asarray(self.log_uncertainty, dtype=np.float32)
        self.log_estimated = np.asarray(self.log_estimated, dtype=np.float32)

        self.log_core_start = np.asarray(self.log_core_start)
        self.log_core_end = np.asarray(self.log_core_end)
        self.log_core_texture = np.asarray(self.log_core_texture)
        self.log_core_step = np.asarray(self.log_core_step)

        self.log_desired = np.asarray(self.log_desired)
        self.log_heading = np.asarray(self.log_heading)
        self.log_q = np.asarray(self.log_q)

        self.log_path = np.asarray(self.log_path)

        return start_borehole, goal_borehole, np.linalg.norm(goal-position)


    def reveal_borehole(self, endpoint, collar):
        """Reveal measurements along a known borehole."""
        length = np.linalg.norm(np.asarray(collar)-np.asarray(endpoint))
        sample_spacing = self.env.spacing
        for point in np.linspace(
            endpoint, collar, max(2, int(np.ceil(length / sample_spacing)) + 1)
        ):
            self.step(*point)