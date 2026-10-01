"""Stepwise magnetic-susceptibility navigation between two borehole ends."""

import os
import sys
import numpy as np

from config import parse_config
from world.env import Environment

# Keep the supplied kinematics and geometric controller unchanged. They may
# live in this folder or one level above it.
try:
    from kinematics.FK_steering_tip import FK_steering_tip
    from kinematics.geometricController import geometricController
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from kinematics.FK_steering_tip import FK_steering_tip
    from kinematics.geometricController import geometricController


def wrap_to_pi(angle):
    return (np.asarray(angle)+np.pi) % (2*np.pi)-np.pi


def rotation_from_x(direction):
    x_axis = np.asarray(direction, dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x_axis, helper)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(helper, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))


def magnetic_tracking_direction(env, position, goal, cfg):
    """Implement the control law from magnetic_tracking.m.

    The gradient-normal term regulates the selected susceptibility level and
    the constant tangential term advances around its contour. The goal only
    chooses which of the two tangent signs leads toward borehole 2.
    """
    goal_vector = np.asarray(goal)-np.asarray(position)
    goal_direction = goal_vector/np.linalg.norm(goal_vector)
    susceptibility = env.get_susceptibility(*position)
    gradient = env.susceptibility_gradient(*position)
    gradient_norm = np.linalg.norm(gradient)

    if gradient_norm < cfg.gradient_epsilon:
        normal = np.zeros(3)
        tangent = goal_direction
    else:
        normal = gradient/gradient_norm
        reference = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(normal, reference)
        if np.linalg.norm(tangent) < cfg.gradient_epsilon:
            reference = np.array([0.0, 1.0, 0.0])
            tangent = np.cross(normal, reference)
        tangent /= np.linalg.norm(tangent)

        # magnetic_tracking.m fixes one cross-product direction. For travel
        # between two boreholes, select its sign without adding goal attraction.
        if np.dot(tangent, goal_direction) < 0.0:
            tangent = -tangent

    normal_speed = cfg.susceptibility_gain * (
        cfg.target_susceptibility-susceptibility
    )
    command = normal_speed*normal + cfg.tangential_speed*tangent
    command_norm = np.linalg.norm(command)
    if command_norm < cfg.gradient_epsilon:
        command = goal_direction
    else:
        command /= command_norm
    return command, susceptibility, gradient


def steer_with_geometric_controller(base_rotation, desired_world, q, alpha, cfg):
    desired_local = base_rotation.T@desired_world
    desired_local /= np.linalg.norm(desired_local)
    qdot = np.ones(2)
    T1, T2, T3, T4 = FK_steering_tip(q, alpha)
    for _ in range(cfg.controller_iterations):
        heading_local = T3[:3, 0]
        if (np.linalg.norm(heading_local-desired_local) <= cfg.controller_direction_tolerance or
                np.linalg.norm(qdot) <= cfg.controller_rate_tolerance):
            break
        qdot = geometricController(q, desired_local, alpha)
        q = wrap_to_pi(q+qdot*cfg.controller_dt)
        T1, T2, T3, T4 = FK_steering_tip(q, alpha)
    heading_world = base_rotation@T3[:3, 0]
    heading_world /= np.linalg.norm(heading_world)
    return heading_world, q, (T1, T2, T3, T4)


def main(argv=None):
    cfg = parse_config(argv)
    env = Environment(cfg.map_path, target_susceptibility=cfg.target_susceptibility)
    start, goal = np.asarray(cfg.start(), dtype=float), np.asarray(cfg.goal(), dtype=float)
    position = start.copy()
    base_rotation = rotation_from_x(goal-start)
    q = np.zeros(2)
    alpha = np.deg2rad(cfg.alpha_deg)

    # The endpoint locations stay unchanged; collars lie vertically above them.
    start_collar = np.array([start[0], start[1], env.zmax])
    goal_collar = np.array([goal[0], goal[1], env.zmax])
    env.reveal_borehole(start, start_collar, cfg.reveal_radius_m)
    env.reveal_borehole(goal, goal_collar, cfg.reveal_radius_m)

    path, susceptibility_log, gradient_log = [position.copy()], [], []
    desired_log, heading_log, q_log = [], [], []
    for step in range(cfg.max_steps):
        env.robot_step(*position, radius_m=cfg.reveal_radius_m)
        distance_to_goal = np.linalg.norm(goal-position)
        if distance_to_goal <= cfg.goal_tolerance_m:
            break

        desired, susceptibility, gradient = magnetic_tracking_direction(
            env, position, goal, cfg)

        heading, q, _ = steer_with_geometric_controller(
            base_rotation,
            desired,
            q,
            alpha,
            cfg,
        )

        step_length = min(
            cfg.step_length_m,
            distance_to_goal,
        )

        # First candidate step using the actual achievable robot heading.
        candidate = env.clip_to_bounds(
            position + step_length * heading
        )

        candidate_susceptibility = env.get_susceptibility(
            *candidate
        )

        # Reject a step that would enter the ore.
        if candidate_susceptibility >= cfg.ore_threshold:
            gradient_norm = np.linalg.norm(gradient)

            if gradient_norm > cfg.gradient_epsilon:
                normal = gradient / gradient_norm

                # Remove the component pointing through the susceptibility contour.
                safe_heading = (
                    heading
                    - np.dot(heading, normal) * normal
                )

                safe_norm = np.linalg.norm(safe_heading)

                if safe_norm > cfg.gradient_epsilon:
                    safe_heading /= safe_norm

                    # Choose the tangent sign that progresses toward the goal.
                    goal_direction = goal - position

                    if np.dot(safe_heading, goal_direction) < 0.0:
                        safe_heading = -safe_heading

                    candidate = env.clip_to_bounds(
                        position
                        + step_length * safe_heading
                    )

                    candidate_susceptibility = (
                        env.get_susceptibility(*candidate)
                    )

        # Only accept the candidate if it remains outside the ore.
        if candidate_susceptibility < cfg.ore_threshold:
            next_position = candidate
        else:
            # No safe 10 cm step was found.
            # Stay in place and obtain another measurement.
            next_position = position.copy()

        susceptibility_log.append(susceptibility)
        gradient_log.append(gradient)
        desired_log.append(desired)
        heading_log.append(heading)
        q_log.append(q.copy())
        position = next_position
        path.append(position.copy())

        # Advance the base frame without adding any corner re-alignment logic.
        base_rotation = rotation_from_x(heading)
        q[:] = 0.0
    else:
        print(f"Warning: goal not reached within {cfg.max_steps} steps")

    # Reveal the final sensor footprint and store everything used by the plot.
    env.robot_step(*position, radius_m=cfg.reveal_radius_m)
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    np.savez_compressed(
        cfg.output_path,
        path=np.asarray(path), uncertainty=env.uncertainty,
        susceptibility=np.asarray(susceptibility_log), gradients=np.asarray(gradient_log),
        desired_directions=np.asarray(desired_log), headings=np.asarray(heading_log),
        q=np.asarray(q_log), start=start, goal=goal,
        start_collar=start_collar, goal_collar=goal_collar,
        target_susceptibility=cfg.target_susceptibility,
        step_length_m=cfg.step_length_m, reveal_radius_m=cfg.reveal_radius_m,
    )
    print(f"Navigation finished after {len(path)-1} step(s); final distance: "
          f"{np.linalg.norm(goal-position):.3f} m")
    if cfg.create_plot:
        from world.plot import plot
        plot(cfg.map_path, cfg.output_path, cfg.html_path)
    return np.asarray(path)


if __name__ == "__main__":
    main()