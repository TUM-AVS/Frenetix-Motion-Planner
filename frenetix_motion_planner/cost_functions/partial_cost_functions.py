__author__ = "Alexander Hobmeier, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import frenetix_motion_planner.trajectories
from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import Scenario
from scipy.integrate import simps
import commonroad_dc.pycrcc as pycrcc
from shapely.geometry import LineString, Point
from cr_scenario_handler.utils import helper_functions as hf
from scipy.spatial.distance import cdist
from risk_assessment.collision_probability import (
    get_collision_probability_fast, get_inv_mahalanobis_dist
)
from risk_assessment.risk_costs import get_responsibility_cost
from risk_assessment.risk_costs import calc_risk


def acceleration_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                      planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the acceleration cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    acceleration_sq = np.square(acceleration)
    cost = simps(acceleration_sq, dx=trajectory.dt)
    
    return cost


def jerk_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
              planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the jerk cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    jerk = np.diff(acceleration) / trajectory.dt
    jerk_sq = np.square(jerk)
    cost = simps(jerk_sq, dx=trajectory.dt)

    return cost


def lateral_jerk_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                  planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    cost = trajectory.trajectory_lat.squared_jerk_integral(trajectory.dt)
    return cost


def longitudinal_jerk_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                  planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    cost = trajectory.trajectory_long.squared_jerk_integral(trajectory.dt)
    return cost


def steering_angle_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                        planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the steering angle cost for the given trajectory.
    """
    raise NotImplementedError


def steering_rate_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                       planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the steering rate cost for the given trajectory.
    """
    raise NotImplementedError


def yaw_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
             planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the yaw cost for the given trajectory.
    """
    raise NotImplementedError


def lane_center_offset_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                            planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculate the average distance of the trajectory to the center line of a lane.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        lanelet_network (LaneletNetwork): Considered lanelet network.

    Returns:
        float: Average distance from the trajectory to the center line of a lane.
    """
    dist = 0.0
    for i, (x, y) in enumerate(zip(trajectory.cartesian.x, trajectory.cartesian.y)):
        # find the lanelet of every position
        lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([np.array([x, y])])
        if len(lanelet_ids[0]) > 0:
            lanelet_id = lanelet_ids[0][0]
            lanelet_obj = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            # find the distance of the current position to the center line of the lanelet
            dist += dist_to_nearest_point(lanelet_obj.center_vertices, [x, y])
        # theirs should always be a lanelet for the current position
        # otherwise the trajectory should not be valid and no costs are calculated
        else:
            dist += 5

    return dist / len(trajectory.cartesian.x)


def velocity_offset_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                         planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the Velocity Offset cost.
    """
    vel = trajectory.cartesian.v
    half_idx = int(len(vel) / 2)
    cost = np.sum(np.abs(vel[half_idx:-1] - desired_speed))
    cost += np.abs(((vel[-1] - desired_speed) ** 2))

    return float(cost)


def longitudinal_velocity_offset_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                                      planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the Velocity Offset cost.
    """
    raise NotImplementedError


def orientation_offset_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                            planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the Orientation Offset cost.
    """
    theta = trajectory.curvilinear.theta
    theta = np.diff(theta) / trajectory.dt
    theta = np.square(theta)
    cost = simps(theta, dx=trajectory.dt)

    return cost


def distance_to_reference_path_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                                    planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the Distance to Reference Path costs.

    Args:
        trajectory (FrenetTrajectory): Frenét trajectory to be checked.

    Returns:
        float: Average distance of the trajectory to the given path.
    """
    # Costs of gerneral deviation from ref path && Additional costs for deviation at final planning point from ref path
    d = trajectory.curvilinear.d
    cost = (np.sum(np.abs(d)) + np.abs(d[-1]) * 5) / len(d + 4)

    return float(cost)


def distance_to_obstacles_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                               planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the Distance to Obstacle cost.
    """
    cost = 0.0
    traj_coords = np.transpose([trajectory.cartesian.x, trajectory.cartesian.y])
    for obstacle in scenario.obstacles:
        state = obstacle.state_at_time(planner.x_0.time_step)
        if state is not None:
            obs_pos = np.reshape([state.position[0], state.position[1]], (1, 2))
            dists = cdist(traj_coords, obs_pos, metric='euclidean')
            cost += np.sum(np.reciprocal(dists ** 2))

    return float(cost)


def path_length_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                     planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the path length cost for the given trajectory.
    """
    velocity = trajectory.cartesian.v
    cost = simps(velocity, dx=trajectory.dt)
    return cost


def time_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
              planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the time cost for the given trajectory.
    """
    raise NotImplementedError


def inverse_duration_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                          planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculates the inverse time cost for the given trajectory.
    """
    return 1 / time_costs(trajectory)


def velocity_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                   planner=None, scenario=None, desired_speed: float=0) -> float:
    """
    Calculate the costs for the velocity of the given trajectory.

    Args:
        trajectory (FrenetTrajectory): Considered trajectory.
        ego_state (State): Current state of the ego vehicle.
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.
        dt (float): Time step size of the scenario.

    Returns:
        float: Costs for the velocity of the trajectory.

    """
    # if the goal area is reached then just consider the goal velocity
    if reached_target_position(np.array([trajectory.cartesian.x[0], trajectory.cartesian.y[0]]), planner.goal_area):
        # if the planning problem has a target velocity
        if hasattr(planner.planning_problem.goal.state_list[0], "velocity"):
            return abs(
                (
                    planner.planning_problem.goal.state_list[0].velocity.start
                    + (
                        planner.planning_problem.goal.state_list[0].velocity.end
                        - planner.planning_problem.goal.state_list[0].velocity.start
                    )
                    / 2
                )
                - np.mean(trajectory.cartesian.v)
            )
        # otherwise prefer slow trajectories
        else:
            return float(np.mean(trajectory.cartesian.v))

    # if the goal is not reached yet, try to reach it
    # get the center points of the possible goal positions
    goal_centers = []
    # get the goal lanelet ids if they are given directly in the planning problem
    if (
        hasattr(planner.planning_problem.goal, "lanelets_of_goal_position")
        and planner.planning_problem.goal.lanelets_of_goal_position is not None
    ):
        goal_lanelet_ids = planner.planning_problem.goal.lanelets_of_goal_position[0]
        for lanelet_id in goal_lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            n_center_vertices = len(lanelet.center_vertices)
            goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
    elif hasattr(planner.planning_problem.goal.state_list[0], "position"):
        # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
        if hasattr(planner.planning_problem.goal.state_list[0].position, "center"):
            goal_centers.append(planner.planning_problem.goal.state_list[0].position.center)
    # if it is a survival scenario with no goal areas, no velocity can be proposed
    else:
        return 0.0

    # get the distances to the previous found goal positions
    distances = np.sqrt(np.sum((np.array(goal_centers) - planner.x_0.position) ** 2, axis=1))
    avg_dist = np.mean(distances)

    # get the remaining time
    _, max_remaining_time_steps = hf.calc_remaining_time_steps(
        planning_problem=planner.planning_problem,
        ego_state_time=planner.x_0.time_step,
        t=0.0,
        dt=trajectory.dt,
    )
    remaining_time = max_remaining_time_steps * trajectory.dt

    # if there is time remaining, calculate the difference between the average desired velocity and the velocity of the trajectory
    if remaining_time > 0.0:
        avg_desired_velocity = avg_dist / remaining_time
        avg_v = np.mean(trajectory.cartesian.v)
        cost = abs(avg_desired_velocity - avg_v)
    # if the time limit is already exceeded, prefer fast velocities
    else:
        cost = 30.0 - np.mean(trajectory.cartesian.v)

    return float(cost)


def reached_target_position(pos: np.array, goal_area) -> bool:
    """
    Check if the given position is in the goal area of the planning problem.

    Args:
        pos (np.array): Position to be checked.
        goal_area (ShapeGroup): Shape group representing the goal area.

    Returns:
        bool: True if the given position is in the goal area.
    """
    # if there is no goal area (survival scenario) return True
    if goal_area is None:
        return True

    point = pycrcc.Point(x=pos[0], y=pos[1])

    # check if the point of the position collides with the goal area
    if point.collide(goal_area) is True:
        return True

    return False


def dist_to_nearest_point(center_vertices: np.ndarray, pos: np.array) -> float:
    """
    Find the closest distance of a given position to a polyline.

    Args:
        center_vertices (np.ndarray): Considered polyline.
        pos (np.array): Conisdered position.

    Returns:
        float: Closest distance between the polyline and the position.
    """
    # create a point and a line, project the point on the line and find the nearest point
    # shapely used
    point = Point(pos)
    linestring = LineString(center_vertices)
    project = linestring.project(point)
    nearest_point = linestring.interpolate(project)

    return hf.distance(pos, np.array([nearest_point.x, nearest_point.y]))


def prediction_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                     planner=None, scenario=None, desired_speed: float=0):

    # prediction_costs_raw = get_collision_probability_fast(
    #     traj=trajectory,
    #     predictions=planner.predictions,
    #     vehicle_params=planner.vehicle_params
    # )
    prediction_costs_raw = get_inv_mahalanobis_dist(traj=trajectory, predictions=planner.predictions,
                                                    vehicle_params=planner.vehicle_params)

    pred_costs = 0
    for key in prediction_costs_raw:
        pred_costs += np.sum(prediction_costs_raw[key])

    return pred_costs


def responsibility_costs(trajectory: frenetix_motion_planner.trajectories.TrajectorySample,
                     planner=None, scenario=None, desired_speed: float=0):
    if planner.predictions is not None and planner.reachset is not None:
        ego_risk_dict, obst_risk_dict, ego_harm_dict, obst_harm_dict, ego_risk, obst_risk = calc_risk(
            traj=trajectory,
            ego_state=planner.rp.x_0,
            predictions=planner.predictions,
            scenario=planner.scenario,
            ego_id=24,
            vehicle_params=planner.vehicle_params,
            road_boundary=planner.rp.road_boundary,
            params_harm=planner.rp.params_harm,
            params_risk=planner.rp.params_risk,
        )
        trajectory._ego_risk = ego_risk
        trajectory._obst_risk = obst_risk

        responsibility_cost, bool_contain_cache = get_responsibility_cost(
            scenario=planner.scenario,
            traj=trajectory,
            ego_state=planner.rp.x_0,
            obst_risk_max=obst_risk_dict,
            predictions=planner.predictions,
            reach_set=planner.rp.reach_set
        )
    else:
        responsibility_cost = 0.0

    return responsibility_cost
