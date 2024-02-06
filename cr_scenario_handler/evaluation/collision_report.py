import traceback
import numpy as np
from typing import List

from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.collision.collision_detection.scenario import create_collision_checker_scenario
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleRole, DynamicObstacle

from commonroad.planning.planning_problem import PlanningProblem

from cr_scenario_handler.utils.configuration import Configuration
from cr_scenario_handler.utils.multiagent_helpers import create_tvobstacle
from risk_assessment.helpers.collision_helper_function import angle_range
from risk_assessment.harm_estimation import harm_model
from risk_assessment.visualization.collision_visualization import collision_vis
from risk_assessment.utils.logistic_regression_symmetrical import \
    get_protected_inj_prob_log_reg_ignore_angle


def coll_report(ego_vehicle_list: List[DynamicObstacle], planner,
                scenario: Scenario, planning_problem: PlanningProblem,
                timestep: int, config: Configuration, collision_report_path: str):
    """Collect and present detailed information about a collision.

    :param ego_vehicle_list: List of ego obstacles for at least the last two time steps.
    :param planner: The planner used by the ego vehicle.
    :param scenario: The simulated scenario.
    :param planning_problem: The planning problem of the ego vehicle.
    :param timestep: Time step at which the collision occurred.
    :param config: Configuration containing the vehicle parameters.
    :param collision_report_path: The path to write the report to.
    """

    # check if the current state is collision-free
    vel_list = []
    # get ego position and orientation
    ego_pos = None
    ego_pos_last = None
    try:
        ego_pos = ego_vehicle_list[-1].state_at_time(timestep).position

    except AttributeError:
        print("None-type error")
        traceback.print_exc()

    (
        _,
        road_boundary,
    ) = boundary.create_road_boundary_obstacle(
        scenario=scenario,
        method="aligned_triangulation",
        axis=2,
    )

    if timestep == 0:
        ego_vel = ego_vehicle_list[-1].initial_state.velocity
        ego_yaw = ego_vehicle_list[-1].initial_state.orientation

        vel_list.append(ego_vel)
    else:
        ego_pos_last = ego_vehicle_list[-2].state_at_time(timestep).position

        delta_ego_pos = ego_pos - ego_pos_last

        ego_vel = np.linalg.norm(delta_ego_pos) / scenario.dt

        vel_list.append(ego_vel)

        ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])

    current_state_collision_object = create_tvobstacle(
        traj_list=[
            [
                ego_pos[0],
                ego_pos[1],
                ego_yaw,
            ]
        ],
        box_length=config.vehicle.length / 2,
        box_width=config.vehicle.width / 2,
        start_time_step=timestep,
    )

    # Add road boundary to collision checker
    cc = create_collision_checker_scenario(scenario)
    _, road_boundary_sg_obb = create_road_boundary_obstacle(scenario)
    cc.add_collision_object(road_boundary_sg_obb)
    cc.add_collision_object(road_boundary)

    if not cc.collide(current_state_collision_object):
        return

    # get the colliding obstacle
    obs_id = None
    for obs in scenario.obstacles:
        co = create_collision_object(obs)
        if current_state_collision_object.collide(co):
            if obs.obstacle_id != ego_vehicle_list[-1].obstacle_id:
                if obs_id is None:
                    obs_id = obs.obstacle_id
                else:
                    print("More than one collision detected")
                    return

    # Collision with boundary
    if obs_id is None:
        ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=ego_vel, coeff=planner.params_harm
        )
        total_harm = ego_harm

        print("Collision with road boundary. (Harm: {:.2f})".format(ego_harm))
        return

    # get information of colliding obstacle
    obs_pos = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=timestep)
        .shape.center
    )
    obs_pos_last = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=timestep - 1)
        .shape.center
    )
    obs_size = (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length
            * scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width
    )

    # filter initial collisions
    if timestep < 1:
        print("Collision at initial state")
        return
    if (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
            == ObstacleRole.ENVIRONMENT
    ):
        obs_vel = 0
        obs_yaw = 0
    else:
        pos_delta = obs_pos - obs_pos_last

        obs_vel = np.linalg.norm(pos_delta) / scenario.dt
        if (
                scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.DYNAMIC
        ):
            obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
        else:
            obs_yaw = scenario.obstacle_by_id(
                obstacle_id=obs_id
            ).initial_state.orientation

    # calculate crash angle
    pdof = angle_range(obs_yaw - ego_yaw + np.pi)
    rel_angle = np.arctan2(
        obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
    )
    ego_angle = angle_range(rel_angle - ego_yaw)
    obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

    # calculate harm
    ego_harm, obs_harm, ego_obj, obs_obj = harm_model(
        scenario=scenario,
        ego_vehicle_sc=ego_vehicle_list[-1],
        vehicle_params=config.vehicle,
        ego_velocity=ego_vel,
        ego_yaw=ego_yaw,
        obstacle_id=obs_id,
        obstacle_size=obs_size,
        obstacle_velocity=obs_vel,
        obstacle_yaw=obs_yaw,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        modes=planner.params_risk,
        coeffs=planner.params_harm,
    )

    # if collision report should be shown
    collision_vis(
        scenario=scenario,
        ego_vehicle=ego_vehicle_list[-1],
        destination=collision_report_path,
        ego_harm=ego_harm,
        ego_type=ego_obj.type,
        ego_v=ego_vel,
        ego_mass=ego_obj.mass,
        obs_harm=obs_harm,
        obs_type=obs_obj.type,
        obs_v=obs_vel,
        obs_mass=obs_obj.mass,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        time_step=timestep,
        modes=planner.params_risk,
        marked_vehicle=ego_vehicle_list[-1].obstacle_id,
        planning_problem=planning_problem,
        global_path=None,
        driven_traj=None,
    )