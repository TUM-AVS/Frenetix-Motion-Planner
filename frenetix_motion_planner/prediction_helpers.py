__author__ = "Maximilian Geisslinger, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""Helper functions to adjust the prediction to the needs of the frenét planner."""

import numpy as np
import os
import sys
from commonroad.scenario.obstacle import ObstacleRole
from commonroad_dc.collision.trajectory_queries import trajectory_queries
import logging

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

from frenetix_motion_planner.utility.helper_functions import create_tvobstacle, distance, ignore_vehicles_in_cone_angle
from wale_net_lite.wale_net import WaleNet
from frenetix_motion_planner.utility.sensor_model import get_visible_objects
from frenetix_motion_planner.utility import reachable_set
from frenetix_motion_planner.utility.responsibility import assign_responsibility_by_action_space

# get logger
msg_logger = logging.getLogger("Message_logger")


def load_prediction(scenario, mode, config=None):
    if mode == "walenet":
        predictor = load_walenet(scenario=scenario)
    else:
        predictor = None

    return predictor


def load_reachset(scenario, config, params_path):
    reach_set = reachable_set.ReachSet(
        scenario=scenario,
        ego_id=24,
        ego_length=config.vehicle.length,
        ego_width=config.vehicle.width,
        work_dir=params_path
    )
    return reach_set


def step_prediction(scenario, predictor, config, ego_state, time_step, occlusion_module=None, ego_id=42):
    if config.prediction.mode:
        visible_obstacles, visible_area = prediction_preprocessing(scenario, ego_state, time_step, config,
                                                                   occlusion_module, ego_id=ego_id)
    else:
        visible_obstacles = None
        visible_area = None

    if config.prediction.mode == "walenet":
        predictions = main_prediction(predictor, scenario, visible_obstacles, time_step, scenario.dt,
                                         [float(config.planning.planning_horizon)])

    elif config.prediction.mode == "ground_truth":
        predictions = get_ground_truth_prediction(visible_obstacles, scenario, time_step,
                                                  int(config.prediction.pred_horizon_in_s/config.planning.dt))
    else:
        predictions = None

    # ignore Predictions in Cone Angle
    if config.prediction.cone_angle > 0 and config.prediction.mode == "walenet":
        predictions = ignore_vehicles_in_cone_angle(predictions, ego_state, config.vehicle.length,
                                                    config.prediction.cone_angle,
                                                    config.prediction.cone_safety_dist)

    return predictions, visible_area


def step_reach_set(reach_set, scenario, x_0, predictions):
    predictions = assign_responsibility_by_action_space(scenario, x_0, predictions)
    reach_set.calc_reach_sets(x_0, list(predictions.keys()))
    return reach_set


def get_obstacles_in_radius(scenario, ego_id: int, ego_state, time_step: int, radius: float):
    """
    Get all the obstacles that can be found in a given radius.

    Args:
        scenario (Scenario): Considered Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_state (State): State of the ego vehicle.
        time_step (int) time step
        radius (float): Considered radius.

    Returns:
        [int]: List with the IDs of the obstacles that can be found in the ball with the given radius centering at the ego vehicles position.
    """
    obstacles_within_radius = []
    for obstacle in scenario.obstacles:
        # do not consider the ego vehicle
        if obstacle.obstacle_id != ego_id:
            occ = obstacle.occupancy_at_time(time_step)
            # if the obstacle is not in the lanelet network at the given time, its occupancy is None
            if occ is not None:
                # calculate the distance between the two obstacles
                dist = distance(
                    pos1=ego_state.initial_state.position,
                    pos2=obstacle.occupancy_at_time(time_step).shape.center,
                )
                # add obstacles that are close enough
                if dist < radius:
                    obstacles_within_radius.append(obstacle.obstacle_id)

    return obstacles_within_radius


def get_dyn_and_stat_obstacles(obstacle_ids: [int], scenario):
    """
    Split a set of obstacles in a set of dynamic obstacles and a set of static obstacles.

    Args:
        obstacle_ids ([int]): IDs of all considered obstacles.
        scenario: Considered scenario.

    Returns:
        [int]: List with the IDs of all dynamic obstacles.
        [int]: List with the IDs of all static obstacles.

    """
    dyn_obstacles = []
    stat_obstacles = []
    for obst_id in obstacle_ids:
        if scenario.obstacle_by_id(obst_id).obstacle_role == ObstacleRole.DYNAMIC:
            dyn_obstacles.append(obst_id)
        else:
            stat_obstacles.append(obst_id)

    return dyn_obstacles, stat_obstacles


def get_orientation_velocity_and_shape_of_prediction(
    predictions: dict, scenario, safety_margin_length=0.5, safety_margin_width=0.2
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
    """
    # go through every predicted obstacle
    obstacle_ids = list(predictions.keys())
    for obstacle_id in obstacle_ids:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # get x- and y-position of the predicted trajectory
        pred_traj = predictions[obstacle_id]['pos_list']
        pred_length = len(pred_traj)

        # there may be some predictions without any trajectory (when the obstacle disappears due to exceeding time)
        if pred_length == 0:
            del predictions[obstacle_id]
            continue

        # for predictions with only one timestep, the gradient can not be derived --> use initial orientation
        if pred_length == 1:
            pred_orientation = [obstacle.initial_state.orientation]
            pred_v = [obstacle.initial_state.velocity]
        else:
            t = [0.0 + i * scenario.dt for i in range(pred_length)]
            x = pred_traj[:, 0][0:pred_length]
            y = pred_traj[:, 1][0:pred_length]

            # calculate the yaw angle for the predicted trajectory
            dx = np.gradient(x, t)
            dy = np.gradient(y, t)
            # if the vehicle does barely move, use the initial orientation
            # otherwise small uncertainties in the position can lead to great orientation uncertainties
            if all(dxi < 0.0001 for dxi in dx) and all(dyi < 0.0001 for dyi in dy):
                init_orientation = obstacle.initial_state.orientation
                pred_orientation = np.full((1, pred_length), init_orientation)[0]
            # if the vehicle moves, calculate the orientation
            else:
                pred_orientation = np.arctan2(dy, dx)

            # get the velocity from the derivation of the position
            pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))

        # add the new information to the prediction dictionary
        predictions[obstacle_id]['orientation_list'] = pred_orientation
        predictions[obstacle_id]['v_list'] = pred_v
        obstacle_shape = obstacle.obstacle_shape
        predictions[obstacle_id]['shape'] = {
            'length': obstacle_shape.length + safety_margin_length,
            'width': obstacle_shape.width + safety_margin_width,
        }

    # return the updated predictions dictionary
    return predictions


def collision_checker_prediction(
    predictions: dict, scenario, ego_co, frenet_traj, time_step
):
    """
    Check predictions for collisions.

    Args:
        predictions (dict): Dictionary with the predictions of the obstacles.
        scenario (Scenario): Considered scenario.
        ego_co (TimeVariantCollisionObject): The collision object of the ego vehicles trajectory.
        frenet_traj (FrenetTrajectory): Considered trajectory.
        time_step (State): Current time step

    Returns:
        bool: True if the trajectory collides with a prediction.
    """
    # check every obstacle in the predictions
    for obstacle in scenario.obstacles:  #list(predictions.keys()):
        obstacle_id = obstacle.obstacle_id
        if obstacle_id not in predictions or obstacle.state_at_time(time_step).velocity > 2:
            continue
        # check if the obstacle is not a rectangle (only shape with attribute length)
        if not hasattr(scenario.obstacle_by_id(obstacle_id).obstacle_shape, 'length'):
            raise Warning('Collision Checker can only handle rectangular obstacles.')
        else:

            # get dimensions of the obstacle
            length = predictions[obstacle_id]['shape']['length']
            width = predictions[obstacle_id]['shape']['width']

            # only check for collision as long as both trajectories (frenét trajectory and prediction) are visible
            if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
                pred_traj = np.reshape(np.repeat(scenario.obstacle_by_id(obstacle_id).state_at_time(time_step).position,
                                                 len((predictions[obstacle_id]['pos_list']))),
                                       (len((predictions[obstacle_id]['pos_list'])), 2), order='F')
            else:
                pred_traj = predictions[obstacle_id]['pos_list']
            pred_length = min(len(frenet_traj.cartesian.x), len(pred_traj))
            if pred_length == 0:
                continue

            # get x, y and orientation of the prediction
            x = pred_traj[:, 0][0:pred_length]
            y = pred_traj[:, 1][0:pred_length]
            if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
                pred_orientation = np.repeat(scenario.obstacle_by_id(obstacle_id).state_at_time(time_step).orientation,
                                             len((predictions[obstacle_id]['orientation_list'])))
            else:
                pred_orientation = predictions[obstacle_id]['orientation_list']

            # create a time variant collision object for the predicted vehicle
            traj = [[x[i], y[i], pred_orientation[i]] for i in range(pred_length)]

            prediction_collision_object_raw = create_tvobstacle(
                traj_list=traj,
                box_length=length / 2,
                box_width=width / 2,
                start_time_step=time_step + 1,
            )

            # preprocess the collision object
            # if the preprocessing fails, use the raw trajectory
            (
                prediction_collision_object,
                err,
            ) = trajectory_queries.trajectory_preprocess_obb_sum(
                prediction_collision_object_raw
            )
            if err:
                prediction_collision_object = prediction_collision_object_raw

            # check for collision between the trajectory of the ego obstacle and the predicted obstacle
            collision_at = trajectory_queries.trajectories_collision_dynamic_obstacles(
                trajectories=[ego_co],
                dynamic_obstacles=[prediction_collision_object],
                method='grid',
                num_cells=32,
            )

            # if there is no collision (returns -1) return False, else True
            if collision_at[0] != -1:
                return True

    return False


def add_static_obstacle_to_prediction(
    predictions: dict, obstacle_id_list: [int], scenario, pred_horizon: int = 50
):
    """
    Add static obstacles to the prediction since predictor can not handle static obstacles.

    Args:
        predictions (dict): Dictionary with the predictions.
        obstacle_id_list ([int]): List with the IDs of the static obstacles.
        scenario (Scenario): Considered scenario.
        pred_horizon (int): Considered prediction horizon. Defaults to 50.

    Returns:
        dict: Dictionary with the predictions.
    """
    for obstacle_id in obstacle_id_list:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        fut_pos = []
        fut_cov = []
        # create a mean and covariance matrix for every time step in the prediction horizon
        for ts in range(int(pred_horizon)):
            fut_pos.append(obstacle.initial_state.position)
            fut_cov.append([[0.02, 0.0], [0.0, 0.02]])

        fut_pos = np.array(fut_pos)
        fut_cov = np.array(fut_cov)

        # add the prediction to the prediction dictionary
        predictions[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return predictions


def get_ground_truth_prediction(
    obstacle_ids: [int], scenario, time_step: int, pred_horizon: int = 50
):
    """
    Transform the ground truth to a prediction. Use this if the prediction fails.

    Args:
        obstacle_ids ([int]): IDs of the visible obstacles.
        scenario (Scenario): considered scenario.
        time_step (int): Current time step.
        pred_horizon (int): Prediction horizon for the prediction.

    Returns:
        dict: Dictionary with the predictions.
    """
    # create a dictionary for the predictions
    prediction_result = {}
    for obstacle_id in obstacle_ids:
        try:
            obstacle = scenario.obstacle_by_id(obstacle_id)
            fut_pos = []
            fut_cov = []
            fut_yaw = []
            fut_v = []
            # predict dynamic obstacles as long as they are in the scenario
            if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
                len_pred = len(obstacle.prediction.occupancy_set)
            # predict static obstacles for the length of the prediction horizon
            else:
                len_pred = pred_horizon
            # create mean and the covariance matrix of the obstacles
            for ts in range(time_step, min(pred_horizon + time_step, len_pred)):
                occupancy = obstacle.occupancy_at_time(ts)
                if occupancy is not None:
                    # create mean and covariance matrix
                    fut_pos.append(occupancy.shape.center)
                    fut_cov.append([[0.1, 0.0], [0.0, 0.1]])
                    fut_yaw.append(occupancy.shape.orientation)
                    fut_v.append(obstacle.prediction.trajectory.state_list[ts].velocity)

            fut_pos = np.array(fut_pos)
            fut_cov = np.array(fut_cov)
            fut_yaw = np.array(fut_yaw)
            fut_v = np.array(fut_v)

            shape_obs = {'length': obstacle.obstacle_shape.length, 'width': obstacle.obstacle_shape.width}
            # add the prediction for the considered obstacle
            prediction_result[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov, 'orientation_list': fut_yaw,
                                              'v_list': fut_v, 'shape': shape_obs}
        except Exception as e:
            msg_logger.warning("Could not calculate ground truth prediction: ", e)

    return prediction_result


def prediction_preprocessing(scenario, ego_state, time_step, config, occlusion_module=None, ego_id=42):
    if config.prediction.calc_visible_area:
        try:
            if config.occlusion.use_occlusion_module:
                visible_obstacles, visible_area = occlusion_module.vis_module.get_visible_area_and_objects(
                                                    time_step=time_step,ego_state=ego_state)
            else:
                visible_obstacles, visible_area = get_visible_objects(
                   scenario=scenario,
                   time_step=time_step,
                   ego_pos=ego_state.initial_state.position,
                   sensor_radius=config.prediction.sensor_radius,
                   ego_id=ego_id,
                )
            return visible_obstacles, visible_area
        except:
            msg_logger.warning("Could not calculate visible area!")
            visible_obstacles = get_obstacles_in_radius(
                scenario=scenario,
                ego_id=ego_id,
                ego_state=ego_state,
                time_step=time_step,
                radius=config.prediction.sensor_radius,
            )
            return visible_obstacles, None
    else:
        visible_obstacles = get_obstacles_in_radius(
            scenario=scenario,
            ego_id=ego_id,
            ego_state=ego_state,
            time_step=time_step,
            radius=config.prediction.sensor_radius,
        )
        return visible_obstacles, None


def main_prediction(predictor, scenario, visible_obstacles, time_step, DT, t_list):
    # get dynamic and static visible obstacles since predictor can not handle static obstacles
    (
        dyn_visible_obstacles,
        stat_visible_obstacles,
    ) = get_dyn_and_stat_obstacles(
        scenario=scenario, obstacle_ids=visible_obstacles)

    # get prediction for dynamic obstacles
    predictions = predictor.step(
        time_step=time_step,
        obstacle_id_list=dyn_visible_obstacles,
        scenario=scenario,
    )
    # create and add prediction of static obstacles
    predictions = add_static_obstacle_to_prediction(
        scenario=scenario,
        predictions=predictions,
        obstacle_id_list=stat_visible_obstacles,
        pred_horizon=max(t_list) / DT,
    )
    predictions = get_orientation_velocity_and_shape_of_prediction(
        predictions=predictions, scenario=scenario
    )

    return predictions


def load_walenet(scenario):
    predictor = WaleNet(scenario=scenario)
    return predictor

# EOF
