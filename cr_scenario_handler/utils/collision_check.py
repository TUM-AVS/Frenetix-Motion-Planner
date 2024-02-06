__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np

from commonroad.scenario.obstacle import ObstacleRole
from commonroad.prediction.prediction import TrajectoryPrediction, SetBasedPrediction

import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object


def get_scenario_dynamic_obstacles_as_tvo(scenario):
    dyn_obstacles_list = list()
    for dyn_obst in scenario.dynamic_obstacles:
        if isinstance(dyn_obst.prediction, TrajectoryPrediction):
            co_raw = create_collision_object(dyn_obst.prediction)
            # preprocess using obb hull for continuous collision detection
            co, err = trajectory_queries.trajectory_preprocess_obb_sum(co_raw)
            if err:
                co = co_raw
            dyn_obstacles_list.append(co)
        else:
            if isinstance(dyn_obst.prediction, SetBasedPrediction):
                co = create_collision_object(dyn_obst.prediction)
                dyn_obstacles_list.append(co)
            else:
                raise Exception('Unknown dynamic obstacle prediction type: ' + str(type(dyn_obst.prediction)))
    return dyn_obstacles_list


def create_tvobstacle(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state[2], state[0], state[1])
        )
    return tv_obstacle


def create_coll_object(trajectory, vehicle_params, ego_state):
    """Create a collision_object of the trajectory for collision checking with road
    boundary and with other vehicles."""
    collision_object_raw2 = create_collision_object(trajectory)
    #TODO Unterschied zu oberen?
    collision_object_raw = create_tvobstacle_trajectory(
        traj_list=trajectory,
        box_length=vehicle_params.length / 2,
        box_width=vehicle_params.width / 2,
        start_time_step=ego_state.time_step,
    )
    # if the preprocessing fails, use the raw trajectory
    collision_object, err = trajectory_preprocess_obb_sum(
        collision_object_raw
    )
    if err:
        collision_object = collision_object_raw

    return collision_object


def create_tvobstacle_trajectory(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    traj_list = traj_list.prediction.trajectory.state_list
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state.orientation, state.position[0], state.position[1])
        )
    return tv_obstacle


def collision_check_prediction(
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

            if len(traj) <= 2:
                # commonroad_dc.collision.trajectory_queries.trajectory_queries.OBBSumException:
                # Invalid input trajectory_preprocess_obb_sum: Input trajectory must consists of at least two time steps
                continue

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
