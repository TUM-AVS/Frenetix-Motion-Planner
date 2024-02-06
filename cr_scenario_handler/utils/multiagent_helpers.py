__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import warnings

from typing import List

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc import pycrcc

from cr_scenario_handler.utils.configuration import VehicleConfiguration


def scenario_without_obstacle_id(scenario: Scenario, obs_ids: List[int]):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not contained in the scenario")
        for obs_id in obs_ids:
            obs = scenario.obstacle_by_id(obs_id)
            if obs is not None:
                scenario.remove_obstacle(obs)
    return scenario


def get_all_obstacle_ids(scenario: Scenario) -> list:
    """Return all obstacle IDs that exist in a Commonroad Scenario

    :param scenario: The scenario of Commonroad we use
    """
    return [obs.obstacle_id for obs in scenario.obstacles]


def trajectory_to_obstacle(state_list: List[State],
                           vehicle_params: VehicleConfiguration, obstacle_id: int):
    """ Convert a state to a dummy DynamicObstacle object.

    :param state_list: List of all states in the trajectory.
    :param vehicle_params: VehicleConfiguration containing the shape of the obstacle.
    :param obstacle_id: ID to be assigned, usually equal to the agent ID.

    :return: A DynamicObstacle with the given state list as trajectory.
    """

    trajectory = Trajectory(initial_time_step=state_list[0].time_step, state_list=state_list)
    shape = Rectangle(vehicle_params.length, vehicle_params.width)
    prediction = TrajectoryPrediction(trajectory, shape)

    return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, trajectory.state_list[0], prediction)


def create_tvobstacle(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.
    Clone of frenetix_motion_planner/utils/helper_functions/create_tvobstacle().

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
