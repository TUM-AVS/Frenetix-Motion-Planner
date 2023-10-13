__author__ = "Maximilian Geisslinger, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""Risk cost function and principles of ethics of risk."""
import numpy as np

from risk_assessment.harm_estimation import get_harm
from risk_assessment.collision_probability import (
    get_collision_probability_fast,
    get_inv_mahalanobis_dist
)
from risk_assessment.helpers.timers import ExecTimer
from frenetix_motion_planner.utility.responsibility import assign_responsibility_by_action_space, calc_responsibility_reach_set


def calc_risk(
    traj,
    ego_state,
    predictions: dict,
    scenario,
    ego_id: int,
    vehicle_params,
    params_harm,
    params_risk,
    road_boundary=None,
    reach_set=None,
    exec_timer=None,
):
    """
    Calculate the risk for the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        scenario (Scenario): Considered scenario.
        ego_id (int): ID of the ego vehicle.
        vehicle_params (VehicleParameters): Vehicle parameters of the
            considered vehicle.
        weights (Dict): Weighing factors. Read from weights.json.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json.
        exec_timer (ExecTimer): Timer for the exec_timing.json.

    Returns:
        float: Weighed risk costs.
        dict: Dictionary with ego harms for every time step concerning every
            obstacle
        dict: Dictionary with obstacle harms for every time step concerning
            every obstacle

    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    modes = params_risk
    coeffs = params_harm

    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate risk/"
        + "collision probability"
    ):

        if modes["fast_prob_mahalanobis"]:
            coll_prob_dict = get_inv_mahalanobis_dist(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
            )

        else:
            coll_prob_dict = get_collision_probability_fast(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
            )

    ego_harm_traj, obst_harm_traj = get_harm(
        scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer
    )

    # Calculate risk out of harm and collision probability
    ego_risk_traj = {}
    obst_risk_traj = {}
    ego_risk_max = {}
    obst_risk_max = {}
    ego_harm_max = {}
    obst_harm_max = {}

    # Pedestrian max harm at position of possible collision
    obst_harm_occ = {}

    for key in ego_harm_traj:
        ego_risk_traj[key] = [
            ego_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(ego_harm_traj[key]))
        ]
        obst_risk_traj[key] = [
            obst_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(obst_harm_traj[key]))
        ]
        if np.max(coll_prob_dict[key]) > 0.001:
            obst_harm_occ[key] = obst_harm_traj[key][np.argmax(coll_prob_dict[key])]
        else:
            obst_harm_occ[key] = 0.0
        # Take max as representative for the whole trajectory
        ego_risk_max[key] = max(ego_risk_traj[key])
        obst_risk_max[key] = max(obst_risk_traj[key])
        ego_harm_max[key] = max(ego_harm_traj[key])
        obst_harm_max[key] = max(obst_harm_traj[key])
    try:
        ego_risk = max(list(ego_risk_max.values()))
        obst_risk = max(list(obst_risk_max.values()))
        obst_harm_occ = max(list(obst_harm_occ.values()))
    except:
        ego_risk = 0
        obst_risk = 0
        obst_harm_occ = 0
    return ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, ego_risk, obst_risk, obst_harm_occ


def get_bayesian_costs(ego_risk_max, obst_risk_max, boundary_harm):
    """
    Bayesian Principle.

    Calculate the risk cost via the Bayesian Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        Dict: Risk costs for the considered trajectory according to the
            Bayesian Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return (sum(ego_risk_max.values()) + sum(obst_risk_max.values()) + boundary_harm) / (
        len(ego_risk_max) * 2
    )


def get_equality_costs(ego_risk_max, obst_risk_max):
    """
    Equality Principle.

    Calculate the risk cost via the Equality Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        equality_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Equality Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(
        [abs(ego_risk_max[key] - obst_risk_max[key]) for key in ego_risk_max]
    ) / len(ego_risk_max)


def get_maximin_costs(ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm, eps=10e-10, scale_factor=10):
    """
    Maximin Principle.

    Calculate the risk cost via the Maximin principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        maximin_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Maximum Principle
    """
    if len(ego_harm_max) == 0:
        return 0

    # Set maximin to 0 if probability (or risk) is 0
    maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
    maximin_obst = [a * int(bool(b < eps)) for a, b in zip(obst_harm_max.values(), obst_risk_max.values())]

    return max(maximin_ego + maximin_obst + [boundary_harm])**scale_factor


def get_ego_costs(ego_risk_max, boundary_harm):
    """
    Calculate the utilitarian ego cost for the given trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.

    Returns:
        Dict: Utilitarian ego risk cost
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(ego_risk_max.values()) + boundary_harm


def get_responsibility_cost(scenario, traj, ego_state, obst_risk_max, predictions, reach_set, mode="reach_set"):
    """Get responsibility cost.

    Args:
        obst_risk_max (_type_): _description_
        predictions (_type_): _description_
        mode (str) : "reach set" for reachable set mode, else assignement by space of action

    Returns:
        _type_: _description_

    """
    bool_contain_cache = None
    if "reach_set" in mode and reach_set is not None:
        resp_cost, bool_contain_cache = calc_responsibility_reach_set(traj, ego_state, reach_set, obst_risk_max)

    else:
        # Assign responsibility to predictions
        predictions = assign_responsibility_by_action_space(
            scenario, ego_state, predictions
        )
        resp_cost = 0

        for key in predictions:
            resp_cost -= predictions[key]["responsibility"] * obst_risk_max[key]

    return resp_cost, bool_contain_cache
