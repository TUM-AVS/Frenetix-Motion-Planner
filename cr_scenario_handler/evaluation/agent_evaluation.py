__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


import os
import traceback
from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InputState, TraceState, State
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.common.solution import Solution, PlanningProblemSolution, VehicleModel, \
    VehicleType, CostFunction, CommonRoadSolutionWriter

from commonroad_dc.feasibility.feasibility_checker import VehicleDynamics, \
    state_transition_feasibility, position_orientation_objective, \
    position_orientation_feasibility_criteria, _angle_diff

from commonroad_dc.feasibility.solution_checker import valid_solution, CollisionException, \
    GoalNotReachedException, MissingSolutionException

from cr_scenario_handler.utils.configuration import Configuration

def create_full_solution_trajectory(config: Configuration, state_list: List[State]) -> Trajectory:
    """
    Create CR solution trajectory from recorded state list of the reactive planner
    Positions are shifted from rear axis to vehicle center due to CR position convention
    """
    new_state_list = list()
    for state in state_list:
        new_state_list.append(
            state.translate_rotate(np.array([config.vehicle.wb_rear_axle * np.cos(state.orientation),
                                             config.vehicle.wb_rear_axle * np.sin(state.orientation)]), 0.0))
    return Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)


def create_planning_problem_solution(config: Configuration, solution_trajectory: Trajectory, scenario: Scenario,
                                     planning_problem: PlanningProblem) -> Solution:
    """
    Creates CommonRoad Solution object
    """
    pps = PlanningProblemSolution(planning_problem_id=planning_problem.planning_problem_id,
                                  vehicle_type=VehicleType(config.vehicle.cr_vehicle_id),
                                  vehicle_model=VehicleModel.KS,
                                  cost_function=CostFunction.WX1,
                                  trajectory=solution_trajectory)

    # create solution object
    solution = Solution(scenario.scenario_id, [pps])
    return solution


def reconstruct_states(config: Configuration, scenario, states: List[Union[State, TraceState]], inputs: List[InputState]):
    """
    reconstructs states from a given list of inputs by forward simulation
    """
    vehicle_dynamics = VehicleDynamics.from_model(VehicleModel.KS, VehicleType(config.vehicle.cr_vehicle_id))

    x_sim_list = list()
    x_sim_list.append(states[0])
    for idx, inp in enumerate(inputs):
        x0, x0_ts = vehicle_dynamics.state_to_array(states[idx])
        u0 = vehicle_dynamics.input_to_array(inp)[0]
        x1_sim = vehicle_dynamics.forward_simulation(x0, u0, scenario.dt, throw=False)
        x_sim_list.append(vehicle_dynamics.array_to_state(x1_sim, x0_ts+1))
    return x_sim_list


def reconstruct_inputs(config: Configuration, scenario, pps: PlanningProblemSolution):
    """
    reconstructs inputs for each state transition using the feasibility checker
    """
    vehicle_dynamics = VehicleDynamics.from_model(pps.vehicle_model, pps.vehicle_type)

    feasible_state_list = []
    reconstructed_inputs = []
    for x0, x1 in zip(pps.trajectory.state_list[:-1], pps.trajectory.state_list[1:]):
        # reconstruct inputs using optimization
        feasible_state, reconstructed_input_state = state_transition_feasibility(x0, x1, vehicle_dynamics,
                                                                                 scenario.dt,
                                                                                 position_orientation_objective,
                                                                                 position_orientation_feasibility_criteria,
                                                                                 1e-8, np.array([2e-2, 2e-2, 3e-2]),
                                                                                 4, 100, False)
        feasible_state_list.append(feasible_state)
        reconstructed_inputs.append(reconstructed_input_state)
    return feasible_state_list, reconstructed_inputs


def check_acceleration(config: Configuration, state_list:  List[Union[State, TraceState]], scenario, plot=False):
    """
    Checks whether the computed acceleration the trajectory matches the velocity difference (dv/dt), i.e., assuming
    piecewise constant acceleration input
    """
    # computed acceleration of trajectory
    a_planned = np.array([state.acceleration for state in state_list])
    a_piecewise_constant = np.array([(a_planned[i] + a_planned[i+1]) / 2 for i in range(len(a_planned)-1)])
    # recalculated acceleration via velocity
    v = np.array([state.velocity for state in state_list])
    a_recalculated = np.diff(v) / scenario.dt
    # check
    diff = np.abs(a_piecewise_constant-a_recalculated)
    acc_correct = np.all(diff < 1e-01)
    print("Acceleration correct: %s, with max deviation %s" % (acc_correct, np.max(diff)))

    if plot:
        plt.figure(figsize=(7, 3.5))
        plt.suptitle("Acceleration check")
        plt.plot(list(range(len(a_planned[1:]))),
                 a_planned[1:], color="black", label="planned acceleration")
        plt.plot(list(range(len(a_piecewise_constant))),
                 a_piecewise_constant, color="green", label="planned (piecewise constant)")
        plt.plot(list(range(len(a_recalculated))),
                 a_recalculated, color="orange", label="recomputed (dv/dt)")
        plt.xlabel("t in s")
        plt.ylabel("a_long in m/s^2")
        plt.legend()
        plt.tight_layout()
        # plt.show()


def plot_states(config: Configuration, state_list: List[Union[State, TraceState]], scenario, save_path: str, reconstructed_states=None, plot_bounds=False):
    """
    Plots states of trajectory from a given state_list
    state_list must contain the following states: steering_angle, velocity, orientation and yaw_rate
    """
    plt.figure(figsize=(7, 8.0))
    plt.suptitle("States")

    # x, y position
    plt.subplot(5, 1, 1)
    plt.plot([state.position[0] for state in state_list],
             [state.position[1] for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot([state.position[0] for state in reconstructed_states],
                 [state.position[1] for state in reconstructed_states], color="blue", label="reconstructed")
    plt.xlabel("x")
    plt.ylabel("y")

    # steering angle
    plt.subplot(5, 1, 2)
    plt.plot(list(range(len(state_list))),
             [state.steering_angle for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.steering_angle for state in reconstructed_states], color="blue", label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(state_list)], [config.vehicle.delta_min, config.vehicle.delta_min],
                 color="red", label="bounds")
        plt.plot([0, len(state_list)], [config.vehicle.delta_max, config.vehicle.delta_max],
                 color="red")
    plt.ylabel("delta")

    # velocity
    plt.subplot(5, 1, 3)
    plt.plot(list(range(len(state_list))),
             [state.velocity for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.velocity for state in reconstructed_states], color="blue", label="reconstructed")
    plt.legend()
    plt.ylabel("v")

    # orientation
    plt.subplot(5, 1, 4)
    plt.plot(list(range(len(state_list))),
             [state.orientation for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.orientation for state in reconstructed_states], color="blue", label="reconstructed")
    plt.ylabel("theta")

    # yaw rate
    plt.subplot(5, 1, 5)
    plt.plot(list(range(len(state_list))),
             [state.yaw_rate for state in state_list], color="black", label="planned")
    reconstructed_yaw_rate = np.diff(np.array([state.orientation for state in reconstructed_states])) / scenario.dt
    reconstructed_yaw_rate = np.insert(reconstructed_yaw_rate, 0, state_list[0].yaw_rate, axis=0)
    plt.plot(list(range(len(state_list))),
             reconstructed_yaw_rate, color="blue", label="reconstructed")
    plt.ylabel("theta_dot")
    plt.tight_layout()

    # Save Output
    plot_path = os.path.join(save_path, "evaluation_plot_states")
    plt.savefig(f"{plot_path}.svg", format='svg')

    # plot errors in position, velocity, orientation
    if reconstructed_states:
        plt.figure(figsize=(7, 8.0))
        plt.suptitle("State Errors (planned vs. forward simulated)")

        plt.subplot(5, 1, 1)
        plt.plot(list(range(len(state_list))), [abs(state_list[i].position[0] - reconstructed_states[i].position[0])
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("x error")
        plt.subplot(5, 1, 2)
        plt.plot(list(range(len(state_list))), [abs(state_list[i].position[1] - reconstructed_states[i].position[1])
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("y error")
        plt.subplot(5, 1, 3)
        plt.plot(list(range(len(state_list))), [abs(_angle_diff(state_list[i].orientation,
                                                                reconstructed_states[i].orientation))
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("delta error")
        plt.subplot(5, 1, 4)
        plt.plot(list(range(len(state_list))), [abs(state_list[i].velocity - reconstructed_states[i].velocity)
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("velocity error")
        plt.subplot(5, 1, 5)
        plt.plot(list(range(len(state_list))), [abs(_angle_diff(state_list[i].steering_angle,
                                                                reconstructed_states[i].steering_angle))
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("theta error")
        plt.tight_layout()

        # Save Output
        plot_path = os.path.join(save_path, "evaluation_plot_states_reconstructed")
        plt.savefig(f"{plot_path}.svg", format='svg')


def plot_inputs(config: Configuration, input_list: List[InputState], save_path: str, reconstructed_inputs=None, plot_bounds=False):
    """
    Plots inputs of trajectory from a given input_list
    input_list must contain the following states: steering_angle_speed, acceleration
    optionally plots reconstructed_inputs
    """
    plt.figure()
    plt.suptitle("Inputs")

    # steering angle speed
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(input_list))),
             [state.steering_angle_speed for state in input_list], color="black", label="planned")
    if reconstructed_inputs:
        plt.plot(list(range(len(reconstructed_inputs))),
                 [state.steering_angle_speed for state in reconstructed_inputs], color="blue",
                 label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(input_list)], [config.vehicle.v_delta_min, config.vehicle.v_delta_min],
                 color="red", label="bounds")
        plt.plot([0, len(input_list)], [config.vehicle.v_delta_max, config.vehicle.v_delta_max],
                 color="red")
    plt.legend()
    plt.ylabel("v_delta in rad/s")

    # acceleration
    plt.subplot(2, 1, 2)
    plt.plot(np.array(list(range(len(input_list)))),
             [state.acceleration for state in input_list], color="black", label="planned")
    if reconstructed_inputs:
        plt.plot(list(range(len(reconstructed_inputs))),
                 [state.acceleration for state in reconstructed_inputs], color="blue", label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(input_list)], [-config.vehicle.a_max, -config.vehicle.a_max],
                 color="red", label="bounds")
        plt.plot([0, len(input_list)], [config.vehicle.a_max, config.vehicle.a_max],
                 color="red")
    plt.ylabel("a_long in m/s^2")
    plt.tight_layout()

    # Save Output
    plot_path = os.path.join(save_path, "evaluation_plot_inputs")
    plt.savefig(f"{plot_path}.svg", format='svg')

def shift_positions_to_center(state_list, config):
    """
    Shifts position from rear-axle to vehicle center
    :param wb_rear_axle: distance between rear-axle and vehicle center
    """
    shifted_states = []
    # shift positions from rear axle to center
    wb_rear_axle = config.vehicle.wb_rear_axle
    for state in state_list:
        orientation = state.orientation
        state_shifted = state.translate_rotate(np.array([wb_rear_axle * np.cos(orientation),
                                                    wb_rear_axle * np.sin(orientation)]), 0.0)
        shifted_states.append(state_shifted)
    return shifted_states
def evaluate(scenario: Scenario, planning_problem: PlanningProblem, id: int,
             recorded_state_list: List[State], recorded_input_list: List[InputState],
             config: Configuration, log_path: str):
    """ Wrapper function for evaluating a planned trajectory.

    Calls all other functions in this file on a list of recorded states.

    :param scenario: The scenario that has been solved.
    :param planning_problem: The planning problem of the ego vehicle.
    :param id: The obstacle ID of the ego vehicle.
    :param recorded_state_list: List of agent states the agent recorded during simulation.
    :param recorded_input_list: List of inputs for every planner step.
    :param config: The configuration.
    :param log_path: Path to write the evaluation results to.
    """

    # create full solution trajectory
    initial_timestep = planning_problem.initial_state.time_step
    recorded_states = shift_positions_to_center(recorded_state_list, config)
    ego_solution_trajectory = Trajectory(initial_time_step=initial_timestep,
                                         state_list=recorded_states[initial_timestep:])

    # plot full ego vehicle trajectory
    # plot_final_trajectory(scenario, planning_problem, ego_solution_trajectory.state_list,
    #                       config, log_path)

    # create CR solution
    solution = create_planning_problem_solution(config, ego_solution_trajectory,
                                                scenario, planning_problem)

    # check feasibility
    # reconstruct inputs (state transition optimizations)
    feasible, reconstructed_inputs = reconstruct_inputs(config, scenario, solution.planning_problem_solutions[0])
    try:
        # reconstruct states from inputs
        reconstructed_states = reconstruct_states(config, scenario, ego_solution_trajectory.state_list,
                                                  reconstructed_inputs)
        # check acceleration correctness
        check_acceleration(config, ego_solution_trajectory.state_list, scenario, plot=True)

        # remove first element from input list
        recorded_input_list.pop(0)

        # evaluate
        plot_states(config, ego_solution_trajectory.state_list, scenario, log_path, reconstructed_states,
                    plot_bounds=False)
        # CR validity check
        print(f"[Agent {id}] Feasibility Check Result: ")
        if valid_solution(scenario, PlanningProblemSet([planning_problem]), solution)[0]:
            print(f"[Agent {id}] Valid")
    except CollisionException:
        print(f"[Agent {id}] Infeasible: Collision")
    except GoalNotReachedException:
        print(f"[Agent {id}] Infeasible: Goal not reached")
    except MissingSolutionException:
        print(f"[Agent {id}] Infeasible: Missing solution")
    except:
        traceback.print_exc()
        print(f"[Agent {id}] Could not reconstruct states")

    plot_inputs(config, recorded_input_list, log_path, reconstructed_inputs, plot_bounds=True)

    # Write Solution to XML File for later evaluation
    solutionwriter = CommonRoadSolutionWriter(solution)
    solutionwriter.write_to_file(log_path, "solution.xml", True)
