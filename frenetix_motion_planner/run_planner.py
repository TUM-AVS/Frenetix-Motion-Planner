__author__ = "Rainer Trauth, Gerald Würsching"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# standard imports
import os
import time
from copy import deepcopy
import logging
from frenetix_motion_planner.utility.logging_helpers import messages_logger_initialization

# commonroad-io
from cr_scenario_handler.utils.collision_report import coll_report

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

# reactive planner
from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from frenetix_motion_planner.state import ReactivePlannerState
from frenetix_motion_planner.utility.visualization import visualize_planner_at_timestep, plot_final_trajectory, make_gif
from cr_scenario_handler.utils.evaluation import create_planning_problem_solution, reconstruct_inputs, plot_states, \
    plot_inputs, reconstruct_states, create_full_solution_trajectory, check_acceleration

from frenetix_motion_planner.utility import helper_functions as hf
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle

from cr_scenario_handler.utils.general import load_scenario_and_planning_problem

import frenetix_motion_planner.prediction_helpers as ph
from behavior_planner.behavior_module import BehaviorModule

from frenetix_motion_planner.occlusion_planning.occlusion_module import OcclusionModule

msg_logger = logging.getLogger("Message_logger")


def run_planner(config, log_path, mod_path, use_cpp):

    DT = config.planning.dt  # planning time step

    # *************************************
    # Message Logger of Run
    # *************************************
    messages_logger_initialization(config, log_path)
    msg_logger.critical("Start Scenario: " + log_path.split("/")[-1])

    # *************************************
    # Open CommonRoad scenario
    # *************************************
    scenario, planning_problem, planning_problem_set = load_scenario_and_planning_problem(config)

    # *************************************
    # Init and Goal State
    # *************************************
    problem_init_state = planning_problem.initial_state

    if not hasattr(problem_init_state, 'acceleration'):
        problem_init_state.acceleration = 0.
    x_0 = deepcopy(problem_init_state)

    # *************************************
    # Initialize Reactive Planner
    # *************************************

    planner = ReactivePlannerCpp(config, scenario, planning_problem, log_path, mod_path) if use_cpp else \
              ReactivePlannerPython(config, scenario, planning_problem, log_path, mod_path)

    # **************************
    # Run Variables
    # **************************
    shape = Rectangle(planner.vehicle_params.length, planner.vehicle_params.width)
    ego_vehicle = DynamicObstacle(42, ObstacleType.CAR, shape, x_0, None)
    planner.set_ego_vehicle_state(current_ego_vehicle=ego_vehicle)

    x_cl = None
    current_count = 0
    planning_times = list()

    behavior = None
    behavior_modul = None
    predictions = None
    reach_set = None
    visible_area = None
    occlusion_map = None
    occlusion_module = None
    desired_velocity = None

    # **************************
    # Convert Initial State
    # **************************
    x_0 = ReactivePlannerState.create_from_initial_state(x_0, config.vehicle.wheelbase, config.vehicle.wb_rear_axle)
    planner.record_state_and_input(x_0)

    # *************************************
    # Load Behavior Planner
    # *************************************
    if not config.behavior.use_behavior_planner:
        route_planner = RoutePlanner(scenario, planning_problem)
        reference_path = route_planner.plan_routes().retrieve_first_route().reference_path
        reference_path = hf.extend_rep_path(reference_path, x_0.position)
    else:
        behavior_modul = BehaviorModule(scenario=scenario,
                                        planning_problem=planning_problem,
                                        init_ego_state=x_0,
                                        dt=DT,
                                        config=config)
        reference_path = behavior_modul.reference_path

    # *****************************
    # Load Prediction and Reach Set
    # *****************************
    predictor = ph.load_prediction(scenario, config.prediction.mode, config)

    if 'responsibility' in config.cost.cost_weights and config.cost.cost_weights['responsibility'] > 0:
        reach_set = ph.load_reachset(scenario, config, mod_path)

    # **************************
    # Initialize Occlusion Module
    # **************************
    if config.occlusion.use_occlusion_module:
        occlusion_module = OcclusionModule(scenario, config, reference_path, log_path, planner)

    # **************************
    # Set External Planner Setups
    # **************************
    planner.update_externals(reference_path=reference_path, occlusion_module=occlusion_module, reach_set=reach_set)

    # **************************
    # Run Planner Cycle
    # **************************
    max_time_steps_scenario = int(config.general.max_steps*planning_problem.goal.state_list[0].time_step.end)
    while not planner.goal_status and current_count < max_time_steps_scenario:

        current_count = len(planner.record_state_list) - 1

        # *******************************
        # Cycle Prediction and Reach Sets
        # *******************************
        if config.prediction.mode:
            predictions, visible_area = ph.step_prediction(scenario, predictor, config, planner.ego_vehicle_history[-1],
                                                           x_0.time_step, occlusion_module)
            if 'responsibility' in config.cost.cost_weights and config.cost.cost_weights['responsibility'] > 0:
                reach_set = ph.step_reach_set(reach_set, scenario, x_0, predictions)

        # **************************
        # Cycle Behavior Planner
        # **************************
        if not config.behavior.use_behavior_planner:
            # set desired velocity
            desired_velocity = hf.calculate_desired_velocity(scenario, planning_problem, x_0, DT, desired_velocity=desired_velocity)
        else:
            behavior = behavior_modul.execute(predictions=predictions, ego_state=x_0, time_step=current_count)
            desired_velocity = behavior_modul.desired_velocity
            reference_path = behavior_modul.reference_path
        if config.occlusion.use_occlusion_module:
            # Only for demonstration reasons
            desired_velocity = 10

        # **************************
        # Cycle Occlusion Module
        # **************************
        if config.occlusion.use_occlusion_module:
            occlusion_module.step(predictions=predictions, x_0=planner.x_0, x_cl=planner.x_cl)
            predictions = occlusion_module.predictions

        # **************************
        # Set Planner Subscriptions
        # **************************
        planner.update_externals(x_0=x_0, x_cl=x_cl, desired_velocity=desired_velocity, predictions=predictions,
                                 behavior=behavior, reach_set=reach_set)

        # **************************
        # Execute Planner
        # **************************
        comp_time_start = time.time()
        optimal = planner.plan()  # returns the planned (i.e., optimal) trajectory
        comp_time_end = time.time()

        # if the planner fails to find an optimal trajectory -> terminate
        if not optimal:
            msg_logger.critical("No Kinematic Feasible and Optimal Trajectory Available!")
            break

        # store planning times
        planning_times.append(comp_time_end - comp_time_start)
        msg_logger.info(f"***Total Planning Time: \t\t{planning_times[-1]:.5f} s")

        # record state and input
        planner.record_state_and_input(optimal[0].state_list[1])

        # update init state and curvilinear state
        x_0 = deepcopy(planner.record_state_list[-1])
        x_cl = (optimal[2][1], optimal[3][1])

        msg_logger.info(f"current time step: {current_count}")
        msg_logger.info(f"current velocity: {x_0.velocity}")
        msg_logger.info(f"current target velocity: {desired_velocity}")

        # **************************
        # Visualize Scenario
        # **************************
        if config.debug.show_plots or config.debug.save_plots:
            visualize_planner_at_timestep(scenario=scenario, planning_problem=planning_problem,
                                          ego=planner.ego_vehicle_history[-1], traj_set=planner.all_traj,
                                          optimal_traj=optimal[0], ref_path=planner.reference_path,
                                          timestep=current_count, config=config, predictions=planner.predictions,
                                          plot_window=config.debug.plot_window_dyn, log_path=log_path,
                                          visible_area=visible_area, occlusion_map=occlusion_map)

        # **************************
        # Check Collision
        # **************************
        crash = planner.check_collision(planner.ego_vehicle_history[-1])
        if crash:
            msg_logger.info("Collision Detected!")
            if config.debug.collision_report and current_count > 0:
                coll_report(planner.ego_vehicle_history, planner, scenario, planning_problem, current_count, config,
                            log_path)
            break

        # **************************
        # Check Goal Status
        # **************************
        planner.check_goal_reached()

    # ******************************************************************************
    # End of Cycle
    # ******************************************************************************

    msg_logger.info(planner.goal_message)
    if planner.full_goal_status:
        msg_logger.info("\n", planner.full_goal_status)
    if not planner.goal_status and current_count >= max_time_steps_scenario:
        msg_logger.info("Scenario Aborted! Maximum Time Step Reached!")

    if not planner.goal_message == "Scenario Successful!":
        with open(os.path.join(mod_path, "logs", "log_failures.csv"), 'a') as file:
            line = str(scenario.scenario_id) + "\n"
            file.write(line)

    # plot  final ego vehicle trajectory
    plot_final_trajectory(scenario, planning_problem, planner.record_state_list, config, log_path)

    # make gif
    if config.debug.gif:
        make_gif(config, scenario, range(0, current_count), log_path, duration=0.25)

    # **************************
    # Evaluate results
    # **************************
    if config.occlusion.use_occlusion_module and config.occlusion.evaluate_occ:
        # plot occlusion evaluation
        occlusion_module.occ_plot.final_evaluation_plot(crash)

    if config.debug.evaluation:
        from commonroad.common.solution import CommonRoadSolutionWriter
        from commonroad_dc.feasibility.solution_checker import valid_solution

        ego_solution_trajectory = create_full_solution_trajectory(config, planner.record_state_list)

        # plot full ego vehicle trajectory
        plot_final_trajectory(scenario, planning_problem, ego_solution_trajectory.state_list, config, log_path)

        # create CR solution
        solution = create_planning_problem_solution(config, ego_solution_trajectory, scenario, planning_problem)

        # check feasibility
        # reconstruct inputs (state transition optimizations)
        feasible, reconstructed_inputs = reconstruct_inputs(config, solution.planning_problem_solutions[0])

        plot_inputs(config, planner.record_input_list, log_path, reconstructed_inputs, plot_bounds=True)

        # Write Solution to XML File for later evaluation
        solutionwriter = CommonRoadSolutionWriter(solution)
        solutionwriter.write_to_file(log_path, "solution.xml", True)

        msg_logger.info(valid_solution(scenario, planning_problem_set, solution))
