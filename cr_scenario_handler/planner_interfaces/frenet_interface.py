__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import traceback
from copy import deepcopy
from typing import List
import numpy as np

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory

from commonroad_dc import pycrcc

from cr_scenario_handler.utils.configuration import Configuration

from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from frenetix_motion_planner.state import ReactivePlannerState
from frenetix_motion_planner.utility import helper_functions as hf

from commonroad_route_planner.route_planner import RoutePlanner

from behavior_planner.behavior_module import BehaviorModule

from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface
from cr_scenario_handler.utils.collision_report import coll_report


class FrenetPlannerInterface(PlannerInterface):

    def __init__(self, config: Configuration, scenario: Scenario,
                 planning_problem: PlanningProblem, log_path: str, mod_path: str):
        """ Class for using the frenetix_motion_planner Frenet planner with the cr_scenario_handler.

        Implements the PlannerInterface.

        :param config: The configuration object.
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param planning_problem: The planning problem of the ego vehicle.
        :param log_path: Path for writing planner-specific log files to.
        :param mod_path: Working directory of the planner.
        """
        self.config = config
        self.scenario = scenario
        self.predictions = None
        self.planning_problem = planning_problem
        self.log_path = log_path
        self.mod_path = mod_path

        # Initialize planner
        self.planner = ReactivePlannerCpp(config, scenario, planning_problem,
                                       log_path, mod_path)

        # Set initial state and curvilinear state
        self.x_0 = ReactivePlannerState.create_from_initial_state(
            deepcopy(planning_problem.initial_state),
            config.vehicle.wheelbase,
            config.vehicle.wb_rear_axle
        )
        self.x_cl = None

        self.planner.set_x_0(self.x_0)

        self.desired_velocity = self.x_0.velocity

        # Set reference path
        self.use_behavior_planner = False
        if not self.use_behavior_planner:
            route_planner = RoutePlanner(scenario, planning_problem)
            self.ref_path = route_planner.plan_routes().retrieve_first_route().reference_path
        else:
            # Load behavior planner
            self.behavior_module = BehaviorModule(scenario=self.scenario, planning_problem=planning_problem,
                                                  init_ego_state=self.x_0, dt=self.scenario.dt, config=config)
            self.ref_path = self.behavior_module.reference_path

        self.planner.set_reference_path(self.ref_path)

        # Set planning problem
        goal_area = hf.get_goal_area_shape_group(
            planning_problem=planning_problem, scenario=scenario
        )
        self.planner.set_goal_area(goal_area)
        self.planner.set_planning_problem(planning_problem)

    def get_all_traj(self):
        """Return the sampled trajectory bundle for plotting purposes."""
        return self.planner.all_traj

    def get_ref_path(self):
        """Return the reference path for plotting purposes."""
        return self.planner.reference_path

    def check_collision(self, ego_vehicle_list: List[DynamicObstacle], timestep: int):
        """ Check for collisions with the ego vehicle.

        Adapted from ReactivePlanner.check_collision to allow using ego obstacles
        that contain the complete (past and future) trajectory.

        :param ego_vehicle_list: List containing the ego obstacles from at least
            the last two time steps.
        :param timestep: Time step to check for collisions at.

        :return: True iff there was a collision.
        """

        ego_vehicle = ego_vehicle_list[-1]

        ego = pycrcc.TimeVariantCollisionObject((timestep + 1))
        ego.append_obstacle(
            pycrcc.RectOBB(0.5 * self.planner.vehicle_params.length, 0.5 * self.planner.vehicle_params.width,
                           ego_vehicle.state_at_time(timestep).orientation,
                           ego_vehicle.state_at_time(timestep).position[0],
                           ego_vehicle.state_at_time(timestep).position[1]))

        if not self.planner.collision_checker.collide(ego):
            return False
        else:
            try:
                goal_position = []

                if self.planner.goal_checker.goal.state_list[0].has_value("position"):
                    for x in self.planner.reference_path:
                        if self.planner.goal_checker.goal.state_list[0].position.contains_point(x):
                            goal_position.append(x)
                    s_goal_1, d_goal_1 = self.planner._co.convert_to_curvilinear_coords(
                        goal_position[0][0],
                        goal_position[0][1])
                    s_goal_2, d_goal_2 = self.planner._co.convert_to_curvilinear_coords(
                        goal_position[-1][0],
                        goal_position[-1][1])
                    s_goal = min(s_goal_1, s_goal_2)
                    s_start, d_start = self.planner._co.convert_to_curvilinear_coords(
                        self.planner.planning_problem.initial_state.position[0],
                        self.planner.planning_problem.initial_state.position[1])
                    s_current, d_current = self.planner._co.convert_to_curvilinear_coords(
                        ego_vehicle.state_at_time(timestep).position[0],
                        ego_vehicle.state_at_time(timestep).position[1])
                    progress = ((s_current - s_start) / (s_goal - s_start))
                elif "time_step" in self.planner.goal_checker.goal.state_list[0].attributes:
                    progress = (timestep - 1 / self.planner.goal_checker.goal.state_list[0].time_step.end)
                else:
                    print('Could not calculate progress')
                    progress = None
            except:
                progress = None
                print('Could not calculate progress')
                traceback.print_exc()

            collision_obj = self.planner.collision_checker.find_all_colliding_objects(ego)[0]
            if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
                obj = collision_obj.obstacle_at_time(timestep)
                center = obj.center()
                last_center = collision_obj.obstacle_at_time(timestep - 1).center()
                r_x = obj.r_x()
                r_y = obj.r_y()
                orientation = obj.orientation()
                self.planner.logger.log_collision(True, self.planner.vehicle_params.length,
                                                  self.planner.vehicle_params.width,
                                                  progress, center,
                                                  last_center, r_x, r_y, orientation)
            else:
                self.planner.logger.log_collision(False, self.planner.vehicle_params.length,
                                                  self.planner.vehicle_params.width,
                                                  progress)

            if self.config.debug.collision_report:
                coll_report(ego_vehicle_list, self.planner, self.scenario,
                            self.planning_problem, timestep, self.config, self.log_path)

            return True

    def update_planner(self, scenario: Scenario, predictions: dict):
        """ Update the planner before the next time step.

        Updates the scenario and the internal states, and sets the new predictions.

        :param scenario: Updated scenario reflecting the new positions of other agents.
        :param predictions: Predictions for the other obstacles in the next time steps.
        """
        self.scenario = scenario
        self.predictions = predictions
        self.planner.update_externals(x_0=self.x_0, x_cl=self.x_cl, scenario=scenario, predictions=predictions)

        if not self.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = hf.calculate_desired_velocity(self.scenario, self.planning_problem,
                                                                  self.x_0, self.config.planning.dt,
                                                                  self.desired_velocity)
            self.planner.set_desired_velocity(self.desired_velocity, self.x_0.velocity)
        else:
            """-----------------------------------------Testing:---------------------------------------------"""
            # Currently not working.
            self.behavior_module.execute(predictions=self.predictions, ego_state=self.x_0,
                                         time_step=self.x_0.time_step)

            # set desired behavior outputs
            self.planner.set_desired_velocity(self.behavior_module.desired_velocity, self.x_0.velocity)
            self.planner.set_reference_path(self.behavior_module.reference_path)

            """--------------------------------------- End Testing ------------------------------------------"""

    def plan(self):
        """ Execute one planing step.

        update_planner has to be called before this function.
        Plans the trajectory for the next time step, updates the
        internal state of the FrenetInterface, and shifts the trajectory
        to the global representation.

        :return: error, trajectory
            where error is:
                0: If an optimal trajectory has been found.
                1: Otherwise.
            and trajectory is:
                A Trajectory object containing the planned trajectory,
                    using the vehicle center for the position: If error == 0
                None: Otherwise
        """
        # plan trajectory
        optimal = self.planner.plan()

        if not optimal:
            # Could not plan feasible trajectory
            return 1, None

        # record the new state for planner-internal logging
        self.planner.record_state_and_input(optimal[0].state_list[1])

        # update init state and curvilinear state
        self.x_0 = deepcopy(self.planner.record_state_list[-1])
        self.x_cl = (optimal[2][1], optimal[3][1])

        # Shift the state list to the center of the vehicle
        shifted_state_list = []
        for x in optimal[0].state_list:
            shifted_state_list.append(x.shift_positions_to_center(self.config.vehicle.wb_rear_axle))

        shifted_trajectory = Trajectory(shifted_state_list[0].time_step,
                                        shifted_state_list)
        return 0, shifted_trajectory

