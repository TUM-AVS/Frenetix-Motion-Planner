__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from copy import deepcopy
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad_route_planner.route_planner import RoutePlanner

import cr_scenario_handler.utils.multiagent_logging as lh
from behavior_planner.behavior_module import BehaviorModule
from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface
import cr_scenario_handler.utils.goalcheck as gc
from cr_scenario_handler.utils.utils_coordinate_system import smooth_ref_path, extend_ref_path_both_ends
from cr_scenario_handler.utils.velocity_planner import VelocityPlanner

from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from frenetix_motion_planner.state import ReactivePlannerState

# from frenetix_occlusion.interface import FOInterface

# msg_logger = logging.getLogger("Message_logger")


class FrenetPlannerInterface(PlannerInterface):

    def __init__(self, agent_id: int, config_planner, config_sim, scenario: Scenario,
                 planning_problem: PlanningProblem, log_path: str, mod_path: str):
        """ Class for using the frenetix_motion_planner Frenet planner with the cr_scenario_handler.

        Implements the PlannerInterface.

        :param config_planner: The configuration object for the trajectory planner.
        :param config_sim: The configuration object for the simulation framework
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param planning_problem: The planning problem of the ego vehicle.
        :param log_path: Path for writing planner-specific log files to.
        :param mod_path: Working directory of the planner.
        """
        self.config_plan = config_planner
        self.config_sim = config_sim
        self.scenario = scenario
        self.id = agent_id
        self.config_sim.simulation.ego_agent_id = agent_id
        self.DT = self.config_plan.planning.dt
        self.replanning_counter = 0
        self.replanning_traj = None
        self.behavior_module_state = None

        self.planning_problem = planning_problem
        self.log_path = log_path
        self.mod_path = mod_path

        # *************************************
        # Message Logger of Run
        # *************************************
        # self.msg_logger = logging.getLogger("Message_logger_" +str(self.id ))
        self.msg_logger = lh.logger_initialization(self.config_plan, log_path, "Message_logger_" + str(self.id))
        self.msg_logger.critical("Start Planner Vehicle ID: " + str(self.id))
        # Init and Goal State

        # Initialize planner
        self.planner = ReactivePlannerCpp(self.config_plan, self.config_sim, scenario, planning_problem, log_path, mod_path, self.msg_logger) \
            if self.config_plan.debug.use_cpp else \
              ReactivePlannerPython(self.config_plan, self.config_sim, scenario, planning_problem, log_path, mod_path, self.msg_logger)

        problem_init_state = planning_problem.initial_state

        if not hasattr(problem_init_state, 'acceleration'):
            problem_init_state.acceleration = 0.
        x_0 = deepcopy(problem_init_state)

        shape = Rectangle(self.planner.vehicle_params.length, self.planner.vehicle_params.width)
        ego_vehicle = DynamicObstacle(agent_id, ObstacleType.CAR, shape, x_0, None)
        self.planner.set_ego_vehicle_state(current_ego_vehicle=ego_vehicle)

        # Set initial state and curvilinear state
        self.x_0 = ReactivePlannerState.create_from_initial_state(
            deepcopy(planning_problem.initial_state),
            self.config_sim.vehicle.wheelbase,
            self.config_sim.vehicle.wb_rear_axle
        )
        self.planner.record_state_and_input(self.x_0)

        self.x_cl = None
        self.desired_velocity = None
        self.occlusion_module = None
        self.behavior_module = None
        self.route_planner = None

        # Set reference path
        if not self.config_sim.behavior.use_behavior_planner:
            self.route_planner = RoutePlanner(lanelet_network=scenario.lanelet_network,
                                              planning_problem=planning_problem,
                                              scenario=scenario,
                                              extended_search=False)
            shortest_route = self.route_planner.plan_routes().retrieve_shortest_route(retrieve_shortest=True)

            # # Init route extendor
            # route_extendor: RouteExtendor = RouteExtendor(shortest_route, extrapolation_length=50)
            #
            # # Extend reference path at start and end
            # route_extendor.extend_reference_path_at_start_and_end()
            reference_path = extend_ref_path_both_ends(shortest_route.reference_path)

            self.reference_path = smooth_ref_path(reference_path)

        else:
            config_sim.behavior.dt = config_planner.planning.dt
            config_sim.behavior.replanning_frequency = config_planner.planning.replanning_frequency
            self.behavior_module = BehaviorModule(scenario=scenario,
                                                  planning_problem=planning_problem,
                                                  init_ego_state=x_0,
                                                  ego_id=agent_id,
                                                  config=config_sim,
                                                  log_path=self.log_path)
            self.reference_path = self.behavior_module.reference_path

        self.goal_area = gc.get_goal_area_shape_group(planning_problem=planning_problem, scenario=scenario)

        # *******************************************************************************
        # Initialize Occlusion Module GO to: https://github.com/TUM-AVS/Frenetix-Occlusion
        # ********************************************************************************
        # if self.config_sim.occlusion.use_occlusion_module:
        #   self.occlusion_module = FOInterface(scenario, self.reference_path, self.config_sim.vehicle, self.DT,
        #                            os.path.join(self.mod_path, "configurations", "simulation", "occlusion.yaml"))

        # **************************
        # Set External Planner Setups
        # **************************
        self.planner.update_externals(x_0=self.x_0, reference_path=self.reference_path, goal_area=self.goal_area,
                                      occlusion_module=self.occlusion_module)
        self.x_cl = self.planner.x_cl

        # ***************************
        # Initialize Velocity Planner
        # ***************************
        self.velocity_planner = VelocityPlanner(scenario=scenario, planning_problem=planning_problem,
                                                coordinate_system=self.coordinate_system)

    @property
    def all_trajectories(self):
        """Return the sampled trajectory bundle for plotting purposes."""
        return self.planner.all_traj

    @property
    def record_input_list(self):
        return self.planner.record_input_list

    @property
    def record_state_list(self):
        return self.planner.record_state_list

    @property
    def vehicle_history(self):
        return self.planner.ego_vehicle_history

    @property
    def coordinate_system(self):
        return self.planner.coordinate_system

    @property
    def optimal_trajectory(self):
        return self.planner.optimal_trajectory

    @property
    def trajectory_pair(self):
        return self.planner.trajectory_pair

    def update_planner(self, scenario: Scenario, predictions: dict):
        """ Update the planner before the next time step.

        Updates the scenario and the internal states, and sets the new predictions.

        :param scenario: Updated scenario reflecting the new positions of other agents.
        :param predictions: Predictions for the other obstacles in the next time steps.
        """
        self.scenario = scenario

        if not self.config_sim.behavior.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = self.velocity_planner.calculate_desired_velocity(self.x_0, self.x_cl[0][0])
        else:
            # raise NotImplementedError
            behavior = self.behavior_module.execute(
                predictions=predictions,
                ego_state=self.x_0,
                time_step=self.replanning_counter)
            self.desired_velocity = behavior.desired_velocity
            if behavior.reference_path is not None:
                self.reference_path = behavior.reference_path

            self.planner.update_externals(behavior=behavior)

            # self.stop_point_s = behavior.stop_point_s
            # self.desired_velocity_stop_point = behavior.desired_velocity_stop_point
            self.behavior_module_state = behavior.behavior_planner_state
        # End TODO

        self.planner.update_externals(scenario=scenario, x_0=self.x_0, x_cl=self.x_cl,
                                      desired_velocity=self.desired_velocity, predictions=predictions)

    def step_interface(self, current_timestep=None):
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

        if int(self.replanning_counter / self.config_plan.planning.replanning_frequency) == 1:
            self.replanning_counter = 0

        if self.replanning_counter == 0 or self.config_plan.planning.replanning_frequency < 2:
            if self.occlusion_module is not None:
                self.occlusion_module.evaluate_scenario(predictions=self.planner.predictions,
                                                        ego_pos=self.x_0.position,
                                                        ego_v=self.x_0.velocity,
                                                        ego_orientation=self.x_0.orientation,
                                                        ego_pos_cl=np.array([self.x_cl[0][0], self.x_cl[1][0]]),
                                                        timestep=current_timestep,
                                                        cosy_cl=self.coordinate_system.ccosy)

            # plan trajectory
            optimal_trajectory_pair = self.planner.plan()

            if not optimal_trajectory_pair:
                # Could not plan feasible trajectory
                self.msg_logger.critical("No Kinematic Feasible and Optimal Trajectory Available!")
                return None, self.replanning_counter

            # record the new state for planner-internal logging
            self.planner.record_state_and_input(optimal_trajectory_pair[0].state_list[1])

            # update init state and curvilinear state
            self.x_0 = deepcopy(self.planner.record_state_list[-1])
            self.x_cl = (optimal_trajectory_pair[2][1], optimal_trajectory_pair[3][1])

            self.msg_logger.info(f"current time step: {current_timestep}")
            self.msg_logger.info(f"current velocity: {self.x_0.velocity}")
            self.msg_logger.info(f"current target velocity: {self.desired_velocity}")

            self.replanning_traj = optimal_trajectory_pair
            selected_trajectory = optimal_trajectory_pair[0]

        else:

            # record the new state for planner-internal logging
            self.planner.record_state_and_input(self.replanning_traj[0].state_list[1+self.replanning_counter])

            # update init state and curvilinear state
            self.x_0 = deepcopy(self.planner.record_state_list[-1])
            self.x_cl = (self.replanning_traj[2][1+self.replanning_counter], self.replanning_traj[3][1+self.replanning_counter])

            current_ego_vehicle = self.planner.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list[self.replanning_counter:],
                                                                                       self.config_sim.simulation.ego_agent_id)
            self.planner.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

            self.planner.plan_postprocessing(optimal_trajectory=self.planner.optimal_trajectory, planning_time=0.0,
                                             replanning_counter=self.replanning_counter)

            self.msg_logger.info(f"current time step: {current_timestep}")
            self.msg_logger.info(f"current velocity: {self.x_0.velocity}")
            self.msg_logger.info(f"current target velocity: {self.desired_velocity}")
            selected_trajectory = self.replanning_traj

        self.replanning_counter += 1

        return selected_trajectory, self.replanning_counter-1
