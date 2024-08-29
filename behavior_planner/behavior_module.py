__author__ = "Moritz Ellermann, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# general imports
import copy
import os
import time

# commonroad imports

# project imports
import behavior_planner.utils.helper_functions as hf
from behavior_planner.utils.helper_logging import BehaviorLogger
from behavior_planner.utils.velocity_planner import VelocityPlanner
from behavior_planner.utils.path_planner import PathPlanner
from behavior_planner.utils.FSM_model import EgoFSM

# TODO: reorganize the State classes into Dataclasses


class BehaviorModule(object):
    """
    Behavior Module: Coordinates Path Planner, Velocity Planner and Finite State Machine (FSM) to determine the
    reference path and desired velocity for the reactive planner.

    TODO: Include FSM
    """

    def __init__(self, scenario, planning_problem, init_ego_state, ego_id, config, log_path):
        """ Init Behavior Module.

        Args:
        pro_path (str): project path.
        scenario: scenario.
        init_ego_state : initialized ego state.
        """
        # start timer
        timer = time.time()

        self.BM_state = BehaviorModuleState()  # behavior module information

        # load config
        self.BM_state.config = config
        self.behavior_config = config.behavior
        self.behavior_config.behavior_log_path_scenario = os.path.join(log_path, "behavior_logs")

        os.makedirs(self.behavior_config.behavior_log_path_scenario, exist_ok=True)

        # initialize loggers
        self.behavior_logger = BehaviorLogger(self.behavior_config)
        self.behavior_message_logger = self.behavior_logger.message_logger

        # init behavior planner and load scenario information
        self.VP_state = self.BM_state.VP_state  # velocity planner information
        self.PP_state = self.BM_state.PP_state  # path planner information
        self.FSM_state = self.BM_state.FSM_state  # FSM information
        self.BM_state.vehicle_params = config.vehicle
        self.BM_state.init_velocity = init_ego_state.velocity
        self.BM_state.dt = config.behavior.dt
        self.BM_state.ego_id = ego_id

        self.BM_state.scenario = scenario
        self.BM_state.planning_problem = planning_problem

        self.BM_state.country = hf.find_country_traffic_sign_id(self.BM_state.scenario)
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=[],
                ego_state=init_ego_state,
                country=self.BM_state.country)

        self.behavior_message_logger.debug("Scenario and Planning Problem initialized")

        # init path planner
        self.path_planner = PathPlanner(BM_state=self.BM_state)
        self.path_planner.execute_route_planning()
        self._retrieve_lane_changes_from_navigation()

        self.behavior_message_logger.debug("Path Planner initialized")

        # init ego FSM
        self.ego_FSM = EgoFSM(BM_state=self.BM_state)
        self.FSM_state = self.ego_FSM.FSM_state

        self.behavior_message_logger.debug("FSM initialized")

        # init velocity planner
        self.velocity_planner = VelocityPlanner(BM_state=self.BM_state)

        self.behavior_message_logger.debug("Velocity Planner initialized")

        # outputs
        self.behavior_output = BehaviorOutput(self.BM_state)
        self.reference_path = self.BM_state.PP_state.reference_path
        self.desired_velocity = None
        self.flags = {"stopping_for_traffic_light": None,
                      "waiting_for_green_light": None
                      }

        # end timer
        timer = time.time() - timer

        # logging
        self.behavior_message_logger.critical("Behavior Module initialized")
        self.behavior_message_logger.info(f"Behavior Initialization Time: \t\t{timer:.5f} s")
        self.behavior_message_logger.debug("simulating scenario: " + str(self.BM_state.scenario.scenario_id))

    def execute(self, predictions, ego_state, time_step):
        """ Execute behavior module.

        TODO: Dynamische Entscheidungen in jedem Time step, highlevel (y.B. Spurwechsel) nur alle 300 - 500 ms

        Args:
        predictions (dict): current predictions.
        ego_state (List): current state of ego vehicle.

        return: behavior_output (BehaviorOutput): Class holding all information for Reactive Planner
        """
        if int(ego_state.time_step % self.behavior_config.replanning_frequency) == 0:
            self.BM_state.plan_dynamics_only = False
        else:
            self.BM_state.plan_dynamics_only = True

        # start timer
        timer = time.time()

        # inputs
        self.BM_state.predictions = predictions
        self.BM_state.ego_state = ego_state
        self.BM_state.time_step = ego_state.time_step  # time_step

        self._get_ego_position(ego_state)

        self.BM_state.future_factor = int(self.BM_state.ego_state.velocity // 4) + 1  # for lane change maneuvers
        self._collect_necessary_information()

        # execute FSM
        self.ego_FSM.execute()

        # execute path planner
        if not self.BM_state.plan_dynamics_only:
            if self.FSM_state.do_lane_change:
                self.path_planner.execute_lane_change()
            if self.FSM_state.undo_lane_change:
                self.path_planner.undo_lane_change()
            self.reference_path = self.PP_state.reference_path

        # execute velocity planner
        self.velocity_planner.execute()
        self.desired_velocity = self.VP_state.desired_velocity

        # calculate stopping points
        self._calculate_stopping_point()

        # update behavior flags
        self.flags["stopping_for_traffic_light"] = self.FSM_state.slowing_car_for_traffic_light
        self.flags["waiting_for_green_light"] = self.FSM_state.waiting_for_green_light

        # update behavior output; input for reactive planner
        self.behavior_output.reference_path = self.reference_path
        self.behavior_output.desired_velocity = self.desired_velocity
        self.behavior_output.stop_point_s = self.BM_state.stop_point_s
        self.behavior_output.desired_velocity_stop_point = self.BM_state.desired_velocity_stop_point
        self.behavior_output.behavior_planner_state = self.BM_state.BP_state.set_values(self.BM_state)

        # end timer
        timer = time.time() - timer

        # logging
        self.behavior_message_logger.debug("VP velocity mode: " + str(self.VP_state.velocity_mode))
        self.behavior_message_logger.debug("VP TTC velocity: " + str(self.VP_state.TTC))
        self.behavior_message_logger.debug("VP MAX velocity: " + str(self.VP_state.MAX))
        if self.VP_state.closest_preceding_vehicle is not None:
            self.behavior_message_logger.debug("VP position of preceding vehicle: " + str(self.VP_state.pos_preceding_veh))
            self.behavior_message_logger.debug("VP velocity of preceding vehicle: " + str(self.VP_state.vel_preceding_veh))
            self.behavior_message_logger.debug("VP distance to preceding vehicle: " + str(self.VP_state.dist_preceding_veh))
            self.behavior_message_logger.debug("VP safety distance to preceding vehicle: " + str(self.VP_state.safety_dist))
        self.behavior_message_logger.debug("VP recommended velocity: " + str(self.VP_state.goal_velocity))
        self.behavior_message_logger.debug("BP recommended desired velocity: " + str(self.desired_velocity))
        self.behavior_message_logger.debug("current ego velocity: " + str(self.BM_state.ego_state.velocity))
        self.behavior_message_logger.info(f"Behavior Planning Time: \t\t{timer:.5f} s")

        self.behavior_logger.log_data(self.BM_state.__dict__)

        return copy.deepcopy(self.behavior_output)

    def _retrieve_lane_changes_from_navigation(self):
        self.BM_state.nav_lane_changes_left = 0
        self.BM_state.nav_lane_changes_right = 0
        lane_change_instructions = hf.retrieve_glb_nav_path_lane_changes(self.BM_state.scenario, self.BM_state.global_nav_route)
        for idx, instruction in enumerate(lane_change_instructions):
            if lane_change_instructions[idx] == 1:
                lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                    self.BM_state.global_nav_route.list_ids_lanelets[idx])
                if lanelet.adj_left == self.BM_state.global_nav_route.list_ids_lanelets[idx + 1]:
                    self.BM_state.nav_lane_changes_left += 1
                if lanelet.adj_right == self.BM_state.global_nav_route.list_ids_lanelets[idx + 1]:
                    self.BM_state.nav_lane_changes_right += 1

    def _get_ego_position(self, ego_state):
        try:
            self.BM_state.ref_position_s = self.PP_state.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                ego_state.position[0], ego_state.position[1])[0]
        except:
            self.behavior_message_logger.error("Ego position out of reference path coordinate system projection domain")

    def _collect_necessary_information(self):
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting_scenario = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=self.PP_state.reference_path_ids,
                ego_state=self.BM_state.ego_state,
                country=self.BM_state.country)

        self.BM_state.current_lanelet = \
            self.BM_state.scenario.lanelet_network.find_lanelet_by_id(self.BM_state.current_lanelet_id)

        self.VP_state.closest_preceding_vehicle, self.VP_state.pos_preceding_veh, self.VP_state.dist_preceding_veh, self.VP_state.vel_preceding_veh =\
            hf.get_closest_preceding_obstacle(predictions=self.BM_state.predictions,
                                              scenario=self.BM_state.scenario,
                                              coordinate_system=self.PP_state.cl_ref_coordinate_system,
                                              lanelet_id=self.BM_state.current_lanelet_id,
                                              ego_position_s=self.BM_state.ref_position_s,
                                              ego_state=self.BM_state.ego_state,
                                              ego_id=self.BM_state.ego_id)

    def _calculate_stopping_point(self):
        """
        Calculates the point up to which the reactive planner should plan
        The point consists of two parts:
        1. The position relative to the reference path (S-position)
        2. The desired velocity at that point
        \n
        If there is a static goal with a stopping point is scheduled, plan to the static goal s_stopping_position and
        aim to have no velocity there. If there is no stop planned choose the maximal distance the current_velocity *
        default_time_horizon, the comfortable_stopping_distance and eventually the stop_point_s of the current static
        goal.
        If a preceding vehicle exists plan to the point, where the preceding vehicle would come to a standstill in case
        of a breaking scenario and aim to have the same velocity as the preceding vehicle or come to a standstill behind
        the preceding vehicle.
        """
        comfort_stopping_point_s = self.BM_state.ref_position_s + self.VP_state.comfortable_stopping_distance
        stop_point_min_dist = max(self.behavior_config.min_stop_point_dist,
                                  self.behavior_config.min_stop_point_time * self.BM_state.ego_state.velocity)

        ####################################################################################################
        #                          Calculating Stop Point from Static Goal Route                           #
        ####################################################################################################
        # calculating the stopping position if the car continues driving with the same velocity for the set duration
        default_time_stopping_point_s = (self.BM_state.ref_position_s
                                         + self.BM_state.ego_state.velocity
                                         * self.behavior_config.default_time_horizon)
        if self.FSM_state.behavior_state_static in ["PrepareTrafficLight", "TrafficLight",
                                                    "PrepareCrosswalk", "Crosswalk",
                                                    "PrepareYieldSign", "YieldSign",
                                                    "PrepareStopSign", "StopSign"]:
            # Situation States of 'Prepare' behavior states
            if self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.startswith("Observing"):
                # ObservingTrafficLight, ObservingCrosswalk, ObservingStopYieldSign
                self.BM_state.stop_point_s = min(self.BM_state.current_static_goal.stop_point_s,
                                                 comfort_stopping_point_s)
                self.BM_state.desired_velocity_stop_point = self.VP_state.goal_velocity
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "SlowingDown":
                self.BM_state.stop_point_s = min(self.BM_state.current_static_goal.stop_point_s,
                                                 comfort_stopping_point_s)
                self.BM_state.desired_velocity_stop_point = 0.0
            # Situation States of behavior states
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "GreenLight":
                self.BM_state.stop_point_s = max(
                    self.BM_state.current_static_goal.stop_point_s,
                    comfort_stopping_point_s,
                    default_time_stopping_point_s
                )
                self.BM_state.desired_velocity_stop_point = self.VP_state.goal_velocity
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.endswith("Clear"):
                # CrosswalkClear, StopYieldSignClear, TurnClear
                self.BM_state.stop_point_s = max(
                    self.BM_state.current_static_goal.stop_point_s,
                    comfort_stopping_point_s,
                    default_time_stopping_point_s
                )
                self.BM_state.desired_velocity_stop_point = self.VP_state.goal_velocity
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "Stopping":
                self.BM_state.stop_point_s = min(self.BM_state.current_static_goal.stop_point_s,
                                                 comfort_stopping_point_s)
                self.BM_state.desired_velocity_stop_point = 0.0
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.startswith("Waiting"):
                # Don't move: WaitingForGreenLight, WaitingForCrosswalkClearance,
                # WaitingForStopYieldSignClearance, WaitingForTurnClearance
                self.BM_state.stop_point_s = self.BM_state.ref_position_s
                self.BM_state.desired_velocity_stop_point = 0.0
                self.BM_state.stop_point_dist = self.BM_state.stop_point_s - self.BM_state.ref_position_s
                self.BM_state.stop_point_mode = "s-pos: current position | vel: 0"
                return  # special case where minimal distance is not wanted
            elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "ContinueDriving":
                self.BM_state.stop_point_s = max(
                    comfort_stopping_point_s,
                    default_time_stopping_point_s
                )
                self.BM_state.desired_velocity_stop_point = self.VP_state.goal_velocity
            else:
                self.behavior_message_logger.warning(
                    f"'{self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state}'"
                    f" is not a valid Situation State for {self.FSM_state.behavior_state_static}")
        else:
            self.BM_state.stop_point_s = max(
                comfort_stopping_point_s,
                default_time_stopping_point_s
            )
            self.BM_state.desired_velocity_stop_point = self.VP_state.goal_velocity

        ####################################################################################################
        #                                 Calculating Stop Point from TTC                                  #
        ####################################################################################################
        ttc_stopping_stop_point_s = None
        if self.VP_state.TTC is not None:
            # calculate the stopping position so that is lies just safely behind the preceding vehicle
            ttc_stopping_stop_point_s = (self.BM_state.ref_position_s
                                         + self.VP_state.dist_preceding_veh
                                         + self.VP_state.stop_dist_preceding_veh
                                         - self.VP_state.min_safety_dist)
            if self.VP_state.vel_preceding_veh < self.behavior_config.standing_obstacle_vel:
                # calculate the closest point to come to a stop behind preceding vehicle
                stop_point_preceding_veh = (self.BM_state.ref_position_s
                                            + self.VP_state.dist_preceding_veh
                                            - self.BM_state.vehicle_params.length / 2
                                            - 0.5)
                self.BM_state.stop_point_s = min(comfort_stopping_point_s, stop_point_preceding_veh)
                self.BM_state.desired_velocity_stop_point = 0.0
                self.BM_state.stop_point_dist = self.BM_state.stop_point_s - self.BM_state.ref_position_s
                self.BM_state.stop_point_mode = "s-pos: preceding vehicle | vel: 0"
                return  # special case where minimal distance is not wanted
            elif (self.FSM_state.behavior_state_static in ["TrafficLight", "Crosswalk", "StopSign", "YieldSign"] and
                    self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "Stopping" and
                    ttc_stopping_stop_point_s < self.BM_state.current_static_goal.stop_point_s):
                # car in front of ego vehicle, stop behind the car in front
                self.BM_state.stop_point_s = min(ttc_stopping_stop_point_s, comfort_stopping_point_s)
                # don't accelerate during stopping
                self.BM_state.desired_velocity_stop_point = min(self.VP_state.vel_preceding_veh,
                                                                self.BM_state.ego_state.velocity)
            else:
                # use TTC as measure for stopping point calculation
                self.BM_state.stop_point_s = min(ttc_stopping_stop_point_s, comfort_stopping_point_s)
                self.BM_state.desired_velocity_stop_point = self.VP_state.vel_preceding_veh

        # make sure to not overshoot the desired position with nose of the car
        self.BM_state.stop_point_s -= self.BM_state.vehicle_params.length / 2
        # make sure minimal stop_point distance margins are met
        self.BM_state.stop_point_s = max(self.BM_state.ref_position_s + stop_point_min_dist,
                                         self.BM_state.stop_point_s, 0)

        ####################################################################################################
        #                              Calculating Stop Point from Final Goal                              #
        ####################################################################################################
        final_s_position, final_velocity, velocity_adapt_s_position = (
            hf.calculate_stop_distance_and_velocity_to_final_goal(self.BM_state))
        if final_s_position is not None:
            self.BM_state.stop_point_s = min(final_s_position, self.BM_state.stop_point_s)
        approx_s_pos_next_planning_cycle = (self.BM_state.ref_position_s + (
            self.BM_state.ego_state.velocity * self.BM_state.dt * self.BM_state.config.behavior.replanning_frequency))
        if final_velocity is not None and velocity_adapt_s_position <= approx_s_pos_next_planning_cycle:
            self.BM_state.desired_velocity_stop_point = final_velocity

        # determine stop point mode
        self.BM_state.stop_point_mode = "s-pos: "
        stop_positions = [
            (None if self.BM_state.current_static_goal.stop_point_s is None else
             self.BM_state.current_static_goal.stop_point_s - self.BM_state.vehicle_params.length / 2),
            (None if final_s_position is None else
             final_s_position - self.BM_state.vehicle_params.length / 2),
            (None if ttc_stopping_stop_point_s is None else
             ttc_stopping_stop_point_s - self.BM_state.vehicle_params.length / 2),
            self.BM_state.ref_position_s + stop_point_min_dist,
            comfort_stopping_point_s - self.BM_state.vehicle_params.length / 2,
            default_time_stopping_point_s - self.BM_state.vehicle_params.length / 2
        ]
        stop_positions_names = [
            "static goal",
            "final goal",
            "TTC",
            "minimal distance",
            "comfortable",
            "default time"
        ]
        closest_dist = stop_positions[-1]
        closest_idx = len(stop_positions) - 1
        for idx, s_pos in enumerate(stop_positions):
            if s_pos is not None and abs(self.BM_state.stop_point_s - s_pos) < closest_dist:
                closest_dist = abs(self.BM_state.stop_point_s - s_pos)
                closest_idx = idx
        self.BM_state.stop_point_mode += f"{stop_positions_names[closest_idx]} | vel: "
        if self.BM_state.desired_velocity_stop_point == 0.0:
            self.BM_state.stop_point_mode += "0"
        elif self.BM_state.desired_velocity_stop_point == final_velocity:
            self.BM_state.stop_point_mode += "final goal"
        elif self.BM_state.desired_velocity_stop_point == self.VP_state.vel_preceding_veh:
            self.BM_state.stop_point_mode += "preceding vehicle"
        elif self.BM_state.desired_velocity_stop_point == self.VP_state.goal_velocity:
            self.BM_state.stop_point_mode += "goal velocity"
        else:
            self.BM_state.stop_point_mode += "unknown"

        self.BM_state.stop_point_dist = self.BM_state.stop_point_s - self.BM_state.ref_position_s


class BehaviorModuleState(object):
    """Behavior Module State class containing all information the Behavior Module is working with."""

    def __init__(self):
        # general
        self.vehicle_params = None
        self.country = None
        self.scenario = None
        self.planning_problem = None
        self.goal_index = None
        self.priority_right = None
        self.plan_dynamics_only = None

        # Behavior Module inputs
        self.ego_state = None
        self.predictions = None
        self.time_step = None
        self.dt = None
        self.ego_id = None

        # FSM and Velocity Planner information
        self.FSM_state = FSMState()
        self.VP_state = VelocityPlannerState()
        self.PP_state = PathPlannerState()
        self.BP_state = BehaviorPlannerState()

        # Behavior Module information
        self.street_setting = None
        self.ref_position_s = None
        self.current_lanelet_id = None
        self.current_lanelet = None
        self.current_static_goal = None

        # velocity
        self.init_velocity = None
        self.speed_limit = None

        # navigation
        self.global_nav_route = None
        self.nav_lane_changes_left = 0
        self.nav_lane_changes_right = 0
        self.overtaking = None

        # stop point
        self.stop_point_s = None
        self.stop_point_dist = None
        self.desired_velocity_stop_point = None
        self.stop_point_mode = None


class FSMState(object):
    """FSM state class containing all information the Finite State Machine is working with."""

    def __init__(self):
        # street setting state
        self.street_setting = None

        # static behavior states
        self.behavior_state_static = None
        self.situation_state_static = None

        # dynamic behavior states
        self.behavior_state_dynamic = None
        self.situation_state_dynamic = None

        # time_step_counter
        self.situation_time_step_counter = None

        # vehicle status
        self.detected_lanelets = None

        # information
        self.lane_change_target_lanelet_id = None
        self.lane_change_target_lanelet = None
        self.obstacles_on_target_lanelet = None

        # information flags
        self.overtake_lange_changes_offset = None  # number of lane changes that need to be undone after overtaking; could also be impelmented differently by checking the the lane changes that the new reference path is giving

        # free space offset
        self.free_space_offset = 0
        self.change_velocity_for_lane_change = None

        # permission flags
        self.free_space_on_target_lanelet = None

        self.lane_change_left_ok = None
        self.lane_change_right_ok = None
        self.lane_change_left_done = None
        self.lane_change_right_done = None

        self.lane_change_prep_right_abort = None
        self.lane_change_prep_left_abort = None
        self.lane_change_right_abort = None
        self.lane_change_left_abort = None

        self.no_auto_lane_change = None

        self.turn_clear = None
        self.crosswalk_clear = None
        self.stop_yield_sign_clear = None
        self.intersection_clear = None

        # action flags
        self.do_lane_change = None
        self.undo_lane_change = None

        # reaction flags
        self.initiated_lane_change = None
        self.undid_lane_change = None

        # traffic light
        self.traffic_light_state = None
        self.slowing_car_for_traffic_light = None
        self.waiting_for_green_light = None


class VelocityPlannerState(object):
    """Velocity Planner State class containing all information the Velocity Planner is working with."""

    def __init__(self):
        # outputs
        self.desired_velocity = None
        self.goal_velocity = None
        self.velocity_mode = None

        # general
        self.ttc_norm = 8
        self.final_velocity_interval = None
        self.final_velocity_center = None
        self.speed_limit_default = None
        self.TTC = None
        self.MAX = None
        self.comfortable_stopping_distance = None

        # TTC velocity
        self.closest_preceding_vehicle = None
        self.pos_preceding_veh = None
        self.dist_preceding_veh = None
        self.vel_preceding_veh = None
        self.ttc_conditioned = None
        self.ttc_relative = None  # optimal relative velocity to the preceding vehicle
        self.stop_dist_preceding_veh = None
        self.min_safety_dist = None
        self.safety_dist = None

        # conditions
        self.condition_factor = None  # factor to express driving conditions of the vehicle; ∈ [0,1]
        self.lon_dyn_cond_factor = None  # factor to express longitudinal driving conditions; ∈ [0,1]
        self.lat_dyn_cond_factor = None  # factor to express lateral driving; ∈ [0,1]
        self.visual_cond_factor = None  # factor to express visual driving conditions; ∈ [0,1]

        # traffic light
        self.stop_distance = None
        self.dist_to_tl = None


class PathPlannerState(object):
    """Velocity Planner State class containing information about the Path Planner State"""

    def __init__(self):
        self.static_route_plan = None
        self.route_plan_ids = None
        self.reference_path = None
        self.reference_path_ids = None
        self.cl_ref_coordinate_system = None
        self.final_s_position_interval = None
        self.final_s_position_center = None


class BehaviorPlannerState(object):
    """ Behavior Planner State class

    This class is holding all externally relevant information of the Behavior Planner
    """

    def __init__(self):
        # FSM States
        self.street_setting = None  # string

        self.behavior_state_static = None  # string
        self.situation_state_static = None  # string

        self.behavior_state_dynamic = None  # string
        self.situation_state_dynamic = None  # string

        self.lane_change_target_lanelet_id = None  # string

        # Velocity Planner
        self.velocity = None  # float
        self.goal_velocity = None  # float
        self.desired_velocity = None  # float  # also passed separately
        self.TTC = None  # float
        self.MAX = None  # float

        self.slowing_car_for_traffic_light = None  # boolean
        self.waiting_for_green_light = None  # boolean

        self.condition_factor = None  # factor to express driving conditions of the vehicle; ∈ [0,1]
        self.lon_dyn_cond_factor = None  # factor to express longitudinal driving conditions; ∈ [0,1]
        self.lat_dyn_cond_factor = None  # factor to express lateral driving; ∈ [0,1]
        self.visual_cond_factor = None  # factor to express visual driving conditions; ∈ [0,1]

        # Path Planner
        # self.reference_path = None  # list of tuples of floats  # just passed separately
        self.reference_path_ids = None  # list of strings

        # Stop Points
        self.stop_point_dist = None
        self.desired_velocity_stop_point = None
        self.stop_point_mode = None

    def set_values(self, BM_state):
        """sets all values of this class, so that BM_state will not be part of the dict of this class, and returns a deepcopy"""

        # FSM States
        self.street_setting = BM_state.FSM_state.street_setting  # string

        self.behavior_state_static = BM_state.FSM_state.behavior_state_static  # string
        self.situation_state_static = BM_state.FSM_state.situation_state_static  # string

        self.behavior_state_dynamic = BM_state.FSM_state.behavior_state_dynamic  # string
        self.situation_state_dynamic = BM_state.FSM_state.situation_state_dynamic  # string

        self.lane_change_target_lanelet_id = BM_state.FSM_state.lane_change_target_lanelet_id  # string

        self.slowing_car_for_traffic_light = BM_state.FSM_state.slowing_car_for_traffic_light  # boolean
        self.waiting_for_green_light = BM_state.FSM_state.waiting_for_green_light  # boolean

        # Velocity Planner
        self.velocity = BM_state.ego_state.velocity if BM_state.ego_state is not None else BM_state.init_velocity  # float
        self.goal_velocity = BM_state.VP_state.goal_velocity  # float
        self.desired_velocity = BM_state.VP_state.desired_velocity  # float
        self.TTC = BM_state.VP_state.TTC  # float
        self.MAX = BM_state.VP_state.MAX  # float

        self.condition_factor = BM_state.VP_state.condition_factor  # ∈ [0,1]
        self.lon_dyn_cond_factor = BM_state.VP_state.lon_dyn_cond_factor  # ∈ [0,1]
        self.lat_dyn_cond_factor = BM_state.VP_state.lat_dyn_cond_factor  # ∈ [0,1]
        self.visual_cond_factor = BM_state.VP_state.visual_cond_factor  # ∈ [0,1]

        # Path Planner
        # self.reference_path = BM_state.PP_state.reference_path  # list of tuples of floats  # passed separately
        self.reference_path_ids = BM_state.PP_state.reference_path_ids  # list of strings

        # Stop Points
        self.stop_point_dist = BM_state.stop_point_dist
        self.desired_velocity_stop_point = BM_state.desired_velocity_stop_point
        self.stop_point_mode = BM_state.stop_point_mode

        return copy.deepcopy(self.__dict__)


class BehaviorOutput(object):
    """Class for collected Behavior Input for Reactive Planner"""

    def __init__(self, BM_state):
        self.desired_velocity = None
        self.reference_path = None
        self.stop_point_s = None
        self.desired_velocity_stop_point = None
        self.behavior_planner_state = BM_state.BP_state.set_values(BM_state)  # deepcopies values
