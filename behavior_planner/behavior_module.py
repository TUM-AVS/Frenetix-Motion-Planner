__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import copy
import os

from commonroad_route_planner.route_planner import RoutePlanner

import behavior_planner.utils.helper_functions as hf
from behavior_planner.utils.helper_logging import BehaviorLogger
from behavior_planner.utils.velocity_planner import VelocityPlanner
from behavior_planner.utils.path_planner import PathPlanner
from behavior_planner.utils.FSM_model import EgoFSM


class BehaviorModule(object):
    """
    Behavior Module: Coordinates Path Planner, Velocity Planner and Finite State Machine (FSM) to determine the
    reference path and desired velocity for the reactive planner.

    TODO: Include FSM
    """

    def __init__(self, scenario, planning_problem, init_ego_state, dt, config, log_path):
        """ Init Behavior Module.

        Args:
        pro_path (str): project path.
        scenario: scenario.
        init_ego_state : initialized ego state.
        """

        self.BM_state = BehaviorModuleState()  # behavior module information

        # load config
        self.BM_state.config = config
        self.behavior_config = self.BM_state.config.behavior
        self.BM_state.config.behavior.behavior_log_path_scenario = os.path.join(log_path, "behavior_logs")

        os.makedirs(self.behavior_config.behavior_log_path_scenario, exist_ok=True)

        # init behavior planner and load scenario information
        self.VP_state = self.BM_state.VP_state  # velocity planner information
        self.PP_state = self.BM_state.PP_state  # path planner information
        self.FSM_state = self.BM_state.FSM_state  # FSM information
        self.BM_state.init_velocity = init_ego_state.velocity
        self.BM_state.dt = dt

        self.BM_state.scenario = scenario
        self.BM_state.planning_problem = planning_problem

        self.BM_state.country = hf.find_country_traffic_sign_id(self.BM_state.scenario)
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=[],
                ego_state=init_ego_state,
                country=self.BM_state.country)

        # get navigation plan
        self.navigation = RoutePlanner(self.BM_state.scenario, self.BM_state.planning_problem)
        self._retrieve_nav_route()
        self._retrieve_lane_changes_from_navigation()

        # init path planner
        self.path_planner = PathPlanner(BM_state=self.BM_state)
        self.path_planner.execute_route_planning()

        # init ego FSM
        self.ego_FSM = EgoFSM(BM_state=self.BM_state)
        self.FSM_state = self.ego_FSM.FSM_state

        # init velocity planner
        self.velocity_planner = VelocityPlanner(BM_state=self.BM_state)

        # initialize loggers
        self.behavior_logger = BehaviorLogger(self.behavior_config)
        self.behavior_message_logger = self.behavior_logger.message_logger

        # outputs
        self.behavior_output = BehaviorOutput(self.BM_state)
        self.reference_path = self.BM_state.PP_state.reference_path
        self.desired_velocity = None
        self.flags = {"stopping_for_traffic_light": None,
                      "waiting_for_green_light": None
                      }

        # logging
        # self.behavior_logger.info("Behavior Module initialized")
        self.behavior_message_logger.debug("simulating scenario: " + str(self.BM_state.scenario.scenario_id))

    def execute(self, predictions, ego_state, time_step):
        """ Execute behavior module.

        TODO: measure time the behavior module takes to execute

        Args:
        predictions (dict): current predictions.
        ego_state (List): current state of ego vehicle.

        return: behavior_output (BehaviorOutput): Class holding all information for Reactive Planner
        """

        # traffic light testing
        # if time_step >= 50:
        #    (self.BM_state.scenario.lanelet_network.find_traffic_light_by_id(3835)).cycle[0].state = \
        #        TrafficLightState('green')
        #    (self.BM_state.scenario.lanelet_network.find_traffic_light_by_id(3835)).cycle[1].state = \
        #        TrafficLightState('green')
        #    (self.BM_state.scenario.lanelet_network.find_traffic_light_by_id(3835)).cycle[2].state = \
        #        TrafficLightState('green')
        #    msg_logger.info("Testing:: Traffic Light now green")

        # inputs
        self.BM_state.predictions = predictions
        self.BM_state.ego_state = ego_state
        self.BM_state.time_step = ego_state.time_step  # time_step TODO see where the time_step comes from

        self._get_ego_position(ego_state)

        self.BM_state.future_factor = int(self.BM_state.ego_state.velocity // 4) + 1  # for lane change maneuvers
        self._collect_necessary_information()

        # execute velocity planner
        self.velocity_planner.execute()
        self.desired_velocity = self.VP_state.desired_velocity

        # execute FSM
        self.ego_FSM.execute()

        # execute path planner
        if self.FSM_state.do_lane_change:
            self.path_planner.execute_lane_change()
        if self.FSM_state.undo_lane_change:
            self.path_planner.undo_lane_change()
        self.reference_path = self.PP_state.reference_path

        # update behavior flags
        self.flags["stopping_for_traffic_light"] = self.FSM_state.slowing_car_for_traffic_light
        self.flags["waiting_for_green_light"] = self.FSM_state.waiting_for_green_light

        # update behavior input for reactive planner
        self.behavior_output.reference_path = self.reference_path
        self.behavior_output.desired_velocity = self.desired_velocity
        # self.behavior_output.behavior_planner_state = vars(self.BM_state.BP_state.set_values(self.BM_state))
        self.behavior_output.behavior_planner_state = self.BM_state.BP_state.set_values(self.BM_state)

        # set current States vor Visualization
        self.behavior_config.behavior_state_static = self.FSM_state.behavior_state_static
        self.behavior_config.situation_state_static = self.FSM_state.situation_state_static
        self.behavior_config.behavior_state_dynamic = self.FSM_state.behavior_state_dynamic
        self.behavior_config.situation_state_dynamic = self.FSM_state.situation_state_dynamic

        # logging
        self.behavior_message_logger.debug("VP velocity mode: " + str(self.VP_state.velocity_mode))
        self.behavior_message_logger.debug("VP TTC velocity: " + str(self.VP_state.TTC))
        self.behavior_message_logger.debug("VP MAX velocity: " + str(self.VP_state.MAX))
        if self.VP_state.closest_preceding_vehicle is not None:
            self.behavior_message_logger.debug("VP position of preceding vehicle: " + str(self.VP_state.closest_preceding_vehicle.get('pos_list')[0]))
            self.behavior_message_logger.debug("VP velocity of preceding vehicle: " + str(self.VP_state.vel_preceding_veh))
            self.behavior_message_logger.debug("VP distance to preceding vehicle: " + str(self.VP_state.dist_preceding_veh))
            self.behavior_message_logger.debug("VP safety distance to preceding vehicle: " + str(self.VP_state.safety_dist))
        self.behavior_message_logger.debug("VP recommended velocity: " + str(self.VP_state.goal_velocity))
        self.behavior_message_logger.debug("BP recommended desired velocity: " + str(self.desired_velocity))
        self.behavior_message_logger.debug("current ego velocity: " + str(self.BM_state.ego_state.velocity))

        self.behavior_logger.log_data(self.BM_state.__dict__)

        return self.behavior_output

    def _retrieve_nav_route(self):
        global_nav_routes = self.navigation.plan_routes()
        self.BM_state.global_nav_route = global_nav_routes.retrieve_shortest_route()

    def _retrieve_lane_changes_from_navigation(self):
        self.BM_state.nav_lane_changes_left = 0
        self.BM_state.nav_lane_changes_right = 0
        lane_change_instructions = hf.retrieve_glb_nav_path_lane_changes(self.BM_state.global_nav_route)
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
        try:
            self.BM_state.nav_position_s = self.PP_state.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                ego_state.position[0], ego_state.position[1])[0]
        except:
            self.behavior_message_logger.error("Ego position out of navigation route coordinate system projection domain")

    def _collect_necessary_information(self):
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting_scenario = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=self.PP_state.reference_path_ids,
                ego_state=self.BM_state.ego_state,
                country=self.BM_state.country)

        self.BM_state.current_lanelet = \
            self.BM_state.scenario.lanelet_network.find_lanelet_by_id(self.BM_state.current_lanelet_id)


class BehaviorModuleState(object):
    """Behavior Module State class containing all information the Behavior Module is working with."""

    def __init__(self):
        # general
        self.vehicle_params = None
        self.country = None
        self.scenario = None
        self.planning_problem = None
        self.priority_right = None
        self.overtaking = None

        # Behavior Module inputs
        self.ego_state = None
        self.predictions = None
        self.time_step = None
        self.dt = None

        # FSM and Velocity Planner information
        self.FSM_state = FSMState()
        self.VP_state = VelocityPlannerState()
        self.PP_state = PathPlannerState()
        self.BP_state = BehaviorPlannerState()

        # Behavior Module information
        self.ref_position_s = None
        self.nav_position_s = None
        self.current_lanelet_id = None
        self.current_lanelet = None
        self.current_static_goal = None

        # velocity
        self.init_velocity = None
        self.speed_limit = None
        self.street_setting = None

        # navigation
        self.global_nav_route = None
        self.nav_lane_changes_left = 0
        self.nav_lane_changes_right = 0


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
        self.speed_limit_default = None
        self.TTC = None
        self.MAX = None

        # TTC velocity
        self.closest_preceding_vehicle = None
        self.dist_preceding_veh = None
        self.vel_preceding_veh = None
        self.ttc_conditioned = None
        self.ttc_relative = None  # optimal relative velocity to the preceding vehicle
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
        self.cl_nav_coordinate_system = None


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
        # self.goal_velocity = None  # string  # passed separately
        self.TTC = None  # float
        self.MAX = None  # float

        self.slowing_car_for_traffic_light = None  # boolean
        self.waiting_for_green_light = None  # boolean

        self.condition_factor = None  # factor to express driving conditions of the vehicle; ∈ [0,1]
        self.lon_dyn_cond_factor = None  # factor to express longitudinal driving conditions; ∈ [0,1]
        self.lat_dyn_cond_factor = None  # factor to express lateral driving; ∈ [0,1]
        self.visual_cond_factor = None  # factor to express visual driving conditions; ∈ [0,1]

        # Path Planner
        # self.reference_path = None  # list of tuples of floats  # passed separately
        self.reference_path_ids = None  # list of strings

    def set_values(self, BM_state):
        """sets all of this classes values, so that BM_state will not be part of the dict of this class"""

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
        # self.goal_velocity = BM_state.VP_state.goal_velocity  # string  # passed separately
        self.TTC = BM_state.VP_state.TTC  # float
        self.MAX = BM_state.VP_state.MAX  # float

        self.condition_factor = BM_state.VP_state.condition_factor  # ∈ [0,1]
        self.lon_dyn_cond_factor = BM_state.VP_state.lon_dyn_cond_factor  # ∈ [0,1]
        self.lat_dyn_cond_factor = BM_state.VP_state.lat_dyn_cond_factor  # ∈ [0,1]
        self.visual_cond_factor = BM_state.VP_state.visual_cond_factor  # ∈ [0,1]

        # Path Planner
        # self.reference_path = BM_state.PP_state.reference_path  # list of tuples of floats  # passed separately
        self.reference_path_ids = BM_state.PP_state.reference_path_ids  # list of strings

        return copy.deepcopy(self.__dict__)


class BehaviorOutput(object):
    """Class for collected Behavior Input for Reactive Planner"""

    def __init__(self, BM_state):
        self.desired_velocity = None
        self.reference_path = None
        self.behavior_planner_state = BM_state.BP_state.set_values(BM_state)
