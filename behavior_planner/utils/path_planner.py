__author__ = "Moritz Ellermann, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# general imports
import copy
from enum import Enum
from itertools import zip_longest
import numpy as np
import logging

# commonroad imports
from commonroad.geometry.shape import Polygon, ShapeGroup
from commonroad_route_planner.route_generation_strategies.default_generation_strategy import DefaultGenerationStrategy

from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem, smooth_ref_path

# project imports
import behavior_planner.utils.helper_functions as hf

# get logger
behavior_message_logger = logging.getLogger("Behavior_logger")


class PathPlanner(object):
    """
    Path Planner: Used by the Behavior Planner to determine the reference path, create a route plan with static goals
    for the FSM and execute lane change maneuvers.

    Possible static goals:
    Street Setting Highway: "StaticDefault", "LaneMerge", "RoadExit".
    Street Setting Country: "StaticDefault", "TurnLeft", "TurnRight", "RoadExit".
    Street Setting Urban: "StaticDefault", "TurnLeft", "TurnRight", "RoadExit", "StopSign", "YieldSign", "TrafficLight".

    TODO: include Turn and Road Exit detection
    TODO: include lane change maneuvers
    """

    def __init__(self, BM_state):
        """ Init Path Planner.

        Args:
        scenario (Scenario): scenario.
        global_nav_route (Route): global navigation route (CR built in reference path).
        """
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.PP_state = BM_state.PP_state

        # reference path planning
        self.reference_path_planner = ReferencePath(BM_state=self.BM_state)

        # route planning
        self.route_planner = RoutePlan(lanelet_network=self.BM_state.scenario.lanelet_network,
                                       global_nav_route=self.BM_state.global_nav_route,
                                       config_sim=self.BM_state.config,
                                       ccosy=self.reference_path_planner.cl_ref_coordinate_system,
                                       country=self.BM_state.country,
                                       street_setting_scenario=self.BM_state.street_setting)

        self.PP_state.reference_path = self.reference_path_planner.reference_path
        self.PP_state.reference_path_ids = self.reference_path_planner.list_ids_ref_path
        self.PP_state.route_plan_ids = self.route_planner.global_nav_path_ids
        self.PP_state.cl_ref_coordinate_system = self.reference_path_planner.cl_ref_coordinate_system

        self.PP_state.final_s_position_interval, self.PP_state.final_s_position_center, self.BM_state.goal_index = (
            hf.calculate_goal_s_position_interval(
                self.PP_state.reference_path,
                self.BM_state.planning_problem.goal.state_list,
                self.PP_state.cl_ref_coordinate_system
            ))

        if self.BM_state.config.behavior.visualize_static_route:
            size = None
            pos = None
            # if str(self.BM_state.scenario.scenario_id).__contains__("Tjunction"):
            #     size = (70, 24)
            #     pos = (-14, 5)
            # elif str(self.BM_state.scenario.scenario_id).__contains__("Zip"):
            #     size = (200, 20)
            #     pos = (-90, 5)
            # elif str(self.BM_state.scenario.scenario_id).__contains__("Carcarana"):
            #     size = (155, 110)
            #     pos = (135, 465)
            # elif str(self.BM_state.scenario.scenario_id).__contains__("Lohmar"):
            #     size = (100, 100)
            #     pos = (-15, 22)
            # elif str(self.BM_state.scenario.scenario_id).__contains__("Stu"):
            #     size = (40, 140)
            #     pos = (-80, 250)
            # else:
            #     size = None
            #     pos = None

            hf.visualize_static_route(self.BM_state.scenario,
                                      self.BM_state.planning_problem,
                                      self.route_planner.static_route_plan,
                                      f"logs/{self.BM_state.scenario.scenario_id}/behavior_logs/",
                                      self.reference_path_planner.cl_ref_coordinate_system,
                                      size=size,
                                      pos=pos)

    def execute_route_planning(self):
        """ Execute path planners static goal planning along the navigation route. Time horizont is the CC Scenario
        Returns: route plan with static goals along navigation route
        """
        self.route_planner.execute_static_planning()

        self.PP_state.static_route_plan = self.route_planner.static_route_plan
        self.PP_state.route_plan_ids = self.route_planner.global_nav_path_ids

    def execute_lane_change(self):
        """ Execute reference path planner and do lane change maneuver
        Returns: updated reference_path, updated curvilinear reference coordinate system
        """
        self.reference_path_planner.create_lane_change(ego_state=self.BM_state.ego_state,
                                                       current_lanelet_id=self.BM_state.current_lanelet_id,
                                                       goal_lanelet_id=self.FSM_state.lane_change_target_lanelet_id)
        self.FSM_state.initiated_lane_change = True

        self.PP_state.reference_path = self.reference_path_planner.reference_path
        self.PP_state.reference_path_ids = self.reference_path_planner.list_ids_ref_path
        self.PP_state.cl_ref_coordinate_system = self.reference_path_planner.cl_ref_coordinate_system

    def undo_lane_change(self):
        """ Execute reference path planner and do lane change maneuver
        Returns: updated reference_path, updated curvilinear reference coordinate system
        """
        self.reference_path_planner.create_lane_change(ego_state=self.BM_state.ego_state,
                                                       current_lanelet_id=self.BM_state.current_lanelet_id,
                                                       goal_lanelet_id=self.BM_state.current_lanelet_id)
        # self.FSM_state.lane_change_right_abort = None
        # self.FSM_state.lane_change_left_abort = None

        self.PP_state.reference_path = self.reference_path_planner.reference_path
        self.PP_state.reference_path_ids = self.reference_path_planner.list_ids_ref_path
        self.PP_state.cl_ref_coordinate_system = self.reference_path_planner.cl_ref_coordinate_system


class ReferencePath(object):
    """ Reference Path: object holding the reference path for the reactive planner. Creates straight base reference path
    with initialization."""
    def __init__(self, BM_state):

        self.BM_state = BM_state
        self.config_sim = BM_state.config

        self.global_nav_route = None
        self.reference_path = None
        self.list_ids_ref_path = None
        self.cl_ref_coordinate_system = None

        self._calc_global_nav_route()

    def _calc_global_nav_route(self):
        general_route = hf.get_shortest_route_cr(self.BM_state.scenario, self.BM_state.planning_problem)
        last_route_part = copy.deepcopy(general_route)
        my_reference_path_parts = [copy.deepcopy(general_route.reference_path)]
        my_planning_problem = copy.deepcopy(self.BM_state.planning_problem)

        lane_change_section_lanelets = []
        for lane_change in hf.retrieve_glb_nav_path_lane_changes(self.BM_state.scenario, last_route_part):
            if len(lane_change) >= 3:
                lane_change_section_lanelets += lane_change[1:-1]
        if len(lane_change_section_lanelets) > 0 and self.BM_state.config.behavior.stepwise_lane_changes:
            behavior_message_logger.debug(f"inserting {len(lane_change_section_lanelets)} "
                                          f"lane changes into reference path")
            midpoints = []
            for lanelet_idx in lane_change_section_lanelets:
                interval = [0, 0]
                for idx, point in enumerate(last_route_part.reference_path):
                    if self.BM_state.scenario.lanelet_network.find_lanelet_by_id(lanelet_idx).polygon.contains_point(np.array(point)):
                        if interval[0] == 0:
                            interval[0] = idx
                        interval[1] = idx
                    else:
                        if interval[0] != 0:
                            break
                midpoint_idx = interval[0] + int((interval[1] - interval[0]) / 2)
                midpoints.append(last_route_part.reference_path[midpoint_idx])

            # buffer for overlapping path points
            overlap_buffer = 2
            for midpoint in midpoints:
                # set the goal state to the end of the previous section
                goal_region_size = (1 / 2) / 2
                my_planning_problem.goal._lanelets_of_goal_position = None
                my_planning_problem.goal.state_list[0].position = ShapeGroup([Polygon(np.array([
                    [midpoint[0] - goal_region_size, midpoint[1] - goal_region_size],
                    [midpoint[0] + goal_region_size, midpoint[1] - goal_region_size],
                    [midpoint[0] + goal_region_size, midpoint[1] + goal_region_size],
                    [midpoint[0] - goal_region_size, midpoint[1] + goal_region_size],
                    [midpoint[0] - goal_region_size, midpoint[1] - goal_region_size]
                ]))])

                # Do the first planning
                my_reference_path_parts[-1] = copy.deepcopy(
                    hf.get_shortest_route_cr(self.BM_state.scenario, my_planning_problem).reference_path)
                # cut the path to matching length
                cutting_points = [0, len(my_reference_path_parts[-1]) - 1]
                for i in range(len(my_reference_path_parts[-1]) - 1):
                    if len(my_reference_path_parts) > 1:
                        candidate = np.linalg.norm(
                            my_planning_problem.initial_state.position - my_reference_path_parts[-1][i])
                        curr_best = np.linalg.norm(
                            my_planning_problem.initial_state.position - my_reference_path_parts[-1][cutting_points[0]])
                        if candidate < curr_best:
                            cutting_points[0] = i
                    candidate = np.linalg.norm(midpoint - my_reference_path_parts[-1][i])
                    curr_best = np.linalg.norm(midpoint - my_reference_path_parts[-1][cutting_points[1]])
                    if candidate < curr_best:
                        cutting_points[1] = i
                cutting_points[1] = min(cutting_points[1] + 1 + overlap_buffer, len(my_reference_path_parts[-1]))
                my_reference_path_parts[-1] = my_reference_path_parts[-1][cutting_points[0]:cutting_points[1]]

                # set the initial state to the beginning of the next section
                my_planning_problem.initial_state.position = midpoint
                midpoint_lanelet = [l_id for l_id in self.BM_state.scenario.lanelet_network.find_lanelet_by_position([midpoint])[0]
                                    if l_id in general_route.lanelet_ids][0]
                closest_center_vec = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(midpoint_lanelet).center_vertices[0]
                curr_best = np.linalg.norm(
                            midpoint - self.BM_state.scenario.lanelet_network.find_lanelet_by_id(midpoint_lanelet).center_vertices[0])
                for idx, point in enumerate(self.BM_state.scenario.lanelet_network.find_lanelet_by_id(midpoint_lanelet).center_vertices[1:]):
                    candidate = np.linalg.norm(midpoint - point)
                    if candidate < curr_best:
                        curr_best = candidate
                        closest_center_vec = np.array([
                            point[0] - self.BM_state.scenario.lanelet_network.find_lanelet_by_id(midpoint_lanelet).center_vertices[idx - 1][0],
                            point[1] - self.BM_state.scenario.lanelet_network.find_lanelet_by_id(midpoint_lanelet).center_vertices[idx - 1][1]])
                my_planning_problem.initial_state.orientation = hf.angle_of_vector(closest_center_vec)
                # set the goal state to the end of the next section
                my_planning_problem.goal = copy.deepcopy(self.BM_state.planning_problem.goal)

                # Do the second planning
                last_route_part = hf.get_shortest_route_cr(self.BM_state.scenario, my_planning_problem)
                my_reference_path_parts.append(copy.deepcopy(last_route_part.reference_path))
                # cut the path to matching length
                cutting_point = 0
                for i in range(len(my_reference_path_parts[-1]) - 1):
                    candidate = np.linalg.norm(
                        my_planning_problem.initial_state.position - my_reference_path_parts[-1][i])
                    curr_best = np.linalg.norm(
                        my_planning_problem.initial_state.position - my_reference_path_parts[-1][cutting_point])
                    if candidate < curr_best:
                        cutting_point = i
                cutting_point = max(0, cutting_point - overlap_buffer)
                my_reference_path_parts[-1] = my_reference_path_parts[-1][cutting_point:]

        my_reference_path = hf.concatenate_reference_path_parts(my_reference_path_parts)
        my_reference_path = hf.extend_points(my_reference_path,
                                             step_size=self.BM_state.config.behavior.dist_between_points,
                                             dist_to_inter=self.BM_state.config.behavior.distance_self_intersection)
        my_reference_path = hf.smooth_reference_path(my_reference_path)

        # generate new Route with the calculated ReferencePath and set class variables accordingly
        general_route.reference_path = my_reference_path
        self.global_nav_route = DefaultGenerationStrategy.update_route(general_route, my_reference_path)
        self.reference_path = self.global_nav_route.reference_path
        self.list_ids_ref_path = self.global_nav_route.lanelet_ids
        self._update_cl_ref_coordinate_system()
        # update BM_state
        self.BM_state.global_nav_route = self.global_nav_route

    def _update_cl_ref_coordinate_system(self):
        self.cl_ref_coordinate_system = CoordinateSystem(reference=self.reference_path, config_sim=self.config_sim)

    def create_lane_change(self, ego_state, current_lanelet_id, goal_lanelet_id, number_vertices_lane_change=6):
        old_path = self.reference_path[:]
        # create straight reference path on goal lanelet
        new_path_ids = hf.create_consecutive_lanelet_id_list(self.BM_state.scenario.lanelet_network, goal_lanelet_id,
                                                             self.BM_state.PP_state.route_plan_ids)
        old_ref_path_ids = self.list_ids_ref_path[:self.list_ids_ref_path.index(current_lanelet_id)+1]
        self.list_ids_ref_path = old_ref_path_ids + new_path_ids
        new_path = hf.compute_straight_reference_path(self.BM_state.scenario.lanelet_network, new_path_ids,
                                                      self.BM_state.config.behavior.dist_between_points)
        # cut old and new path at current position
        cut_idx_old = np.argmin([np.linalg.norm(x) for x in (np.subtract(old_path, ego_state.position))])
        cut_idx_new = np.argmin([np.linalg.norm(x) for x in (np.subtract(new_path, ego_state.position))])
        old_path = old_path[:cut_idx_old + self.BM_state.future_factor, :]
        new_path = new_path[self.BM_state.future_factor + cut_idx_new + number_vertices_lane_change:, :]
        # create final reference path
        reference_path = np.concatenate((old_path, new_path), axis=0)
        self.reference_path = smooth_ref_path(reference_path)
        self._update_cl_ref_coordinate_system()


class RoutePlan(object):
    """ Route Plan: object holding static route plan and navigation route."""

    def __init__(self, lanelet_network, global_nav_route, config_sim, ccosy, country, street_setting_scenario="Urban"):

        self.lanelet_network = lanelet_network
        self.global_nav_route = global_nav_route
        self.global_nav_path = global_nav_route.reference_path
        self.global_nav_path_ids = global_nav_route.lanelet_ids
        self.config_sim = config_sim
        self.cl_ref_coordinate_system = ccosy
        self.country = country
        self.street_setting_scenario = street_setting_scenario

        self.static_route_plan = None

        self.yield_signs = []
        self.stop_signs = []
        self.traffic_lights = []
        self.turns = []
        self.road_exits = []
        self.lane_merges = []
        self.intersections = []

        self.execute_static_planning()

    def execute_static_planning(self):
        """Creates a plan of all static intermediate goals. Sets beginning and end point with the cl cosy coordinate s
        along the reference path.

        TODO: detect turns, crosswalk and road exits
        """
        self.static_route_plan = []

        self._look_for_traffic_lights_and_signs()
        self._look_for_lane_merges()
        self._look_for_intersections()
        # self.look_for_road_exits()
        # self.look_for_turns()

        for (stop_sign, yield_sign, traffic_light, road_exit, lane_merge, intersection) in \
                zip_longest(self.stop_signs, self.yield_signs, self.traffic_lights, self.road_exits, self.lane_merges,
                            self.intersections):
            for static_goal in (stop_sign, yield_sign, traffic_light, road_exit, lane_merge, intersection):

                if static_goal is not None:
                    goal = None
                    prep = None

                    # static goal length depends on the estimated speed
                    preparation_time = self.config_sim.behavior.preparation_time
                    goal_time = self.config_sim.behavior.goal_time

                    speed_factor = hf.get_speed_limit(
                        [static_goal.get('goal_lanelet_id')] +
                        self.lanelet_network.find_lanelet_by_id(static_goal.get('goal_lanelet_id')).predecessor,
                        self.lanelet_network, self.country)
                    # TODO hard coded values
                    if speed_factor is None:
                        if self.street_setting_scenario == "Highway":
                            speed_factor = 130 / 3.6
                        elif self.street_setting_scenario == "Country":
                            speed_factor = 100 / 3.6
                        elif self.street_setting_scenario == "Urban":
                            speed_factor = 50 / 3.6
                        else:
                            speed_factor = 50 / 3.6
                    speed_factor = min(130 / 3.6, speed_factor)  # don't exceed Richtgeschwindigkeit

                    static_prep_goal_length = speed_factor * preparation_time
                    static_goal_length = speed_factor * goal_time

                    # all static goals with stop line
                    if static_goal.get('type') in ['StopSign', 'YieldSign', 'TrafficLight', 'Crosswalk']:
                        start_s = max([0.001, static_goal.get('stop_position_s') - static_goal_length])
                        try:
                            start_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                            end_s = static_goal.get('position_s')
                            end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(end_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        traffic_sign_object = self.lanelet_network.find_traffic_light_by_id(static_goal.get('id'))
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=start_s,
                                          start_xy=start_xy,
                                          end_s=end_s,
                                          end_xy=end_xy,
                                          stop_point_s=static_goal.get('stop_position_s'),
                                          stop_point_xy=static_goal.get('stop_position_xy'),
                                          goal_object=traffic_sign_object,
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))
                        try:
                            prep_start_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(
                                    max([0.001, start_s - static_prep_goal_length]), 0).tolist()
                            prep_end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, start_s - static_prep_goal_length]),
                                          start_xy=prep_start_xy,
                                          end_s=start_s,
                                          end_xy=prep_end_xy,
                                          stop_point_s=static_goal.get('stop_position_s'),
                                          stop_point_xy=static_goal.get('stop_position_xy'),
                                          goal_object=traffic_sign_object,
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))

                    # all static goals with a lane change maneuver
                    elif static_goal.get('type') in ['LaneMerge', 'RoadExit']:
                        start_s = max([0.001, static_goal.get('position_s') - static_goal_length])
                        end_s = static_goal.get('position_s')
                        try:
                            start_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                            end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(end_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=start_s,
                                          start_xy=start_xy,
                                          end_s=end_s,
                                          end_xy=end_xy,
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))
                        try:
                            prep_start_xy = \
                                self.cl_ref_coordinate_system.convert_to_cartesian_coords(
                                    max([0.001, start_s - static_prep_goal_length]), 0).tolist()
                            prep_end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, start_s - static_prep_goal_length]),
                                          start_xy=prep_start_xy,
                                          end_s=start_s,
                                          end_xy=prep_end_xy,
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))

                    # turns
                    elif static_goal.get('type') in ['TurnRight', 'TurnLeft']:
                        start_s = static_goal.get('start_s')
                        end_s = static_goal.get('end_s')
                        try:
                            start_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                            end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(end_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=start_s,
                                          start_xy=start_xy,
                                          end_s=end_s,
                                          end_xy=end_xy)
                        try:
                            prep_start_xy = \
                                self.cl_ref_coordinate_system.convert_to_cartesian_coords(
                                    max([0.001, start_s - static_prep_goal_length]), 0).tolist()
                            prep_end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, start_s - static_prep_goal_length]),
                                          start_xy=prep_start_xy,
                                          end_s=start_s,
                                          end_xy=prep_end_xy)

                    # intersections
                    elif static_goal.get('type') == 'Intersection':
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=static_goal.get('start_s'),
                                          start_xy=static_goal.get('start_xy'),
                                          end_s=static_goal.get('end_s'),
                                          end_xy=static_goal.get('end_xy'),
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))
                        try:
                            prep_start_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(
                                max([0.001, static_goal.get('start_s') - static_prep_goal_length]), 0).tolist()
                            prep_end_xy = self.cl_ref_coordinate_system.convert_to_cartesian_coords(
                                static_goal.get('start_s'), 0).tolist()
                        except AttributeError:
                            behavior_message_logger.error(
                                f"PP start or stop s_position of {static_goal.get('type')} is out of projection domain")
                            continue
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, static_goal.get('start_s') - static_prep_goal_length]),
                                          start_xy=prep_start_xy,
                                          end_s=static_goal.get('start_s'),
                                          end_xy=prep_end_xy,
                                          goal_lanelet_id=static_goal.get('goal_lanelet_id'))

                    # just in case
                    else:
                        pass

                    self.static_route_plan += [prep, goal]

        # sort goals for cl cosy coordinate s
        self.static_route_plan.sort(key=lambda x: x.start_s)
        self._straighten_static_route_plan()

    def _look_for_traffic_lights_and_signs(self):
        self.yield_signs = []
        self.stop_signs = []
        self.traffic_lights = []

        for lanelet_id in self.global_nav_path_ids:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is None:
                continue
            # center point of stop line
            stop_position_x = (lanelet.stop_line.start[0] + lanelet.stop_line.end[0]) / 2
            stop_position_y = (lanelet.stop_line.start[1] + lanelet.stop_line.end[1]) / 2
            stop_position_xy = [stop_position_x, stop_position_y]
            try:
                stop_position_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                    stop_position_x, stop_position_y)[0]
            except:
                behavior_message_logger.error("PP stop line position of traffic sign or light is out of projection domain")
                stop_position_s = None

            if lanelet.stop_line.traffic_sign_ref is not None:
                for traffic_sign_id in lanelet.stop_line.traffic_sign_ref:
                    if traffic_sign_id is not None:
                        traffic_sign = self.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
                        traffic_sign_position_xy = [traffic_sign.position[0], traffic_sign.position[1]]
                        try:
                            traffic_sign_position_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                                traffic_sign_position_xy[0], traffic_sign_position_xy[1])[0]
                        except:
                            behavior_message_logger.error("PP position of traffic sign is out of projection domain")
                            traffic_sign_position_s = None
                        for traffic_sign_element in traffic_sign.traffic_sign_elements:

                            if stop_position_s is None and traffic_sign_position_s is not None:
                                stop_position_s = traffic_sign_position_s
                            elif stop_position_s is not None and traffic_sign_position_s is None:
                                traffic_sign_position_s = stop_position_s
                            elif stop_position_s is None and traffic_sign_position_s is None:
                                behavior_message_logger.warning("PP traffic sign is out of projection domain")
                                continue

                            if traffic_sign_element.traffic_sign_element_id.name == 'YIELD':
                                self.yield_signs += [{'id': traffic_sign_id,
                                                      'type': 'YieldSign',
                                                      'position_s': traffic_sign_position_s,
                                                      'position_xy': traffic_sign_position_xy,
                                                      'stop_position_s': stop_position_s,
                                                      'stop_position_xy': stop_position_xy,
                                                      'goal_lanelet_id': lanelet_id}]
                            if traffic_sign_element.traffic_sign_element_id.name == 'STOP':
                                self.stop_signs += [{'id': traffic_sign_id,
                                                     'type': 'StopSign',
                                                     'position_s': traffic_sign_position_s,
                                                     'position_xy': traffic_sign_position_xy,
                                                     'stop_position_s': stop_position_s,
                                                     'stop_position_xy': stop_position_xy,
                                                     'goal_lanelet_id': lanelet_id}]

            if lanelet.stop_line.traffic_light_ref is not None:
                for traffic_light_id in lanelet.stop_line.traffic_light_ref:
                    if traffic_light_id is not None:
                        traffic_light = self.lanelet_network.find_traffic_light_by_id(traffic_light_id)
                        traffic_light_position_xy = [traffic_light.position[0], traffic_light.position[1]]
                        try:
                            traffic_light_position_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                                traffic_light.position[0], traffic_light.position[1])[0]
                        except:
                            behavior_message_logger.warning('PP traffic light position out of projection domain')
                            traffic_light_position_s = None

                        if stop_position_s is None and traffic_light_position_s is not None:
                            stop_position_s = traffic_light_position_s
                        elif stop_position_s is not None and traffic_light_position_s is None:
                            traffic_light_position_s = stop_position_s
                        elif stop_position_s is None and traffic_light_position_s is None:
                            behavior_message_logger.warning("PP traffic light is out of projection domain")
                            continue

                        if traffic_light.active:
                            self.traffic_lights += [{'id': traffic_light_id,
                                                     'type': 'TrafficLight',
                                                     'position_xy': traffic_light_position_xy,
                                                     'stop_position_xy': stop_position_xy,
                                                     'position_s': traffic_light_position_s,
                                                     'stop_position_s': stop_position_s,
                                                     'goal_lanelet_id': lanelet_id}]

    def _look_for_lane_merges(self):
        self.lane_merges = []
        # TODO look for lane merges in the scenario but not compare the two incoming orientation but rather look if
        #  the lanes are in an intersection or not
        for lanelet_id in self.global_nav_path_ids:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            if len(lanelet.predecessor) > 1:  # one of the driven lanelets has two predecessors
                pred1 = self.lanelet_network.find_lanelet_by_id(lanelet.predecessor[0])
                pred2 = self.lanelet_network.find_lanelet_by_id(lanelet.predecessor[1])
                if np.allclose(pred1.center_vertices[-1], pred2.center_vertices[-1]):  # same end point of merging lanes
                    orient1 = pred1.center_vertices[1] - pred1.center_vertices[0]
                    orient2 = pred2.center_vertices[1] - pred2.center_vertices[0]
                    orient1 = orient1 / np.linalg.norm(orient1)
                    orient2 = orient2 / np.linalg.norm(orient2)
                    if np.allclose(orient1, orient2, atol=0.1):  # similar orientation or merging lanes
                        try:
                            merging_point_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[0][0], lanelet.center_vertices[0][1])[0]
                        except:
                            behavior_message_logger.error("PP merging point is out of projection domain")
                            continue
                        self.lane_merges += [{'type': 'LaneMerge',
                                              'position_xy': lanelet.center_vertices[0],
                                              'position_s': merging_point_s,
                                              'goal_lanelet_id': lanelet_id}]

    def _look_for_intersections(self):
        self.intersections = []

        for intersection in self.lanelet_network.intersections:
            for lanelet_id in self.global_nav_path_ids:
                for intersection_element in intersection.incomings:
                    if (lanelet_id in intersection_element.successors_left) or \
                            (lanelet_id in intersection_element.successors_right) or \
                            (lanelet_id in intersection_element.successors_straight):
                        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
                        start_xy = lanelet.center_vertices[0].tolist()
                        try:
                            start_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[0][0], lanelet.center_vertices[0][1])[0]
                        except:
                            start_s = None
                            behavior_message_logger.error("PP start of intersection out of projection domain")
                        end_xy = lanelet.center_vertices[-1].tolist()
                        try:
                            end_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[-1][0], lanelet.center_vertices[-1][1])[0]
                        except:
                            end_s = None
                            behavior_message_logger.error("PP end of intersection out of projection domain")

                        if start_s is None and end_s is not None:
                            start_s = max([0.001, end_s - 15])
                        elif start_s is not None and end_s is None:
                            try:
                                max_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(self.global_nav_path[-1])
                                end_s = min(start_s + 15, max_s)
                            except:
                                behavior_message_logger.error("PP end of reference path out of projection domain")
                                continue
                        elif start_s is None and end_s is None:
                            behavior_message_logger.warning("PP intersection is out of projection domain")
                            continue

                        self.intersections += [{'id': intersection_element.incoming_id,
                                                'type': 'Intersection',
                                                'start_xy': start_xy,
                                                'start_s': start_s,
                                                'end_xy': end_xy,
                                                'end_s': end_s,
                                                'goal_lanelet_id': lanelet_id}]

    def _look_for_road_exits(self):

        return self.road_exits

    def _look_for_turns(self):
        # turn look like --_^-- in se graf and lane changes curvature looks like --_^^_-- in se graf
        # maybe look at the reference path curvature: if greater than 0.03 it might be a turn
        # maybe rather look at the intervals and their values: intervals between y=0 points
        # look at the change of orientation of a lanelets center vertices
        # take interpoint distance into account
        # set start_s and end_s so start and beginning of curvatures spikes (up AND down)
        # sharp turns can happen without intersection

        return self.turns

    def _straighten_static_route_plan(self, recursion_counter=0):
        """Checks for overlapping static goals and straightens them out and fills gaps between goals with StaticDefault.
        """
        class StaticGoalPrio(Enum):
            # signs, preparing is less important than the sign itself
            TrafficLight = 95
            StopSign = 90
            YieldSign = 85
            Crosswalk = 80

            PrepareTrafficLight = 75
            PrepareStopSign = 70
            PrepareYieldSign = 65
            PrepareCrosswalk = 60

            # lanes
            TurnLeft = 45
            PrepareTurnLeft = 40

            TurnRight = 46
            PrepareTurnRight = 41

            LaneMerge = 35
            PrepareLaneMerge = 25  # rather respect road exit than LaneMerge preparation

            RoadExit = 30
            PrepareRoadExit = 20

            Intersection = 11
            PrepareIntersection = 10

            # other
            StaticDefault = 1
            removable = 0

            def __lt__(self, other):
                if isinstance(other, StaticGoalPrio):
                    return self.value < other.value
                raise NotImplemented

            def __le__(self, other):
                if isinstance(other, StaticGoalPrio):
                    return self.value <= other.value
                raise NotImplemented

            def __gt__(self, other):
                if isinstance(other, StaticGoalPrio):
                    return self.value > other.value
                raise NotImplemented

            def __ge__(self, other):
                if isinstance(other, StaticGoalPrio):
                    return self.value >= other.value
                raise NotImplemented

        end_nav_path_s = self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
            self.global_nav_path[-1][0],
            self.global_nav_path[-1][1])[0]
        # if no static goal object was found on reference path add only StaticDefault
        if len(self.static_route_plan) == 0:
            self.static_route_plan = [StaticGoal(goal_type='StaticDefault',
                                                 start_s=0,
                                                 start_xy=self.global_nav_path[0],
                                                 end_s=end_nav_path_s,
                                                 end_xy=self.global_nav_path[-1])]
        else:
            self.static_route_plan.sort(key=lambda x: x.end_s)
            # remove yield and stop signs at active traffic lights
            for i in range(len(self.static_route_plan)-1):
                if self.static_route_plan[i].end_s == self.static_route_plan[i + 1].end_s:
                    if self.static_route_plan[i].goal_type == 'TrafficLight' and \
                            self.static_route_plan[i + 1].goal_type == 'YieldSign':
                        self.static_route_plan = self.static_route_plan[:i + 1] + self.static_route_plan[i + 2:]
                    elif self.static_route_plan[i].goal_type == 'YieldSign' and \
                            self.static_route_plan[i + 1].goal_type == 'TrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i + 1:]
                    elif self.static_route_plan[i].goal_type == 'TrafficLight' and \
                            self.static_route_plan[i + 1].goal_type == 'StopSigns':
                        self.static_route_plan = self.static_route_plan[:i + 1] + self.static_route_plan[i + 2:]
                    elif self.static_route_plan[i].goal_type == 'StopSigns' and \
                            self.static_route_plan[i + 1].goal_type == 'TrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i + 1:]
                    if self.static_route_plan[i].goal_type == 'PrepareTrafficLight' and \
                            self.static_route_plan[i + 1].goal_type == 'PrepareYieldSign':
                        self.static_route_plan = self.static_route_plan[:i + 1] + self.static_route_plan[i + 2:]
                    elif self.static_route_plan[i].goal_type == 'PrepareYieldSign' and \
                            self.static_route_plan[i + 1].goal_type == 'PrepareTrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i + 1:]
                    elif self.static_route_plan[i].goal_type == 'PrepareTrafficLight' and \
                            self.static_route_plan[i + 1].goal_type == 'PrepareStopSigns':
                        self.static_route_plan = self.static_route_plan[:i + 1] + self.static_route_plan[i + 2:]
                    elif self.static_route_plan[i].goal_type == 'PrepareStopSigns' and \
                            self.static_route_plan[i + 1].goal_type == 'PrepareTrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i + 1:]

            static_route_static_default_goals = []
            self.static_route_plan.sort(key=lambda x: x.end_s)
            removable = "removable"
            current_goal_idx = 0
            preceding_goal_idx = 0
            for i in range(len(self.static_route_plan) - 1):
                i = len(self.static_route_plan) - i - 1
                preceding_goal_idx = i - 1
                # make sure to always compare with the last valid goal
                if self.static_route_plan[i].goal_type != removable:
                    current_goal_idx = i
                # cut overlapping goals
                if self.static_route_plan[current_goal_idx].start_s < self.static_route_plan[preceding_goal_idx].end_s:
                    # check if static goals are TrafficLights, Signs or Crosswalks
                    if min(StaticGoalPrio[self.static_route_plan[current_goal_idx].goal_type].value,
                           StaticGoalPrio[self.static_route_plan[preceding_goal_idx].goal_type].value) < 80:
                        # decide which goal is more important
                        if (StaticGoalPrio[self.static_route_plan[current_goal_idx].goal_type] <
                                StaticGoalPrio[self.static_route_plan[preceding_goal_idx].goal_type]):
                            # previous goal has higher priority -> shorten current goal start to end of previous goal
                            self.static_route_plan[current_goal_idx].start_s = self.static_route_plan[preceding_goal_idx].end_s
                            self.static_route_plan[current_goal_idx].start_xy = self.static_route_plan[preceding_goal_idx].end_xy
                            if self.static_route_plan[current_goal_idx].end_s <= self.static_route_plan[preceding_goal_idx].end_s:
                                # goals overlap at the end -> shorten current goal end to start of previous goal
                                self.static_route_plan[current_goal_idx].end_s = self.static_route_plan[preceding_goal_idx].start_s
                                self.static_route_plan[current_goal_idx].end_xy = self.static_route_plan[preceding_goal_idx].start_xy
                            if self.static_route_plan[current_goal_idx].end_s - self.static_route_plan[current_goal_idx].start_s <= 0:
                                # current goal can be deleted
                                self.static_route_plan[current_goal_idx].goal_type = removable
                        else:
                            # current goal has equal or higher priority -> shorten previous goal end to start of current goal
                            if (StaticGoalPrio[self.static_route_plan[current_goal_idx].goal_type].value < 80 and
                                    current_goal_idx + 1 < len(self.static_route_plan)):
                                if (self.static_route_plan[preceding_goal_idx].goal_type.startswith("Prepare") and
                                    self.static_route_plan[preceding_goal_idx].goal_type.endswith(
                                        self.static_route_plan[current_goal_idx + 1].goal_type)):
                                    # make sure not to separate preparation with its goal
                                    self.static_route_plan[current_goal_idx].end_s = self.static_route_plan[preceding_goal_idx].start_s
                                    self.static_route_plan[current_goal_idx].end_xy = self.static_route_plan[preceding_goal_idx].start_xy
                                    if self.static_route_plan[current_goal_idx].end_s - self.static_route_plan[current_goal_idx].start_s <= 0:
                                        # current goal can be deleted
                                        self.static_route_plan[current_goal_idx].goal_type = removable
                            else:
                                self.static_route_plan[preceding_goal_idx].end_s = self.static_route_plan[current_goal_idx].start_s
                                self.static_route_plan[preceding_goal_idx].end_xy = self.static_route_plan[current_goal_idx].start_xy
                                if self.static_route_plan[preceding_goal_idx].end_s - self.static_route_plan[preceding_goal_idx].start_s <= 0:
                                    # preceding goal can be deleted
                                    self.static_route_plan[preceding_goal_idx].goal_type = removable
                    else:
                        # static goals are TrafficLights, Signs or Crosswalks. Their end point is always relevant
                        # TODO maybe look at priorities if goal are closer than a predefined threshold
                        self.static_route_plan[current_goal_idx].start_s = self.static_route_plan[preceding_goal_idx].end_s
                        self.static_route_plan[current_goal_idx].start_xy = self.static_route_plan[preceding_goal_idx].end_xy
                        if self.static_route_plan[current_goal_idx].end_s - self.static_route_plan[current_goal_idx].start_s <= 0:
                            # current goal can be deleted
                            behavior_message_logger.warning(
                                f"PP {self.static_route_plan[current_goal_idx].goal_type} at s-position "
                                f"'{self.static_route_plan[current_goal_idx].end_s}' is not considered because of "
                                f"{self.static_route_plan[preceding_goal_idx].goal_type} at s-position "
                                f"'{self.static_route_plan[preceding_goal_idx].end_s}'")
                            self.static_route_plan[current_goal_idx].goal_type = removable
                elif self.static_route_plan[current_goal_idx].start_s > self.static_route_plan[preceding_goal_idx].end_s:
                    static_route_static_default_goals.append(StaticGoal(
                        goal_type='StaticDefault',
                        start_s=self.static_route_plan[preceding_goal_idx].end_s,
                        start_xy=self.static_route_plan[preceding_goal_idx].end_xy,
                        end_s=self.static_route_plan[current_goal_idx].start_s,
                        end_xy=self.static_route_plan[current_goal_idx].start_xy
                    ))
            idx = 0
            while idx < len(self.static_route_plan):
                if self.static_route_plan[idx].goal_type == removable:
                    self.static_route_plan = self.static_route_plan[:idx] + self.static_route_plan[idx + 1:]
                else:
                    idx += 1

            self.static_route_plan += static_route_static_default_goals
            self.static_route_plan.sort(key=lambda x: x.start_s)

        # add StaticDefault at beginning
        if self.static_route_plan[0].start_s > 0:
            self.static_route_plan = [StaticGoal(goal_type='StaticDefault',
                                                 start_s=0,
                                                 start_xy=self.global_nav_path[0],
                                                 end_s=self.static_route_plan[0].start_s,
                                                 end_xy=self.static_route_plan[0].start_xy)] + self.static_route_plan
        # add StaticDefault at end
        if self.static_route_plan[-1].end_s != end_nav_path_s:
            self.static_route_plan += [StaticGoal(goal_type='StaticDefault',
                                                  start_s=self.static_route_plan[-1].end_s,
                                                  start_xy=self.static_route_plan[-1].end_xy,
                                                  end_s=end_nav_path_s,
                                                  end_xy=self.global_nav_path[-1])]
        if not self._valid_static_route_plan():
            if recursion_counter > 5:
                behavior_message_logger.error("PP creating a valid static route plan failed")
            else:
                behavior_message_logger.debug("PP straightening static route plan again")
                return self._straighten_static_route_plan(recursion_counter + 1)
        return self.static_route_plan

    def _valid_static_route_plan(self) -> bool:
        if self.static_route_plan[0].start_s != 0:
            return False
        for idx, static_goal in enumerate(self.static_route_plan[:-1]):
            if static_goal.end_s != self.static_route_plan[idx + 1].start_s:
                return False
        if self.static_route_plan[-1].end_s != self.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                self.global_nav_path[-1][0],
                self.global_nav_path[-1][1])[0]:
            return False
        return True


class StaticGoal(object):
    def __init__(self, goal_type, start_s, start_xy, end_s, end_xy, stop_point_s=None,
                 stop_point_xy=None, goal_object=None, goal_lanelet_id=None):
        self.goal_type = goal_type
        self.start_s = start_s
        self.start_xy = start_xy
        self.end_s = end_s
        self.end_xy = end_xy
        self.stop_point_s = stop_point_s
        self.stop_point_xy = stop_point_xy
        self.goal_object = goal_object
        self.goal_lanelet_id = goal_lanelet_id
