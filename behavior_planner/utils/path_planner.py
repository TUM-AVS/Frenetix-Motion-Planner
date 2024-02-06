__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import behavior_planner.utils.helper_functions as hf
from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem, smooth_ref_path

from itertools import zip_longest
import numpy as np
import logging

# get logger
msg_logger = logging.getLogger("Message_logger")


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

        # route planning
        self.route_planner = RoutePlan(lanelet_network=BM_state.scenario.lanelet_network,
                                       global_nav_route=BM_state.global_nav_route)
        self.PP_state.route_plan_ids = self.route_planner.global_nav_path_ids

        self.PP_state.cl_nav_coordinate_system = self.route_planner.cl_nav_coordinate_system

        # reference path planning
        self.reference_path_planner = ReferencePath(lanelet_network=BM_state.scenario.lanelet_network,
                                                    global_nav_route=BM_state.global_nav_route,
                                                    BM_state=BM_state)

        self.PP_state.reference_path = self.reference_path_planner.reference_path
        self.PP_state.reference_path_ids = self.reference_path_planner.list_ids_ref_path
        self.PP_state.cl_ref_coordinate_system = self.reference_path_planner.cl_ref_coordinate_system

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
        #self.FSM_state.lane_change_right_abort = None
        #self.FSM_state.lane_change_left_abort = None

        self.PP_state.reference_path = self.reference_path_planner.reference_path
        self.PP_state.reference_path_ids = self.reference_path_planner.list_ids_ref_path
        self.PP_state.cl_ref_coordinate_system = self.reference_path_planner.cl_ref_coordinate_system


class RoutePlan(object):
    """ Route Plan: object holding static route plan and navigation route."""

    def __init__(self, lanelet_network, global_nav_route):

        self.lanelet_network = lanelet_network
        self.global_nav_route = global_nav_route
        self.global_nav_path = global_nav_route.reference_path
        self.global_nav_path_ids = global_nav_route.list_ids_lanelets
        self.cl_nav_coordinate_system = CoordinateSystem(reference=self.global_nav_path)

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

        TODO: at the moment only traffic lights, stop and yield sign and lane merges are detected
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
                    # all static goals with stop line
                    if static_goal.get('type') in ['StopSign', 'YieldSign', 'TrafficLight']:
                        distance = 20
                        start_s = max([0.001, static_goal.get('stop_position_s') - distance])
                        start_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        end_s = static_goal.get('position_s')
                        end_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(end_s, 0).tolist()
                        traffic_sign_object = self.lanelet_network.find_traffic_light_by_id(static_goal.get('id'))
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=start_s,
                                          start_xy=start_xy,
                                          end_s=end_s,
                                          end_xy=end_xy,
                                          stop_point_s=static_goal.get('stop_position_s'),
                                          stop_point_xy=static_goal.get('stop_position_xy'),
                                          goal_object=traffic_sign_object)
                        prep_start_xy =\
                            self.cl_nav_coordinate_system.convert_to_cartesian_coords(
                                max([0.001, start_s - 10]), 0).tolist()
                        prep_end_xy = \
                            self.cl_nav_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, start_s - 10]),
                                          start_xy=prep_start_xy,
                                          end_s=start_s,
                                          end_xy=prep_end_xy,
                                          stop_point_s=static_goal.get('stop_position_s'),
                                          stop_point_xy=static_goal.get('stop_position_xy'),
                                          goal_object=traffic_sign_object)

                    # all static goals with a lane change maneuver
                    elif static_goal.get('type') in ['LaneMerge', 'RoadExit']:
                        distance = 50
                        start_s = max([0.001, static_goal.get('position_s') - distance])
                        start_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        end_s = static_goal.get('position_s')
                        end_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(end_s, 0).tolist()
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=start_s,
                                          start_xy=start_xy,
                                          end_s=end_s,
                                          end_xy=end_xy)
                        prep_start_xy = \
                            self.cl_nav_coordinate_system.convert_to_cartesian_coords(
                                max([0.001, start_s - 10]), 0).tolist()
                        prep_end_xy = \
                            self.cl_nav_coordinate_system.convert_to_cartesian_coords(start_s, 0).tolist()
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, start_s - 10]),
                                          start_xy=prep_start_xy,
                                          end_s=start_s,
                                          end_xy=prep_end_xy)

                    # intersections
                    elif static_goal.get('type') == 'Intersection':
                        goal = StaticGoal(goal_type=static_goal.get('type'),
                                          start_s=static_goal.get('start_s'),
                                          start_xy=static_goal.get('start_xy'),
                                          end_s=static_goal.get('end_s'),
                                          end_xy=static_goal.get('end_xy'))
                        prep_start_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(
                            max([0.001, static_goal.get('start_s') - 10]), 0).tolist()
                        prep_end_xy = self.cl_nav_coordinate_system.convert_to_cartesian_coords(
                            static_goal.get('start_s'), 0).tolist()
                        prep = StaticGoal(goal_type='Prepare' + static_goal.get('type'),
                                          start_s=max([0.001, static_goal.get('start_s') - 10]),
                                          start_xy=prep_start_xy,
                                          end_s=static_goal.get('start_s'),
                                          end_xy=prep_end_xy)
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
                stop_position_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                    stop_position_x, stop_position_y)[0]
            except:
                msg_logger.error("PP stop line position of traffic sign or light is out of projection domain")
                stop_position_s = None

            if lanelet.stop_line.traffic_sign_ref is not None:
                for traffic_sign_id in lanelet.stop_line.traffic_sign_ref:
                    if traffic_sign_id is not None:
                        traffic_sign = self.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
                        traffic_sign_position_xy = [traffic_sign.position[0], traffic_sign.position[1]]
                        try:
                            traffic_sign_position_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                                traffic_sign_position_xy[0], traffic_sign_position_xy[1])[0]
                        except:
                            msg_logger.error("PP position of traffic sign is out of projection domain")
                            traffic_sign_position_s = None
                        for traffic_sign_element in traffic_sign.traffic_sign_elements:

                            if stop_position_s is None and traffic_sign_position_s is not None:
                                stop_position_s = traffic_sign_position_s
                            elif stop_position_s is not None and traffic_sign_position_s is None:
                                traffic_sign_position_s = stop_position_s
                            elif stop_position_s is None and traffic_sign_position_s is None:
                                msg_logger.warning("PP traffic sign is out of projection domain")
                                continue

                            if traffic_sign_element.traffic_sign_element_id.name == 'YIELD':
                                self.yield_signs += [{'id': traffic_sign_id,
                                                      'type': 'YieldSign',
                                                      'position_s': traffic_sign_position_s,
                                                      'position_xy': traffic_sign_position_xy,
                                                      'stop_position_s': stop_position_s,
                                                      'stop_position_xy': stop_position_xy}]
                            if traffic_sign_element.traffic_sign_element_id.name == 'STOP':
                                self.stop_signs += [{'id': traffic_sign_id,
                                                     'type': 'StopSign',
                                                     'position_s': traffic_sign_position_s,
                                                     'position_xy': traffic_sign_position_xy,
                                                     'stop_position_s': stop_position_s,
                                                     'stop_position_xy': stop_position_xy}]

            if lanelet.stop_line.traffic_light_ref is not None:
                for traffic_light_id in lanelet.stop_line.traffic_light_ref:
                    if traffic_light_id is not None:
                        traffic_light = self.lanelet_network.find_traffic_light_by_id(traffic_light_id)
                        traffic_light_position_xy = [traffic_light.position[0], traffic_light.position[1]]
                        try:
                            traffic_light_position_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                                traffic_light.position[0], traffic_light.position[1])[0]
                        except:
                            msg_logger.warning('PP traffic light position out of projection domain')
                            traffic_light_position_s = None

                        if stop_position_s is None and traffic_light_position_s is not None:
                            stop_position_s = traffic_light_position_s
                        elif stop_position_s is not None and traffic_light_position_s is None:
                            traffic_light_position_s = stop_position_s
                        elif stop_position_s is None and traffic_light_position_s is None:
                            msg_logger.warning("PP traffic light is out of projection domain")
                            continue

                        if traffic_light.active:
                            self.traffic_lights += [{'id': traffic_light_id,
                                                     'type': 'TrafficLight',
                                                     'position_xy': traffic_light_position_xy,
                                                     'stop_position_xy': stop_position_xy,
                                                     'position_s': traffic_light_position_s,
                                                     'stop_position_s': stop_position_s}]

    def _look_for_lane_merges(self):
        self.lane_merges = []

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
                            merging_point_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[0][0], lanelet.center_vertices[0][1])[0]
                        except:
                            merging_point_s = None
                            msg_logger.error("PP merging point is out of projection domain")
                            continue
                        self.lane_merges += [{'type': 'LaneMerge',
                                              'position_xy': lanelet.center_vertices[0],
                                              'position_s': merging_point_s}]

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
                            start_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[0][0], lanelet.center_vertices[0][1])[0]
                        except:
                            start_s = None
                            msg_logger.error("PP start of intersection out of projection domain")
                        end_xy = lanelet.center_vertices[-1].tolist()
                        try:
                            end_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                                lanelet.center_vertices[-1][0], lanelet.center_vertices[-1][1])[0]
                        except:
                            end_s = None
                            msg_logger.error("PP end of intersection out of projection domain")

                        if start_s is None and end_s is not None:
                            start_s = max([0.001, end_s - 15])
                        elif start_s is not None and end_s is None:
                            end_s = start_s + 15
                        elif start_s is None and end_s is None:
                            msg_logger.warning("PP intersection is out of projection domain")
                            continue

                        self.intersections += [{'id': intersection_element.incoming_id,
                                                'type': 'Intersection',
                                                'start_xy': start_xy,
                                                'start_s': start_s,
                                                'end_xy': end_xy,
                                                'end_s': end_s}]

    def _look_for_road_exits(self):

        return self.road_exits

    def _look_for_turns(self):

        return self.turns

    def _straighten_static_route_plan(self):
        """Checks for overlapping static goals and straightens them out and fills gaps between goals with StaticDefault.
        """

        end_nav_path_s = self.cl_nav_coordinate_system.convert_to_curvilinear_coords(
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
            for i in range(len(self.static_route_plan)-1):
                # remove yield signs at active traffic lights
                if self.static_route_plan[i].start_s == self.static_route_plan[i+1].start_s:
                    if self.static_route_plan[i].goal_type == 'TrafficLight' and \
                            self.static_route_plan[i+1].goal_type == 'YieldSign':
                        self.static_route_plan = self.static_route_plan[:i+1] + self.static_route_plan[i+2:]
                    elif self.static_route_plan[i].goal_type == 'YieldSign' and \
                            self.static_route_plan[i+1].goal_type == 'TrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i+1:]
                    if self.static_route_plan[i].goal_type == 'PrepareTrafficLight' and \
                            self.static_route_plan[i+1].goal_type == 'PrepareYieldSign':
                        self.static_route_plan = self.static_route_plan[:i+1] + self.static_route_plan[i+2:]
                    elif self.static_route_plan[i].goal_type == 'PrepareYieldSign' and \
                            self.static_route_plan[i+1].goal_type == 'PrepareTrafficLight':
                        self.static_route_plan = self.static_route_plan[:i] + self.static_route_plan[i+1:]
                # no goals with s < 0
                if self.static_route_plan[i].start_s < 0:
                    self.static_route_plan[i].start_s = 0
                # cut overlapping goals
                if self.static_route_plan[i].end_s > self.static_route_plan[i+1].start_s:
                    if self.static_route_plan[i+1].goal_type[:7] == 'Prepare':
                        self.static_route_plan[i+1].start_s = self.static_route_plan[i].end_s
                    else:
                        self.static_route_plan[i].end_s = self.static_route_plan[i+1].start_s
                        self.static_route_plan[i].end_xy = self.static_route_plan[i+1].start_xy
                # fill with StaticDefault
                elif self.static_route_plan[i].end_s < self.static_route_plan[i+1].start_s:
                    goal = StaticGoal(goal_type='StaticDefault',
                                      start_s=self.static_route_plan[i].end_s,
                                      end_s=self.static_route_plan[i+1].start_s)
                    self.static_route_plan += [goal]

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
        return self.static_route_plan


class ReferencePath(object):
    """ Reference Path: object holding the reference path for the reactive planner. Creates straight base reference path
    with initialization."""
    def __init__(self, lanelet_network, global_nav_route, BM_state):

        self.BM_state = BM_state
        self.lanelet_network = lanelet_network

        self.cl_ref_coordinate_system = None
        self.reference_path = None
        self.list_ids_ref_path = None
        self._create_base_ref_path(global_nav_route)

    def _create_base_ref_path(self, global_nav_route):
        # create lanelet list for straight base route
        base_lanelet_ids = hf.create_consecutive_lanelet_id_list(self.lanelet_network,
                                                                 global_nav_route.list_ids_lanelets[0],
                                                                 self.BM_state.PP_state.route_plan_ids)
        # create empty list_portions
        base_list_portions = []
        for i in base_lanelet_ids:
            base_list_portions += [(0, 1)]
        # create base reference path
        base_ref_path = hf.compute_straight_reference_path(self.lanelet_network, base_lanelet_ids)
        self.list_ids_ref_path = base_lanelet_ids
        self.reference_path = base_ref_path
        # update curvilinear reference coordinate system
        self._update_cl_ref_coordinate_system()

    def _update_cl_ref_coordinate_system(self):
        self.cl_ref_coordinate_system = CoordinateSystem(reference=self.reference_path)

    def create_lane_change(self, ego_state, current_lanelet_id, goal_lanelet_id, number_vertices_lane_change=6):
        old_path = self.reference_path[:]
        # create straight reference path on goal lanelet
        new_path_ids = hf.create_consecutive_lanelet_id_list(self.lanelet_network, goal_lanelet_id,
                                                             self.BM_state.PP_state.route_plan_ids)
        old_ref_path_ids = self.list_ids_ref_path[:self.list_ids_ref_path.index(current_lanelet_id)+1]
        self.list_ids_ref_path = old_ref_path_ids + new_path_ids
        new_path = hf.compute_straight_reference_path(self.lanelet_network, new_path_ids)
        # cut old and new path at current position
        cut_idx_old = np.argmin([np.linalg.norm(x) for x in (np.subtract(old_path, ego_state.position))])
        cut_idx_new = np.argmin([np.linalg.norm(x) for x in (np.subtract(new_path, ego_state.position))])
        old_path = old_path[:cut_idx_old + self.BM_state.future_factor, :]
        new_path = new_path[self.BM_state.future_factor + cut_idx_new + number_vertices_lane_change:, :]
        # create final reference path
        reference_path = np.concatenate((old_path, new_path), axis=0)
        self.reference_path = smooth_ref_path(reference_path)


class StaticGoal(object):
    def __init__(self, goal_type, start_s=None, start_xy=None, end_s=None, end_xy=None, stop_point_s=None,
                 stop_point_xy=None, goal_object=None):

        self.goal_type = goal_type
        self.start_s = start_s
        self.start_xy = start_xy
        self.end_s = end_s
        self.end_xy = end_xy
        self.stop_point_s = stop_point_s
        self.stop_point_xy = stop_point_xy
        self.goal_object = goal_object
