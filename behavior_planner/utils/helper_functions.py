__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
from shapely.geometry import LineString, Point

# commonroad imports
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry

from commonroad_route_planner.utility.route import chaikins_corner_cutting, resample_polyline


def get_remaining_path(ego_state, ref_path):
    point_, index = find_nearest_point_to_path(ref_path, ego_state.position)
    ref_path = ref_path[index:]
    return ref_path


def sort_by_distance(ego_state, objects):
    obj_sorted = sorted(objects, key=lambda x: distance(ego_state.position, x.initial_state.position))
    return obj_sorted


def calc_remaining_time_steps(
    ego_state_time: float, t: float, planning_problem, dt: float
):
    """
    Get the minimum and maximum amount of remaining time steps.

    Args:
        ego_state_time (float): Current time of the state of the ego vehicle.
        t (float): Checked time.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.

    Returns:
        int: Minimum remaining time steps.
        int: Maximum remaining time steps.
    """
    considered_time_step = int(ego_state_time + t / dt)
    if hasattr(planning_problem.goal.state_list[0], "time_step"):
        min_remaining_time = (
            planning_problem.goal.state_list[0].time_step.start - considered_time_step
        )
        max_remaining_time = (
            planning_problem.goal.state_list[0].time_step.end - considered_time_step
        )
        return min_remaining_time, max_remaining_time
    else:
        return False


def distance(pos1: np.array, pos2: np.array):
    """
    Return the euclidean distance between 2 points.

    Args:
        pos1 (np.array): First point.
        pos2 (np.array): Second point.

    Returns:
        float: Distance between point 1 and point 2.
    """
    return np.linalg.norm(pos1 - pos2)


def dist_to_nearest_point(center_vertices: np.ndarray, pos: np.array):
    """
    Find the closest distance of a given position to a polyline.

    Args:
        center_vertices (np.ndarray): Considered polyline.
        pos (np.array): Considered position.

    Returns:
        float: Closest distance between the polyline and the position.
    """
    # create a point and a line, project the point on the line and find the nearest point
    # shapely used
    point = Point(pos)
    linestring = LineString(center_vertices)
    project = linestring.project(point)
    nearest_point = linestring.interpolate(project)

    return distance(pos, np.array([nearest_point.x, nearest_point.y])), [nearest_point.x, nearest_point.y]


def find_nearest_point_to_path(center_vertices: np.ndarray, pos: np.array):
    dists = []
    for idx, path in enumerate(center_vertices):
        dists.append(distance(center_vertices[idx], pos))
    index = np.where(dists == min(dists))[0][0]
    return center_vertices[index], index


def get_lanelet_information(scenario, reference_path_ids, ego_state, country: SupportedTrafficSignCountry):
    """Get the current lanelet id, the legal speed limit and the street setting from the CommonRoad scenario

    Args:
        scenario (Scenario): scenario.
        ego_state: ego state
        country (SupportedTrafficSignCountry): country e.g: SupportedTrafficSignCountry.GERMANY)

    Returns:
        int: current_lanelet_id, speed_limit
        str: street setting ('Highway', 'Country', 'Urban)
    """
    traffic_signs = TrafficSigInterpreter(country=country, lanelet_network=scenario.lanelet_network)

    current_position = ego_state.position
    current_lanelets = scenario.lanelet_network.find_lanelet_by_position([current_position])[0]
    current_lanelet = None

    # get legal speed limit
    speed_limit = traffic_signs.speed_limit(lanelet_ids=frozenset(current_lanelets))

    if len(current_lanelets) == 1:
        current_lanelet = current_lanelets[0]
    elif len(current_lanelets) > 1:
        for lanelet_id in current_lanelets:
            if lanelet_id in reference_path_ids:
                current_lanelet = lanelet_id
            else:
                current_lanelet = current_lanelets[0]
    else:
        print("no lanelets detected")


    # check for street setting
    lanelet_type = scenario.lanelet_network.find_lanelet_by_id(current_lanelet).lanelet_type
    if LaneletType('highway') in lanelet_type:
        street_setting = 'Highway'
    elif LaneletType('interstate') in lanelet_type:
        street_setting = 'Highway'
    elif LaneletType('country') in lanelet_type:
        street_setting = 'Country'
    elif LaneletType('urban') in lanelet_type:
        street_setting = 'Urban'
    else:  # default is urban
        street_setting = 'Urban'

    if scenario.scenario_id.map_name == 'US101':  # only highway scenarios
        street_setting = 'Highway'

    return current_lanelet, speed_limit, street_setting


def find_country_traffic_sign_id(scenario):
    """finds corresponding SupportedTrafficSignCountry as defined in scenario

    Args:
        scenario (Scenario): scenario.

    Returns:
        SupportedTrafficSignCountry
    """

    country_id = scenario.scenario_id.country_id

    if country_id == 'DEU':
        return SupportedTrafficSignCountry.GERMANY
    elif country_id == 'USA':
        return SupportedTrafficSignCountry.USA
    elif country_id == 'CHN':
        return SupportedTrafficSignCountry.CHINA
    elif country_id == 'ESP':
        return SupportedTrafficSignCountry.SPAIN
    elif country_id == 'RUS':
        return SupportedTrafficSignCountry.RUSSIA
    elif country_id == 'ARG':
        return SupportedTrafficSignCountry.ARGENTINA
    elif country_id == 'BEL':
        return SupportedTrafficSignCountry.BELGIUM
    elif country_id == 'FRA':
        return SupportedTrafficSignCountry.FRANCE
    elif country_id == 'GRC':
        return SupportedTrafficSignCountry.GREECE
    elif country_id == 'HRV':
        return SupportedTrafficSignCountry.CROATIA
    elif country_id == 'ITA':
        return SupportedTrafficSignCountry.ITALY
    elif country_id == 'PRI':
        return SupportedTrafficSignCountry.PUERTO_RICO
    elif country_id == 'ZAM':
        return SupportedTrafficSignCountry.ZAMUNDA
    else:
        return None


def get_closest_preceding_obstacle(predictions, lanelet_network, coordinate_system, lanelet_id, ego_position_s,
                                   ego_state):
    obstacles_on_lanelet = get_predicted_obstacles_on_lanelet(predictions, lanelet_network, lanelet_id,
                                                              search_by_shape=False)
    closest_obstacle = None
    closest_obstacle_pos_s = None
    for obstacle_id in obstacles_on_lanelet:
        obstacle = obstacles_on_lanelet.get(obstacle_id)
        obstacle_position_xy = obstacle.get('pos_list')[0]
        try:
            obstacle_position_s = coordinate_system.convert_to_curvilinear_coords(obstacle_position_xy[0],
                                                                                  obstacle_position_xy[1])[0]
        except:
            print("VP object out of projection domain. Object position: ", obstacle_position_xy)
            continue

        if obstacle_position_s > ego_position_s:
            if closest_obstacle is None:
                closest_obstacle = obstacle
                closest_obstacle_pos_s = obstacle_position_s
            else:
                if obstacle_position_s < closest_obstacle_pos_s:
                    closest_obstacle = obstacle
                    closest_obstacle_pos_s = obstacle_position_s

    if closest_obstacle is not None:
        dist_preceding_veh = distance(ego_state.position, closest_obstacle.get('pos_list')[0])
        vel_preceding_veh = closest_obstacle.get('v_list')[0]
    else:
        dist_preceding_veh = None
        vel_preceding_veh = None

    return closest_obstacle, dist_preceding_veh, vel_preceding_veh


def get_predicted_obstacles_on_lanelet(predictions, lanelet_network, lanelet_id, search_point=None,
                                       search_distance=None, search_by_shape=False):  # add config search by shape
    lanelet_ids = create_consecutive_lanelet_id_list(lanelet_network, lanelet_id)
    obstacles_on_lanelet = dict()
    for obstacle_id in predictions:
        obstacle_position = predictions.get(obstacle_id).get('pos_list')[0]
        if search_by_shape:
            obstacle_orientation = predictions.get(obstacle_id).get('orientation_list')[0]
            obstacle_length = predictions.get(obstacle_id).get('shape').get('length')
            obstacle_width = predictions.get(obstacle_id).get('shape').get('width')
            obstacle_shape = Rectangle(length=obstacle_length*2, width=obstacle_width*0.2, center=obstacle_position,
                                       orientation=obstacle_orientation)
            obstacle_lanelet_ids = lanelet_network.find_lanelet_by_shape(obstacle_shape)
        else:
            obstacle_lanelet_ids = lanelet_network.find_lanelet_by_position([obstacle_position])[0]
        if search_point is None:
            for obstacle_lanelet_id in obstacle_lanelet_ids:
                if obstacle_lanelet_id in lanelet_ids:
                    obstacles_on_lanelet[obstacle_id] = predictions.get(obstacle_id)
        else:
            for obstacle_lanelet_id in obstacle_lanelet_ids:
                if obstacle_lanelet_id in lanelet_ids:
                    obstacle_distance = distance(search_point, obstacle_position)
                    if obstacle_distance <= search_distance:
                        obstacles_on_lanelet[obstacle_id] = predictions.get(obstacle_id)

    return obstacles_on_lanelet


def create_consecutive_lanelet_id_list(lanelet_network, start_lanelet_id, navigation_route_ids=None):
    consecutive_lanelet_ids = [start_lanelet_id]
    # predecessors
    end = False
    while not end:
        lanelet = lanelet_network.find_lanelet_by_id(consecutive_lanelet_ids[0])
        if lanelet.predecessor:
            if len(lanelet.predecessor) == 1:
                consecutive_lanelet_ids = lanelet.predecessor + consecutive_lanelet_ids
            else:
                consecutive_lanelet_ids = [lanelet.predecessor[0]] + consecutive_lanelet_ids
        else:
            end = True
    # successors
    end = False
    while not end:
        lanelet = lanelet_network.find_lanelet_by_id(consecutive_lanelet_ids[-1])
        if lanelet.successor:
            if len(lanelet.successor) == 1:
                consecutive_lanelet_ids += lanelet.successor
            else:
                if navigation_route_ids is not None:
                    for successor_id in lanelet.successor:
                        if successor_id in navigation_route_ids:
                            consecutive_lanelet_ids += [successor_id]
                else:
                    consecutive_lanelet_ids += [lanelet.successor[0]]
        else:
            end = True
    return consecutive_lanelet_ids


def retrieve_glb_nav_path_lane_changes(route):
    lane_changes = route._compute_lane_change_instructions()
    return lane_changes


def compute_straight_reference_path(lanelet_network, list_ids_lanelets):
    """Computes reference path given the list of portions of each lanelet

    lanelet_network (LaneletNetwork): lanelet_network of scenario
    list_ids_lanelets (List(int)): list of lanelet ids
    """
    reference_path = None
    for lanelet_id in list_ids_lanelets:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        if reference_path is None:
            reference_path = lanelet.center_vertices
        else:
            reference_path = np.concatenate((reference_path, lanelet.center_vertices), axis=0)

    reference_path = resample_polyline(reference_path, 1)
    return reference_path


def smooth_reference_path(reference_path):
    reference_path_resampled = resample_polyline(reference_path, 2)
    reference_path_smooth = chaikins_corner_cutting(reference_path_resampled)
    return reference_path_smooth
