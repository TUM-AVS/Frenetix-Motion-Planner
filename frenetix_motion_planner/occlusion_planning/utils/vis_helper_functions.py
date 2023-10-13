__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import logging
import numpy as np
from shapely.geometry import Polygon, Point
import math

# get logger
msg_logger = logging.getLogger("Message_logger")


def calc_visible_area_from_lanelet_geometry(lanelet_polygon, ego_pos, sensor_radius=50, buffer=0.0):
    """
    Calculates the visible area at the ego position in consideration of the lanelet network geometry
    Args:
        lanelet_polygon: lanelet network as combined polygon (wall buffer shall be included here)
        ego_pos: ego position
        sensor_radius: sensor radius, if no sensor radius is given the entire lanelet will be used
        buffer: optional wall buffer - better include the wall buffer in the lanelet polygon

    Returns:
    visible area
    """

    # do not use buffer, buffer shall be included in lanelet_polygon
    if buffer != 0.0:
        msg_logger.debug('not use buffer in calculation of visible area - '
              'buffer shall be included in the lanelets!')

    # load lanelet network polygon from class and cut it to sensor radius
    if sensor_radius is not None:
        visible_area = lanelet_polygon.intersection(Point(ego_pos).buffer(sensor_radius))
    else:
        visible_area = lanelet_polygon

    # buffer if necessary
    visible_area = visible_area.buffer(buffer)

    # Subtract areas that can not be seen due to geometry
    if visible_area.geom_type == 'MultiPolygon':
        points_vis_area = np.concatenate([np.array(p.exterior.xy).T for p in visible_area.geoms])
    else:
        points_vis_area = np.array(visible_area.exterior.xy).T

    # find points of vertices and create points far away
    for idx in range(points_vis_area.shape[0] - 1):
        vert_point1 = points_vis_area[idx]
        vert_point2 = points_vis_area[idx + 1]

        # create polygon from vertices
        pol = create_polygon_from_vertices(vert_point1, vert_point2, ego_pos)

        # subtract polygon from visible area if it is valid
        if pol.is_valid:
            area_check = visible_area.difference(pol)
            # shapely has a bug, that visible area can be empty after performing .difference for no reason
            if area_check.is_empty:
                # a very small buffer fixes that
                visible_area = visible_area.buffer(0.0001).difference(pol)
            else:
                visible_area = area_check

    return visible_area


def calc_visible_area_from_obstacle_occlusions(visible_area, ego_pos, obstacles, sensor_radius,
                                               return_shapely_obstacles=False):
    """
    Calculate occlusions from obstacles and subtract them from visible_area
    Args:
        return_shapely_obstacles: specify whether to return a multipolygon of the obstacles or not
        visible_area: visible area
        ego_pos: ego position
        obstacles: list of obstacles of type OcclusionObstacle or EstimationObstacle
        sensor_radius: sensor radius

    Returns:
    updated visible area
    optional - multipolygon of obstacles
    """

    obstacles_polygon = Polygon([])

    # Calculate occlusions from obstacles and subtract them from visible_area
    for obst in obstacles:

        # obstacle position is not empty, this happens if dynamic obstacle is not available at timestep
        if obst.pos is not None:

            # check if within sensor radius or if obstacle intersects with visible area
            if obst.pos_point.within(visible_area) or obst.polygon.intersects(visible_area):
                # calculate occlusion polygon that is caused by the obstacle
                occlusion, c1, c2 = get_polygon_from_obstacle_occlusion(ego_pos, obst.corner_points,
                                                                        sensor_radius=sensor_radius)

                # Subtract obstacle shape from visible area
                visible_area = visible_area.difference(obst.polygon.buffer(0.005, join_style=2))
                obstacles_polygon = obstacles_polygon.union(obst.polygon)

                # Subtract occlusion caused by obstacle (everything behind obstacle) from visible area
                if occlusion.is_valid:
                    visible_area = visible_area.difference(occlusion)

    if return_shapely_obstacles:
        return visible_area, obstacles_polygon

    return visible_area


def calc_corner_points(pos, orientation, obstacle_shape):
    """Calculate corner points of a dynamic obstacles in global coordinate system.

    Arguments:
        pos:  position of the object (center position) in global coordinate system [x,y] - np.array
        orientation: orientation of the object in rad - float
        obstacle_shape: shape of the object [width and length]

    Returns:
        corner points of object
    """
    corner_points = _rotate_point_by_angle(obstacle_shape.vertices[0:4], orientation)
    corner_points = [p + pos for p in corner_points]
    return np.array(corner_points)


def create_polygon_from_vertices(vert_point1, vert_point2, ego_pos):
    """Creates a polygon for the area that is occluded from two vertice points.

    Arguments:
        vert_point1 {[list]} -- [x,y of first point of object]
        vert_point2 {[list]} -- [x,y of second point of object]
        ego_pos {[list]} -- [x,y of ego position]

    Returns:
        pol [Shapely polygon] -- [Represents the occluded area]
    """

    pol_point1 = vert_point1 + 100 * (vert_point1 - ego_pos)
    pol_point2 = vert_point2 + 100 * (vert_point2 - ego_pos)

    pol = Polygon([vert_point1, vert_point2, pol_point2, pol_point1, vert_point1])

    return pol


def get_polygon_from_obstacle_occlusion(ego_pos, corner_points, sensor_radius=50):

    # Identify points for geometric projection
    c1, c2 = _identify_projection_points(corner_points, ego_pos)

    # Create polygon with points far away in the ray direction of ego pos
    c3 = c2 + _unit_vector(c2 - ego_pos) * sensor_radius * 1.1
    c4 = c1 + _unit_vector(c1 - ego_pos) * sensor_radius * 1.1

    occlusion = Polygon([c1, c2, c3, c4])
    return occlusion, c1, c2


def _identify_projection_points(corner_points, ego_pos):
    """This function identifies the two points of a rectangular object that are the edges from an ego pos point of view.

    Arguments:
        corner_points {[type]} -- [description]
        ego_pos {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    max_angle = 0

    for edge_point1 in corner_points:
        for edge_point2 in corner_points:
            ray1 = edge_point1 - ego_pos
            ray2 = edge_point2 - ego_pos

            angle = angle_between(ray1, ray2)

            if angle > max_angle:
                max_angle = angle
                ret_edge_point1 = edge_point1
                ret_edge_point2 = edge_point2

    return ret_edge_point1, ret_edge_point2


def _rotate_point_by_angle(point, angle):
    """Rotate any point by an angle.

    Arguments:
        point:
        angle:
        point {[type]} -- [description]
        angle {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(rotation_matrix, point.transpose()).transpose()


def _unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2':"""
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_positive(v1, v2):
    """Returns the positive angle (mathematically positive) in radians between vectors 'v1' and 'v2':"""
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    angle_rad = math.atan2(np.linalg.det([v1_u, v2_u]), dot_product)
    if angle_rad < 0:
        angle_rad += 2 * math.pi
    return angle_rad

