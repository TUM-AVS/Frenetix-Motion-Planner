__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import logging
import numpy as np
import time
import warnings

# Suppress RuntimeWarning from shapely
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import nearest_points
from risk_assessment.helpers.collision_helper_function import angle_range

# get logger
msg_logger = logging.getLogger("Message_logger")


def calc_occluded_areas(ego_pos=None, visible_area=None, ref_path=None, lanelets=None, scope=50,
                        ref_path_buffer=10, cut_backwards=True):

    # Convert ego_pos to Point and calculate the sensor area (use entire given lanelets if scope is None)
    ego_pos_point = Point(ego_pos)
    if scope is not None:
        sensor_area_poly = ego_pos_point.buffer(scope)
    else:
        sensor_area_poly = lanelets

    # Find the closest node on the reference path in order to shorten it
    idx_closest_node_on_ref_path = closest_node(ego_pos, ref_path)
    ref_path_cut = ref_path[idx_closest_node_on_ref_path:, :]

    # Convert the ref_path numpy array to a Shapely Linestring and buffer it to get an area
    ref_path_cut_ls = LineString(ref_path_cut)
    ref_path_cut_poly = ref_path_cut_ls.buffer(ref_path_buffer)

    # Find the intersection between the lanelet polygon and the reference path
    lanelet_ref_path_cut_poly = lanelets.intersection(ref_path_cut_poly)

    # Find the relevant area --> lanelet along reference path and within the sensor radius
    relevant_area_poly = lanelet_ref_path_cut_poly.intersection(sensor_area_poly)

    # remove backward areas
    if cut_backwards:
        # Find the real closest point on the reference path (not the nearest node)
        nearest_point_on_ref_path = nearest_points(ref_path_cut_ls, ego_pos_point)[0]

        # convert shapely Point to numpy array
        nearest_point_on_ref_path_np = np.array(nearest_point_on_ref_path.coords).flatten()

        # calculate the normal and tangent vector to the reference path on the given point (to remove backwards area)
        normal_vector, tangent_vector = normal_vector_at_point(ref_path_cut_ls, nearest_point_on_ref_path_np)

        # calculate the corner points of the desired polygon
        corner_points_normal = np.vstack((nearest_point_on_ref_path_np + normal_vector * ref_path_buffer * 1.05,
                                          nearest_point_on_ref_path_np - normal_vector * ref_path_buffer * 1.05))
        corner_points_tangent = corner_points_normal - tangent_vector * ref_path_buffer * 1.05

        # Convert the points to a polygon and reorder the coordinates (for shapely)
        poly_points_to_cut_np = np.vstack((corner_points_normal, corner_points_tangent))
        poly_points_to_cut_np = poly_points_to_cut_np[[0, 1, 3, 2], :]
        poly_to_cut = Polygon(poly_points_to_cut_np)

        # remove backwards area from the relevant area
        relevant_area_poly = relevant_area_poly.difference(poly_to_cut)

    # Find the new visible area (within the relevant area)
    visible_area_in_relevant_area_poly = relevant_area_poly.intersection(visible_area)

    # Find the occluded areas in the relevant area --> difference between relevant area and visible area
    occluded_area_in_relevant_area_poly = relevant_area_poly.difference(visible_area_in_relevant_area_poly)

    return_plot = [sensor_area_poly, ref_path_cut_poly]

    return visible_area_in_relevant_area_poly, occluded_area_in_relevant_area_poly, return_plot


def np_replace_negative_with_zero(arr, column_index):
    arr[arr[:, column_index] < 0, column_index] = 0
    return arr


def np_replace_non_negative_with_value(arr, column_index, value):
    arr[arr[:, column_index] >= 0, column_index] = value
    return arr


def np_replace_max_value_with_new_value(arr, column_index, value_max, new_value):
    arr[arr[:, column_index] > value_max, column_index] = new_value
    return arr


def analyze_runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        msg_logger.debug(f'{func.__name__} took {end - start} seconds to run.')
        return result
    return wrapper


def normal_vector_at_point(line_ls, p_np):
    """
    Calculates the normal vector on a linestring at a given point
    Args:
        line_ls: shapely linestring
        p_np: numpy 1D array with x and y coordinates of the point

    Returns:
        normal vector on linestring at given point
        tangent vector on linestring at given point
    """

    # Check if point is within the bounds of the LineString
    if not line_ls.bounds[0] <= p_np[0] <= line_ls.bounds[2] or \
       not line_ls.bounds[1] <= p_np[1] <= line_ls.bounds[3]:
        raise ValueError("Point is not within the bounds of the LineString.")

    # Calculate the tangent point of the line at the nearest point
    tangent_point_np = np.array(line_ls.interpolate(line_ls.project(Point(p_np)) + 1e-9).coords).flatten()

    # Normalize the tangent, rotate it 90 degrees counterclockwise
    tangent_vector = unit_vector(tangent_point_np - p_np)

    # Calculate the normal vector
    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

    return normal_vector, tangent_vector


def unit_vector(vector):
    """Returns the unit vector of the vector.

    Args:
        vector: numppy array of vector
    """
    return vector / np.linalg.norm(vector)


def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def point_cloud(polygon, resolution=1):
    if not polygon.is_empty:
        x_min, y_min, x_max, y_max = polygon.bounds
        x_min = int(x_min // resolution)
        y_min = int(y_min // resolution)
        x_max = int(x_max // resolution)
        y_max = int(y_max // resolution)
        x_y_list = []
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
                x = i * resolution
                y = j * resolution
                p = Point(x, y)
                g = polygon.intersection(p)
                if g.is_empty:
                    continue
                x_y_hash = hash_x_y(x, y)
                x_y_list.append([x_y_hash, x, y])
        if len(x_y_list) == 0:
            return None
        else:
            x_y_np = np.array(x_y_list)
    else:
        x_y_np = None
    return x_y_np


def hash_x_y(value1, value2):
    # Combine the two values into a string
    combined_value = str(value1) + str(value2)
    # Use a hash function to create a unique value
    unique_value = hash(combined_value)
    return unique_value


def remove_unwanted_shapely_elements(polys) -> MultiPolygon:
    """
    This function removes every Geometry except Polygons from a GeometryCollection
    and converts the remaining Polygons to a MultiPolygon.

    Args:
        polys: GeometryCollection

    Returns: MultiPolygon

    """
    if polys.geom_type == "GeometryCollection":
        poly_list = []
        for pol in polys.geoms:
            if pol.geom_type == 'Polygon':
                poly_list.append(pol)

        multipolygon = convert_list_to_multipolygon(poly_list)
    else:
        multipolygon = polys

    return multipolygon.buffer(0)


def convert_list_to_multipolygon(poly_list) -> MultiPolygon:
    """
    This function converts a list of Polygons to a MultiPolygon

    Args:
        poly_list: List of Polygons

    Returns: MultiPolygon

    """
    polys = MultiPolygon(poly_list)

    return polys


def plot_polygons(ax, polys, color='b', zorder=1, opacity=1):
    """
    Helper function to plot polygons.
    Args:
        ax: axis
        polys: list of polygons, Polygon or MultiPolygon
        color: optional color
        zorder:
    """
    if ax is None:
        fig, ax = plt.subplots()

    ret_obj = None

    try:
        if type(polys) == list:
            for pol in polys:
                if pol.geom_type == 'Polygon':
                    ret_obj, = ax.plot(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        elif polys.geom_type == 'Polygon':
            ret_obj, = ax.plot(polys.exterior.xy[0], polys.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        elif polys.geom_type == 'MultiPolygon' or polys.geom_type == 'GeometryCollection':
            for pol in polys.geoms:
                if pol.geom_type == 'Polygon':
                    ret_obj, = ax.plot(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        return ret_obj
    except:
        msg_logger.warning('Could not plot the Polygon')


def fill_polygons(ax, polys, color='b', zorder=1, opacity=1):
    """
    Helper function to fill polygons.
    Args:
        ax: axis
        polys: list of polygons, Polygon or MultiPolygon
        color: optional color
        zorder:
        opacity:
    """
    try:
        if type(polys) == list:
            for pol in polys:
                if pol.geom_type == 'Polygon':
                    ret_obj = ax.fill(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        elif polys.geom_type == 'Polygon':
            ret_obj = ax.fill(polys.exterior.xy[0], polys.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        elif polys.geom_type == 'MultiPolygon' or polys.geom_type == 'GeometryCollection':
            for pol in polys.geoms:
                if pol.geom_type == 'Polygon':
                    ret_obj = ax.fill(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

        return ret_obj
    except:
        msg_logger.warning('Could not plot the Polygon')


def plot_occ_map(ax, occ_map, cmap, alpha=0.5):
    scatter = ax.scatter(occ_map[:, 1], occ_map[:, 2], c=occ_map[:, 3], cmap=cmap, s=10, alpha=alpha)
    #handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    #ax.legend(handles, labels, loc="upper right", title="Occlusion")


def normalize_costs_z(costs, max_costs=100) -> np.array:

    # if max of costs is 0 return costs
    if max(costs) == 0.0:
        return costs

    # calc z scores and normalize costs
    std_dev = np.std(costs)
    if std_dev == 0:
        z_scores = np.zeros_like(costs)
    else:
        z_scores = (costs - np.mean(costs)) / std_dev

    z_range = np.max(z_scores) - np.min(z_scores)
    if z_range == 0:
        norm_costs = np.zeros_like(z_scores)
    else:
        norm_costs = (z_scores - np.min(z_scores)) / z_range * max_costs

    # if all cost values are almost the same, the result will be nan --> return zeros
    if any(np.isnan(norm_costs)):
        return np.zeros(len(costs))

    return norm_costs


def normalize_costs_log(costs, max_costs=100):

    costs_log = np.log(costs)
    costs_log[np.isinf(costs_log)] = 0

    # Min-Max-normalization to 0 - max_costs
    costs_norm = (costs_log - np.min(costs_log)) / (np.max(costs_log) - np.min(costs_log)) * max_costs

    return costs_norm


def normalize_costs_iqr(costs, max_costs=10):
    # calculate 25 and 75 quantile
    q25 = np.quantile(costs, 0.25)
    q75 = np.quantile(costs, 0.75)

    # calculate inter quartile ratio
    iqr = q75-q25

    # normalize costs using iqr and scale it to 0-max_costs
    norm_costs = (costs - q25) / iqr
    norm_costs = np.clip(norm_costs, 0, 1)
    norm_costs *= max_costs

    return norm_costs


def find_max_s_in_trajectories(trajectories):
    max_s = max([s for traj in trajectories for s in traj.curvilinear.s])
    return max_s


def compute_vehicle_polygons(x_traj, y_traj, orientations, width, length) -> dict:
    """
    Computes the polygon of a vehicle at each time step, given its trajectory and orientation.

    Args:
    - x_traj: numpy array, x-coordinates of the vehicle's trajectory
    - y_traj: numpy array, y-coordinates of the vehicle's trajectory
    - orientations: numpy array, orientation of the vehicle at each time step (in radians)
    - width: float, width of the vehicle
    - length: float, length of the vehicle

    Returns:
    - polygons: list of shapely Polygon objects, the polygon of the vehicle at each time step
    """

    # Define the vertices of the vehicle's polygon
    vertices = np.array([[-length / 2, -width / 2],
                         [length / 2, -width / 2],
                         [length / 2, width / 2],
                         [-length / 2, width / 2]])

    # Initialize the list of polygons
    polygons = []

    # Iterate over the trajectory
    for i in range(len(x_traj)):
        # Compute the rotation matrix
        rot_matrix = np.array([[np.cos(orientations[i]), -np.sin(orientations[i])],
                               [np.sin(orientations[i]), np.cos(orientations[i])]])

        # Rotate the vertices
        rotated_vertices = np.dot(vertices, rot_matrix.T)

        # Translate the vertices
        translated_vertices = rotated_vertices + np.array([x_traj[i], y_traj[i]])

        # Create the polygon and add it to the list
        poly = Polygon(translated_vertices)
        polygons.append(poly)

    single_polygon = convert_list_to_multipolygon(polygons)

    polygon_return = {'polygons': polygons, 'convex_hull': single_polygon.buffer(0)}

    return polygon_return


def scale_array(arr, threshold) -> np.array:
    # scale array if threshold is exceeded
    max_value = max(arr)
    if max_value > threshold:

        scale_factor = threshold / np.max(arr)
        # scaled array
        scaled_arr = arr * scale_factor

        return scaled_arr
    else:
        return arr


def remove_small_areas(multipolygon, threshold=1):

    # list for cleaned polygons
    cleaned_polygons = []

    # iterate over polygons
    for polygon in multipolygon.geoms:
        # check if polygon area is greater than threshold
        if polygon.area >= threshold:
            # append polygon to list
            cleaned_polygons.append(polygon)

    # combine list of cleaned polygons to multipolygon
    cleaned_multipolygon = MultiPolygon(cleaned_polygons)

    return cleaned_multipolygon.buffer(0)


def calc_easy_covariance_vector(position_vector):
    n = len(position_vector)

    covariance_vector = []

    for i in range(n):
        covariance_matrix = np.array([[i * 0.0015, i * 0.0001], [i * 0.0001, i * 0.0015]])
        covariance_vector.append(covariance_matrix)

    return np.array(covariance_vector)

