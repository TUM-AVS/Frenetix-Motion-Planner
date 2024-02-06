__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import logging
import os
from datetime import datetime
import subprocess
import math
import zipfile
import numpy as np
from typing import Tuple
# third party
from scipy.spatial.kdtree import KDTree
import yaml
import pickle
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum
import commonroad_dc.pycrcc as pycrcc

# get logger
msg_logger = logging.getLogger("Message_logger")


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


def create_coll_object(trajectory, vehicle_params, ego_state):
    """Create a collision_object of the trajectory for collision checking with road boundary and with other vehicles."""

    if ego_state is None:
        time_step = 0
    else:
        time_step = ego_state.time_step

    collision_object_raw = create_tvobstacle_trajectory(
        traj_list=trajectory,
        box_length=vehicle_params.length / 2,
        box_width=vehicle_params.width / 2,
        start_time_step=time_step,
    )
    # if the preprocessing fails, use the raw trajectory
    collision_object, err = trajectory_preprocess_obb_sum(
        collision_object_raw
    )
    if err:
        collision_object = collision_object_raw

    return collision_object


def create_tvobstacle_trajectory(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    traj_list = traj_list.prediction.trajectory.state_list
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state.orientation, state.position[0], state.position[1])
        )
    return tv_obstacle


def create_tvobstacle(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state[2], state[0], state[1])
        )
    return tv_obstacle


def delete_folder(path):
    if os.path.exists(path):
        # shutil.rmtree(path)
        subpr_handle = subprocess.Popen("sudo rm -rf " + path, shell=True)
        wait = subpr_handle.wait()


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)


def createfolder_if_not_existent(inputpath):
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
        name_folder = inputpath.rsplit('/')[-1]
        msg_logger.debug("Create " + name_folder + " folder")


def create_time_in_date_folder(inputpath):
    # directory with time stamp to save csv
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
    if not os.path.exists(os.path.join(inputpath, date)):
        os.makedirs(os.path.join(inputpath, date), mode=0o777)
    os.makedirs(os.path.join(inputpath, date, time), mode=0o777)

    return os.path.join(inputpath, date, time)


def zip_log_files(inputpath):
    filePaths = []
    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(inputpath):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)

    # writing files to a zipfile
    zip_file = zipfile.ZipFile(inputpath + '.zip', 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)

    msg_logger.debug(inputpath + '.zip file is created successfully!')

    # Remove Log files
    # shutil.rmtree(inputpath)


def open_config_file(path: str):
    # Load config with the set of tuning parameters
    with open(path) as f:
        config_parameters_ = yaml.safe_load(f)
    return config_parameters_


def delete_empty_folders(path: str):

    folders = list(os.walk(path))[1:]

    for folder in folders:
        inner_folders = list(os.walk(folder[0]))[1:]
        for inner_folder in inner_folders:
            # folder example: ('FOLDER/3', [], ['file'])
            if not inner_folder[2]:
                os.rmdir(inner_folder[0])
        if not folder[2]:
            try:
                os.rmdir(folder[0])
            except:
                pass


def find_lanelet_by_position_and_orientation(lanelet_network, position, orientation):
    """Return the IDs of lanelets within a certain radius calculated from an initial state (position and orientation).

    Args:
        lanelet_network ([CommonRoad LaneletNetwork Object]): [description]
        position ([np.array]): [position of the vehicle to find lanelet for]
        orientation ([type]): [orientation of the vehicle for finding best lanelet]

    Returns:
        [int]: [list of matching lanelet ids]
    """
    # TODO: Shift this function to commonroad helpers
    lanelets = []
    initial_lanelets = lanelet_network.find_lanelet_by_position([position])[0]
    best_lanelet = initial_lanelets[0]
    radius = math.pi / 5.0  # ~0.63 rad = 36 degrees, determined empirically
    min_orient_diff = math.inf
    for lnlet in initial_lanelets:
        center_line = lanelet_network.find_lanelet_by_id(lnlet).center_vertices
        lanelet_orientation = calc_orientation_of_line(center_line[0], center_line[-1])
        orient_diff = orientation_diff(orientation, lanelet_orientation)

        if orient_diff < min_orient_diff:
            min_orient_diff = orient_diff
            best_lanelet = lnlet
            if orient_diff < radius:
                lanelets = [lnlet] + lanelets
        elif orient_diff < radius:
            lanelets.append(lnlet)

    if not lanelets:
        lanelets.append(best_lanelet)

    return lanelets


def calc_orientation_of_line(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the orientation of the line connecting two points (angle in radian, counter-clockwise defined).

    Args:
        point1 (np.ndarray): Starting point.
        point2 (np.ndarray): Ending point.

    Returns:
        float: Orientation in radians.

    """
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])


def orientation_diff(orientation_1: float, orientation_2: float) -> float:
    """
    Calculate the orientation difference between two orientations in radians.

    Args:
        orientation_1 (float): Orientation 1.
        orientation_2 (float): Orientation 2.

    Returns:
        float: Orientation difference in radians.

    """
    return math.pi - abs(abs(orientation_1 - orientation_2) - math.pi)


def load_pickle(path, name):
    file_name = os.path.join(path, name + ".pickle")
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


def calculate_angle_between_lines(line_1, line_2):
    # calculate angle between pairs of lines
    angle1 = math.atan2(line_1[0][1] - line_1[1][1], line_1[0][0] - line_1[1][0])
    angle2 = math.atan2(line_2[0][1] - line_2[1][1], line_2[0][0] - line_2[1][0])
    angle_degrees = (angle1-angle2) * 360 / (2*math.pi)
    if angle_degrees > 180:
        angle_degrees = angle_degrees-360
    return angle_degrees


def rotate_glob_loc(global_matrix, rot_angle, matrix=True):
    """
    helper function to rotate matrices from global to local coordinates (vehicle coordinates)

    Angle Convention:
    yaw = 0: local x-axis parallel to global y-axis
    yaw = -np.pi / 2: local x-axis parallel to global x-axis --> should result in np.eye(2)

    rot_mat: Rotation from global to local (x_local = np.matmul(rot_mat, x_global))
    """

    rot_mat = np.array(
        [
            [np.cos(rot_angle), np.sin(rot_angle)],
            [-np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )

    mat_temp = np.matmul(rot_mat, global_matrix)

    if matrix:
        return np.matmul(mat_temp, rot_mat.T)

    return mat_temp


def pi_range(yaw):
    """Clip yaw to (-pi, +pi]."""
    if yaw <= -np.pi:
        yaw += 2 * np.pi
        return yaw
    elif yaw > np.pi:
        yaw -= 2 * np.pi
        return yaw
    return yaw


def extend_reference_path_at_end(reference_path: np.ndarray, final_position: np.ndarray,
                                 additional_lenght_in_meters: float = 5.0,
                                 distance_threshold_in_meters: float = np.inf) -> Tuple[np.ndarray, bool]:
    """
    Adds additional points at the end along a line between first two points of reference path.
    Returns new reference path and success indicator

    Reasoning
    ----------
    Otherwise, an occuring edge-case would be that
    the rear axel of the vehicle is after the last point of the ref path, which makes
    Frenet-Localization problematic.

    If the closest point to the final position is the last point of the reference path and their
    distance is below a certain threshold (currently infinity), additional points are being placed

    Credits to: Tobias Mascetta, TUM, Cyber-physical Systems Chair

    """

    # TODO: tmasc --> if there are now significant changes coming, use one class method for both start and end

    # check if the closest point to the final position is last point
    nn_distance, nn_idx = KDTree(reference_path).query(final_position)

    if (nn_idx == reference_path.shape[0] - 1 and nn_distance < distance_threshold_in_meters):
        # get distance between first two points to know what the pseudo-uniform sampling would be
        point_0: np.ndarray = reference_path[-2, :]
        point_1: np.ndarray = reference_path[-1, :]
        distance: float = np.linalg.norm(point_1 - point_0)
        num_new_points: int = math.ceil(additional_lenght_in_meters / distance)

        delta_x: float = float(point_1[0] - point_0[0])
        delta_y: float = float(point_1[1] - point_0[1])

        for idx in range(1, num_new_points + 1):
            new_point: np.ndarray = np.asarray([point_1[0] + idx * delta_x, point_1[1] + idx * delta_y])
            reference_path: np.ndarray = np.vstack((reference_path, new_point))

        return reference_path, True
    else:
        return reference_path, False

# EOF
