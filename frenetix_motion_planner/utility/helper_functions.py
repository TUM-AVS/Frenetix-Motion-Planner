__author__ = "Maximilian Geisslinger, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os
from datetime import datetime
import subprocess
import zipfile
import logging
import math
import ruamel.yaml as yaml
import pickle
import yaml as yml
import matplotlib.colors as colors
import numpy as np
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_object,
)
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum
from commonroad_dc.pycrcc import ShapeGroup
import commonroad_dc.pycrcc as pycrcc
from shapely.geometry import Point
from commonroad.geometry.shape import Polygon

# get logger
msg_logger = logging.getLogger("Message_logger")


def calculate_desired_velocity(scenario, planning_problem, x_0, DT, desired_velocity=None) -> float:

    in_goal = False

    if desired_velocity is None:
        if hasattr(planning_problem.goal.state_list[0], 'velocity'):
            desired_velocity = (planning_problem.goal.state_list[0].velocity.start + planning_problem.goal.state_list[
                0].velocity.end) / 2
        else:
            desired_velocity = x_0.velocity + 5
    try:
        # if the goal is not reached yet, try to reach it
        # get the center points of the possible goal positions
        goal_centers = []
        # get the goal lanelet ids if they are given directly in the planning problem
        if (
                hasattr(planning_problem.goal, "lanelets_of_goal_position")
                and planning_problem.goal.lanelets_of_goal_position is not None
        ):
            in_goal = Point(x_0.position).within(scenario.lanelet_network.find_lanelet_by_id(planning_problem.goal.
                                                                                   lanelets_of_goal_position[0][0]).
                                                                                    polygon.shapely_object)
            goal_lanelet_ids = planning_problem.goal.lanelets_of_goal_position[0]
            for lanelet_id in goal_lanelet_ids:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                n_center_vertices = len(lanelet.center_vertices)
                goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
        elif hasattr(planning_problem.goal.state_list[0], "position"):
            if isinstance(planning_problem.goal.state_list[0], Polygon):
                in_goal = Point(x_0.position).within(planning_problem.goal.state_list[0].shapely_object)

            # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
            if hasattr(planning_problem.goal.state_list[0].position, "center"):
                goal_centers.append(planning_problem.goal.state_list[0].position.center)
        # if it is a survival scenario with no goal areas, no velocity can be proposed
        elif hasattr(planning_problem.goal.state_list[0], "time_step"):
            if x_0.time_step > planning_problem.goal.state_list[0].time_step.end:
                return 0.0
            else:
                return x_0.velocity
        else:
            return 0.0

        distances = []
        for goal_center in goal_centers:
            distances.append(distance(goal_center, x_0.position))

        # calculate the average distance to the goal positions
        avg_dist = np.mean(distances)

        _, max_remaining_time_steps = calc_remaining_time_steps(
            planning_problem=planning_problem,
            ego_state_time=x_0.time_step,
            t=0.0,
            dt=DT,
        )
        remaining_time = max_remaining_time_steps * DT

        # if there is time remaining, calculate the difference between the average desired velocity and the velocity of the trajectory
        if remaining_time > 0.0:
            desired_velocity_new = avg_dist / remaining_time
        else:
            desired_velocity_new = planning_problem.goal.state_list[0].velocity.end

    except:
        msg_logger.info("Could not calculate desired velocity")
        desired_velocity_new = desired_velocity

    if in_goal and hasattr(planning_problem.goal.state_list[0], 'velocity'):
        desired_velocity_new = (max(planning_problem.goal.state_list[0].velocity.start, 0.01) + planning_problem.goal.state_list[
                0].velocity.end) / 2

    if np.abs(desired_velocity - desired_velocity_new) > 5 or np.abs(x_0.velocity - desired_velocity_new) > 5:
        if np.abs(x_0.velocity - desired_velocity_new) > 5:
            desired_velocity = x_0.velocity + 1
        if desired_velocity_new > desired_velocity:
            desired_velocity_new = desired_velocity + 2
        else:
            desired_velocity_new = desired_velocity - 2

    return desired_velocity_new


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
        config_parameters_ = yml.load(f, Loader=yaml.RoundTripLoader)
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


def get_goal_area_shape_group(planning_problem, scenario):
    """
    Return a shape group that represents the goal area.

    Args:
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.

    Returns:
        ShapeGroup: Shape group representing the goal area.
    """
    # get goal area collision object
    # the goal area is either given as lanelets
    if (
        hasattr(planning_problem.goal, "lanelets_of_goal_position")
        and planning_problem.goal.lanelets_of_goal_position is not None
    ):
        # get the polygons of every lanelet
        lanelets = []
        for lanelet_id in planning_problem.goal.lanelets_of_goal_position[0]:
            lanelets.append(
                scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                ).convert_to_polygon()
            )

        # create a collision object from these polygons
        goal_area_polygons = create_collision_object(lanelets)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or the goal area is given as positions
    elif hasattr(planning_problem.goal.state_list[0], "position"):
        # get the polygons of every goal area
        goal_areas = []
        for goal_state in planning_problem.goal.state_list:
            goal_areas.append(goal_state.position)

        # create a collision object for these polygons
        goal_area_polygons = create_collision_object(goal_areas)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or it is a survival scenario
    else:
        goal_area_co = None

    return goal_area_co


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


def ignore_vehicles_in_cone_angle(predictions, ego_pose, veh_length, cone_angle, cone_safety_dist):
    """Ignore vehicles behind ego for prediction if inside specific cone.

    Cone is spaned from center of rear-axle (cog - length / 2.0)

    cone_angle = Totel Angle of Cone. 0.5 per side (right, left)

    return bool: True if vehicle is ignored, i.e. inside cone
    """

    ego_pose = np.array([ego_pose.initial_state.position[0], ego_pose.initial_state.position[1], ego_pose.initial_state.orientation])
    cone_angle = cone_angle / 180 * np.pi
    ignore_pred_list = list()

    for i in predictions:
        ignore_object = True
        obj_pose = np.array(
            [predictions[i]['pos_list'][0][0],
            predictions[i]['pos_list'][0][1], predictions[i]['orientation_list'][0]]
        )

        # Function not necessary since we already have global coordinates
        # obj_pose[:2] += rotate_loc_glob(
        #     np.array([veh_length / 2.0, 0.0]), obj_pose[2], matrix=False
        # )

        loc_obj_pos = rotate_glob_loc(
            obj_pose[:2] - ego_pose[:2], ego_pose[2], matrix=False
        )
        loc_obj_pos[0] += veh_length / 2.0

        if loc_obj_pos[0] > -cone_safety_dist:
            ignore_object = False

        obj_angle = pi_range(math.atan2(loc_obj_pos[1], loc_obj_pos[0]) - np.pi)

        if abs(obj_angle) > cone_angle / 2.0:
            ignore_object = False
        if ignore_object:
            ignore_pred_list.append(i)

    if len(ignore_pred_list) > 0:
        for obj in range(len(ignore_pred_list)):
            del predictions[ignore_pred_list[obj]]

    return predictions


def pi_range(yaw):
    """Clip yaw to (-pi, +pi]."""
    if yaw <= -np.pi:
        yaw += 2 * np.pi
        return yaw
    elif yaw > np.pi:
        yaw -= 2 * np.pi
        return yaw
    return yaw


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


def green_to_red_colormap():
    """Define a colormap that fades from green to red."""
    # This dictionary defines the colormap
    cdict = {
        "red": (
            (0.0, 0.0, 0.0),  # no red at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.8, 0.8),
        ),  # set to 0.8 so its not too bright at 1
        "green": (
            (0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no green at 1
        "blue": (
            (0.0, 0.0, 0.0),  # no blue at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no blue at 1
    }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap("GnRd", cdict)

    return GnRd


def extend_points(points):
    """Extend the list of points with additional points in the orientation of the line between the two first points."""
    p1, p2 = points[0], points[1]
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    dist = distance(points[0], points[1])
    num_new_points = int(5/dist)

    new_points = []
    for i in range(1, num_new_points + 1):
        new_point = (p1[0] - i * delta_x, p1[1] - i * delta_y)
        new_points.append(new_point)

    # Stack new_points and points and convert them to a numpy array
    return np.vstack((new_points[::-1], points))


def extend_rep_path(ref_path, init_pos):
    """This function is needed due to the fact that we have to shift the planning position of the reactive planner
    to the rear axis. In some scenatios the ref path is not long enough to locate the initial position in curv state"""
    close_point = min(ref_path, key=lambda point: distance(init_pos, point))
    if close_point[0] == ref_path[0, 0] and close_point[1] == ref_path[0, 1]:
        ref_path = extend_points(ref_path)
    return ref_path

# EOF
