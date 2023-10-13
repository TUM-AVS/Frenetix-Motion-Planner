__author__ = "Maximilian Geisslinger, Tobias Markus"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import sys


def point_in_rectangle(r1, r2, p):
    """Calculates weather a point p is in a recangle defined by two corner points r1 and r2.
    If true the position in relative coordinates between 0 and 2 in and 0 and 12 in y are returned.

    Arguments:
        r1 {[list]} -- [[x,y] point of rectancle corner]
        r2 {[list]} -- [[x,y] point of rectancle corner]
        p {[list]} -- [[x,y] point which is tested to be in that rectanle]

    Returns:
        [bool / tuple] -- [False if point is not in rectangle, else relative position]
    """
    bottom_left = [min(r1[0], r2[0]), min(r1[1], r2[1])]
    top_right = [max(r1[0], r2[0]), max(r1[1], r2[1])]

    if (
        p[0] > bottom_left[0]
        and p[0] < top_right[0]
        and p[1] > bottom_left[1]
        and p[1] < top_right[1]
    ):
        x = int((p[0] - bottom_left[0]) / (top_right[0] - bottom_left[0]) * 3)
        y = int((top_right[1] - p[1]) / (top_right[1] - bottom_left[1]) * 13)
        return (x, y)
    else:
        return False


def transform_trajectories(trajectories_list, now_point, theta):
    """Transform a list of trajectories by translation and rotation.

    Arguments:
        trajectories_list {[list]} -- [list of trajectories being transformed]
        now_point {[list]} -- [[x,y] translation]
        orientation {[float]} -- [rotation]
    Returns:

        [trans_traj_list] -- [tranformed trajectory list]
    """
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    trans_traj_list = []

    for tr in trajectories_list:
        tr_1 = tr - now_point
        tr_2 = np.matmul(tr_1, rot_mat)
        trans_traj_list.append(tr_2)

    return trans_traj_list


def transform_back(trajectory, translation, rotation):
    """Back transformation of a single trajectory

    Arguments:
        trajectory {[list]} -- [Trajectory points in x,y]
        translation {[list]} -- [[x,y] translation]
        rotation {[float]} -- [rotation]

    Returns:
        [type] -- [description]
    """
    rotation = -rotation
    translation = -translation
    rot_mat = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    trajectory[:, :2] = np.matmul(trajectory[:, :2], rot_mat)
    trajectory[:, :2] = trajectory[:, :2] - translation

    # Transform sigmas
    if trajectory.shape[1] > 2:
        sigma_x = 1 / (trajectory[:, 2] + sys.float_info.epsilon)
        sigma_y = 1 / (trajectory[:, 3] + sys.float_info.epsilon)
        rho = trajectory[:, 4]

        sigma_cov = np.array(
            [
                [sigma_x ** 2, rho * sigma_x * sigma_y],
                [rho * sigma_x * sigma_y, sigma_y ** 2],
            ]
        )

        # Swap axes to shape (50,2,2)
        sigma_cov = sigma_cov.swapaxes(0, 2)
        sigma_cov = sigma_cov.swapaxes(1, 2)

        # sigma_cov_trans = np.matmul(rot_mat.T, sigma_cov)
        # sigma_cov_trans = np.matmul(sigma_conv_trans, rot_mat)
        # print(np.linalg.eig(sigma_cov_trans[0,:,:]))

        sigma_cov_trans = rot_mat.T @ sigma_cov @ rot_mat

        return trajectory[:, :2], sigma_cov_trans

    else:
        return trajectory[:, :2]


def get_sigmas_from_covariance(sigma_conv_trans):
    sigma_x_post = np.sqrt(sigma_conv_trans[:, 0, 0])
    sigma_y_post = np.sqrt(sigma_conv_trans[:, 1, 1])
    rho_post = sigma_conv_trans[:, 0, 1] / (sigma_x_post * sigma_y_post)

    sigma_x_post = np.expand_dims(sigma_x_post, axis=1)
    sigma_y_post = np.expand_dims(sigma_y_post, axis=1)
    rho_post = np.expand_dims(rho_post, axis=1)

    return np.concatenate((1 / sigma_x_post, 1 / sigma_y_post, rho_post), axis=1)

    # fut_pred_post = np.concatenate((trajectory[:, :2], sigma_x_post, sigma_y_post, rho_post), axis=1)


def abs_to_rel_coord(curr_pos, curr_orient, abs_coord):
    """Transform absolute coordinate to car-relative coordinate.

    Args:
        curr_pos (Union[tuple, list, np_array]):    current global position of the car.
                                                    format: (x,y)
        curr_orient (float): current global orientation of the car in rad
        abs_coord (Union[tuple, list, np_array]):   absolute coordinate that has to be
                                                    transformed.
                                                    Coord-List also possible

    Returns:
        np_array: transformed car-relative coordinate in the form (x,y)
    """
    abs_coord = np.array(abs_coord)
    curr_pos = np.array(curr_pos)
    if abs_coord.ndim == 2:
        abs_coord = np.transpose(abs_coord)
        curr_pos = curr_pos.reshape(2, 1)
    rot_mat = np.array(
        [
            [np.cos(curr_orient), np.sin(curr_orient)],
            [-np.sin(curr_orient), np.cos(curr_orient)],
        ]
    )
    rel_coord = abs_coord - curr_pos
    rel_coord = np.matmul(rot_mat, rel_coord)
    if rel_coord.ndim == 2:
        rel_coord = np.transpose(rel_coord)
    return rel_coord


if __name__ == "__main__":

    trajectory = np.random.rand(2, 2)
    trajectory_transformed = transform_trajectories([trajectory], 10, 0.5)[0]
    trajectory_back = transform_back(trajectory_transformed, 10, 0.5)

    if trajectory.all() == trajectory_back.all():
        print("Test OK")
    else:
        raise ValueError
