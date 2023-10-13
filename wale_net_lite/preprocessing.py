__author__ = "Maximilian Geisslinger, Tobias Markus"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import time

# Third party imports
import numpy as np
# import matplotlib.pyplot as plt

# Custom imports
from .geometry import point_in_rectangle, abs_to_rel_coord


def generate_self_rendered_sc_img(
    watch_radius, scenario, curr_pos, curr_orient, res=256, light_lane_dividers=True
):
    """Render scene image in relative position."""
    # region generate_self_rendered_sc_img()
    # region read inputs
    lane_net = scenario.lanelet_network
    pixel_dist = 2 * watch_radius / res
    interp_factor = 0.8
    # endregion
    timer = time.time()

    # region read all lanelet boundarys into a list
    # IDEA speed up scene rendering even more multiprocessing

    bound_list = []
    type_list = []
    for lanelet in lane_net.lanelets:
        bound_list.append(lanelet.left_vertices)
        line_type = "road-boundary" if lanelet.adj_left is None else "lane-marking"
        type_list.append(line_type)
        bound_list.append(lanelet.right_vertices)
        line_type = "road-boundary" if lanelet.adj_right is None else "lane-marking"
        type_list.append(line_type)
    # endregion

    # region translate rotate image
    bound_list = [
        abs_to_rel_coord(curr_pos, curr_orient, bound_line) for bound_line in bound_list
    ]
    # endregion
    # print(f"Time for reading points:{time.time() - timer}")
    # timer = time.time()

    # region limit_boundarys to watch_radius
    # region limit_boundary_subfunction
    def limit_boundary(boundary):
        array = np.empty(len(boundary))
        last_point_was_out = None  # This line makes the linter happy
        # Loop over all points
        for index, point in enumerate(boundary):
            # Check index to avoid indexerrors
            if index > 0:
                # check if point is outside of viewing window
                point_is_out = bool(max(abs(point)) > watch_radius)
                # If point is inside
                if point_is_out is False:
                    array[index] = False
                    # Add the neighbour, so that a continous line
                    # to the image border can be rendered
                    if last_point_was_out is True:
                        array[index - 1] = False
                # if point is outside of watch_radius
                else:
                    # Add this point as neighbor if the last point was in
                    if last_point_was_out is False:
                        array[index] = False
                    # Remove point from boundary line
                    else:
                        array[index] = True
            else:
                # Handling of first element
                point_is_out = bool(max(abs(point)) > watch_radius)
                array[index] = point_is_out
            last_point_was_out = point_is_out
        return array

    # endregion
    # Call the function
    limit_bound_list = [
        np.delete(bound, limit_boundary(bound).astype(bool), axis=0)
        for bound in bound_list
    ]
    # endregion

    # print(f"Time for limiting array:{time.time() - timer}")
    # timer = time.time()
    # region Interpolate boundary lines

    # region interpolate_boundary() subfunction
    def interpolate_boundary(boundary):
        # region calc curve length of boundary
        curve_length = np.zeros(len(boundary), dtype=np.uint8)
        bound_array = np.array(boundary)
        for index, point in enumerate(bound_array[1:], start=1):
            curve_length[index] = curve_length[index - 1] + np.linalg.norm(
                point - boundary[index - 1]
            )
        # endregion
        # region interpolate over curve_length
        if len(curve_length) > 0:
            eval_array = np.arange(0, curve_length[-1], pixel_dist * interp_factor)
            rv = np.array(
                [
                    np.interp(eval_array, curve_length, bound_array.transpose()[0]),
                    np.interp(eval_array, curve_length, bound_array.transpose()[1]),
                ]
            )
            return rv
        # if no point is left return None
        return None
        # endregion

    # endregion

    # region call subfunction and add concat pixel values
    interp_bound_list = []
    for bound_line, line_type in zip(limit_bound_list, type_list):
        if line_type == "road-boundary":
            value = 255
        elif line_type == "lane-marking":
            value = 127
        interp_line = interpolate_boundary(bound_line)
        if interp_line is not None:
            value_vec = np.ones(shape=(1, interp_line.shape[1]), dtype=np.uint8) * value
            arr = np.concatenate([interp_line, value_vec], axis=0)
            # print(f"arr shape={arr.shape} dtype={arr.dtype}")
            # print(arr)
            interp_bound_list.append(arr)
        else:
            continue
    # endregion
    # endregion
    # print(f"Time for creating interpolation points:{time.time() - timer}")
    # timer = time.time()

    # region create image indexes
    interp_bound_arr = np.concatenate(interp_bound_list, axis=1)
    pixel_indexes = np.concatenate(
        [
            interp_bound_arr[0:2] // pixel_dist + res / 2,
            interp_bound_arr[2].reshape(1, interp_bound_arr.shape[1]),
        ],
        axis=0,
    )

    # endregion

    # region limit index indices to resolution
    pixel_indexes = np.delete(
        pixel_indexes,
        np.logical_or(
            np.amax(pixel_indexes[0:2], axis=0) > res - 1,
            np.amin(pixel_indexes[0:2], axis=0) < 0,
        ),
        axis=1,
    )

    # endregion

    # print(f"Time for creating index-set:{time.time() - timer}")
    # timer = time.time()

    # region build full-size image
    # create empty black image
    img = np.zeros((res, res))
    pixel_values = pixel_indexes[2] if light_lane_dividers else 0
    # print(pixel_indexes.shape)
    # print(pixel_indexes)
    # print(pixel_values)
    # add values to image
    img[pixel_indexes[1].astype(int), pixel_indexes[0].astype(int)] = pixel_values
    # endregion
    #print(f"Time for building image:{time.time() - timer}")
    
    # print(f"type={img.dtype} shape={img.shape}")
    # plt.imshow(img, interpolation='nearest')
    # plt.show()

    # saving the full size image needs less space than the pixel_index_data
    # there must be any kind of optimisation for saving pickling large tensors
    # in the background
    # pylint: disable=not-callable
    return img
    # pylint: enable=not-callable
    # endregion


def generate_nbr_array(trans_traj_list, time_step, pp=30, window_size=[18, 78], initial_time_step_list=None):
    """Generates the array of trajectories around the vehicle being predicted

    Arguments:
        trans_traj_list {[type]} -- [description]
        time_step {[type]} -- [description]

    Keyword Arguments:
        pp {int} -- [description] (default: {31})
        window_size {list} -- [description] (default: {[18, 78]})

    Returns:
        [type] -- [description]
    """

    # Define initial_time_step list with zeros if None
    if initial_time_step_list is None:
        initial_time_step_list = [0] * len(trans_traj_list)

    # Define window to identify neihbors
    r1 = [int(-i / 2) for i in window_size]  # [-9, -39]
    r2 = [int(i / 2) for i in window_size]

    nbrs = np.zeros((3, 13, pp, 2))
    pir_list = []
    for nbr, init_ts in zip(trans_traj_list, initial_time_step_list):
        try:
            now_point_nbr = nbr[time_step]
        except IndexError:
            continue

        pir = point_in_rectangle(r1, r2, now_point_nbr)
        if pir:
            nbr_tmp = []
            for i in reversed(range(pp)):
                if time_step - (i + init_ts) >= 0:
                    nbr_tmp.append(nbr[time_step - (i + init_ts)])
                else:
                    nbr_tmp.append([np.nan, np.nan])

            nbrs[pir] = nbr_tmp
            pir_list.append(pir)

    return nbrs, pir_list, r1, r2
