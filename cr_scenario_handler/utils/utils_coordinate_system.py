__author__ = "Gerald Würsching, Christian Pek"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Alpha"

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cr_scenario_handler.utils.helper_functions as hf
from scipy.interpolate import splprep, splev
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import compute_pathlength_from_polyline, compute_curvature_from_polyline, \
    compute_orientation_from_polyline, resample_polyline, chaikins_corner_cutting
from commonroad.common.util import make_valid_orientation


def extend_points(points):
    """Extend the list of points with additional points in the orientation of the line between the two first points."""
    p1, p2 = points[0], points[1]
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    dist = hf.distance(points[0], points[1])
    num_new_points = int(5/dist)

    new_points = []
    for i in range(1, num_new_points + 1):
        new_point = (p1[0] - i * delta_x, p1[1] - i * delta_y)
        new_points.append(new_point)

    # Stack new_points and points and convert them to a numpy array
    return np.vstack((new_points[::-1], points))


def extend_ref_path(ref_path, init_pos):
    """This function is needed due to the fact that we have to shift the planning position of the reactive planner
    to the rear axis. In some scenatios the ref path is not long enough to locate the initial position in curv state"""
    close_point = min(ref_path, key=lambda point: hf.distance(init_pos, point))
    if close_point[0] == ref_path[0, 0] and close_point[1] == ref_path[0, 1]:
        ref_path = extend_points(ref_path)
    return ref_path


def smooth_ref_path(reference: np.ndarray):
    _, idx = np.unique(reference, axis=0, return_index=True)
    reference = reference[np.sort(idx)]

    # reference = preprocess_ref_path(reference)

    distances = np.sqrt(np.sum((reference[0:-2:2]-reference[1:-1:2])**2, axis=1))
    dist_sum_in_m = np.round(np.sum(distances), 3)
    average_dist_in_m = 0.125  # np.round(np.average(distances), 3)

    t = int(4 / average_dist_in_m)  # 5 meters distance per point
    reference = reference[::t]
    spline_discretization = int(6 * dist_sum_in_m)  # 2 = 0.5 m distances between points

    tck, u = splprep(reference.T, u=None, k=3, s=0.0)
    u_new = np.linspace(u.min(), u.max(), spline_discretization)
    x_new, y_new = splev(u_new, tck, der=0)
    reference = np.array([x_new, y_new]).transpose()
    # reference = resample_polyline(reference, 1)

    # remove duplicated vertices in reference path
    _, idx = np.unique(reference, axis=0, return_index=True)
    reference = reference[np.sort(idx)]

    return reference


def interpolate_angle(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Interpolates an angle value between two angles according to the miminal value of the absolute difference
    :param x: value of other dimension to interpolate
    :param x1: lower bound of the other dimension
    :param x2: upper bound of the other dimension
    :param y1: lower bound of angle to interpolate
    :param y2: upper bound of angle to interpolate
    :return: interpolated angular value (in rad)
    """
    def absmin(x):
        return x[np.argmin(np.abs(x))]

    delta = y2 - y1
    # delta_2pi_minus = delta - 2 * np.pi
    # delta_2pi_plus = delta + 2 * np.pi
    # delta = absmin(np.array([delta, delta_2pi_minus, delta_2pi_plus]))

    return make_valid_orientation(delta * (x - x1) / (x2 - x1) + y1)


def extrapolate_ref_path(reference_path: np.ndarray, resample_step: float = 0.25) -> np.ndarray:
    """
    Function to extrapolate the end of the reference path in order to avoid CCosy errors and/or invalid trajectory
    samples when the reference path is shorter than the planning horizon.
    :param reference_path: original reference path
    :param resample_step: interval for resampling
    :return extrapolated reference path
    """
    p = np.poly1d(np.polyfit(reference_path[-2:, 0], reference_path[-2:, 1], 1))
    x = 2.3*reference_path[-1, 0] - reference_path[-2, 0]
    new_polyline = np.concatenate((reference_path, np.array([[x, p(x)]])), axis=0)
    return resample_polyline(new_polyline, step=resample_step)


def preprocess_ref_path(ref_path: np.ndarray, resample_step: float = 0.1, max_curv_desired: float = 0.1):
    """
    Function to preprocess the reference path for maximum curvature and sampling distance
    """
    ref_path_preprocessed = deepcopy(ref_path)
    max_curv = max_curv_desired + 0.2
    while max_curv > max_curv_desired:
        ref_path_preprocessed = np.array(chaikins_corner_cutting(ref_path_preprocessed))
        ref_path_preprocessed = resample_polyline(ref_path_preprocessed, resample_step)
        abs_curv = compute_curvature_from_polyline(ref_path_preprocessed)
        max_curv = max(abs_curv)
    return ref_path_preprocessed


# TODO use wrapper class of CCosy in commonroad_dc instead
class CoordinateSystem:

    def __init__(self, reference: np.ndarray = None, ccosy: CurvilinearCoordinateSystem = None, config_sim=None):
        if ccosy is None:
            assert reference is not None, '<CoordinateSystem>: Please provide a reference path OR a ' \
                                          'CurvilinearCoordinateSystem object.'
            # set reference and create ccosy from given reference

            self.reference = reference
        else:
            assert ccosy is not None, '<CoordinateSystem>: Please provide a reference path OR a ' \
                                          'CurvilinearCoordinateSystem object.'
            # set ccosy and use reference from given ccosy
            self.ccosy = ccosy

        # initialize reference state vectors
        self._ref_pos = compute_pathlength_from_polyline(self.reference)
        self._ref_curv = compute_curvature_from_polyline(self.reference)
        self._ref_theta = np.unwrap(compute_orientation_from_polyline(self.reference))
        self._ref_curv_d = np.gradient(self._ref_curv, self._ref_pos)
        self._ref_curv_dd = np.gradient(self._ref_curv_d, self._ref_pos)

        # For manual debugging reasons:
        if config_sim.visualization.ref_path_debug:
            plt.clf()
            # plt.plot(self._ref_pos[10:-10], self._ref_curv[10:-10], "k")
            plt.plot(self._ref_pos[10:-10], self._ref_curv_d[10:-10], "r")
            plt.savefig(os.path.join(config_sim.simulation.log_path, 'curvature_ref_path.svg'))

    @property
    def reference(self) -> np.ndarray:
        """returns reference path used by CCosy due to slight modifications within the CCosy module"""
        return self._reference

    @property
    def ref_cruv_dd(self) -> np.ndarray:
        """change of curvature rate along reference path"""
        return self._ref_curv_dd

    @reference.setter
    def reference(self, reference):
        """set reference path and creates Curvilinear Coordinate System from given reference"""
        self._ccosy = CurvilinearCoordinateSystem(reference)
        self._reference = np.asarray(self.ccosy.reference_path())

    @property
    def ccosy(self) -> CurvilinearCoordinateSystem:
        """return Curvlinear Coordinate System"""
        return self._ccosy

    @ccosy.setter
    def ccosy(self, ccosy: CurvilinearCoordinateSystem):
        """set ccosy and use reference from given ccosy object"""
        self._ccosy = ccosy
        self._reference = np.asarray(self.ccosy.reference_path())

    @property
    def ref_pos(self) -> np.ndarray:
        """position (s-coordinate) along reference path"""
        return self._ref_pos

    @property
    def ref_curv(self) -> np.ndarray:
        """curvature along reference path"""
        return self._ref_curv

    @property
    def ref_curv_d(self) -> np.ndarray:
        """curvature rate along reference path"""
        return self._ref_curv_d

    @property
    def ref_theta(self) -> np.ndarray:
        """orientation along reference path"""
        return self._ref_theta

    def convert_to_cartesian_coords(self, s: float, d: float) -> np.ndarray:
        """convert curvilinear (s,d) point to Cartesian (x,y) point"""
        try:
            cartesian_coords = self._ccosy.convert_to_cartesian_coords(s, d)
        except:
            cartesian_coords = None

        return cartesian_coords

    def convert_to_curvilinear_coords(self, x: float, y: float) -> np.ndarray:
        """convert Cartesian (x,y) point to curviinear (s,d) point"""
        return self._ccosy.convert_to_curvilinear_coords(x, y)
