__author__ = "Gerald Würsching, Christian Pek"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Alpha"

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import compute_pathlength_from_polyline, compute_curvature_from_polyline, \
    compute_orientation_from_polyline, resample_polyline, chaikins_corner_cutting
from commonroad.common.util import make_valid_orientation


def smooth_ref_path(reference: np.ndarray):
    _, idx = np.unique(reference, axis=0, return_index=True)
    reference = reference[np.sort(idx)]

    distances = np.sqrt(np.sum((reference[50:-50:2]-reference[51:-49:2])**2, axis=1))
    dist_sum_in_m = np.round(np.sum(distances), 3)
    average_dist_in_m = np.round(np.average(distances), 3)

    t = int(6 / average_dist_in_m)  # 5 meters distance per point
    reference = reference[::t]
    spline_discretization = int(1 * dist_sum_in_m)  # 2 = 0.5 m distances between points

    tck, u = splprep(reference.T, u=None, k=3, s=0.0)
    u_new = np.linspace(u.min(), u.max(), spline_discretization)
    x_new, y_new = splev(u_new, tck, der=0)
    reference = np.array([x_new, y_new]).transpose()
    reference = resample_polyline(reference, 1)

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


def extrapolate_ref_path(reference_path: np.ndarray, resample_step: float = 2.0) -> np.ndarray:
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


def preprocess_ref_path(ref_path: np.ndarray, resample_step: float = 1.0, max_curv_desired: float = 0.01):
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

    def __init__(self, reference: np.ndarray = None, ccosy: CurvilinearCoordinateSystem = None):
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
        # plt.clf()
        # plt.plot(self._ref_pos[20:-20], self._ref_curv_d[20:-20])
        # plt.savefig('curv_window.png')

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

    def plot_reference_states(self):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(7, 7.5))
        plt.suptitle("Reference path states")

        # orientation theta
        plt.subplot(3, 1, 1)
        plt.plot(self.ref_pos, self.ref_theta, color="k")
        plt.xlabel("s")
        plt.ylabel("theta_ref")

        # curvature kappa
        plt.subplot(3, 1, 2)
        plt.plot(self.ref_pos, self.ref_curv, color="k")
        plt.xlabel("s")
        plt.ylabel("kappa_ref")

        # curvature rate kappa_dot
        plt.subplot(3, 1, 3)
        plt.plot(self.ref_pos, self.ref_curv_d, color="k")
        plt.xlabel("s")
        plt.ylabel("kappa_dot_ref")
        plt.tight_layout()
        plt.show()
