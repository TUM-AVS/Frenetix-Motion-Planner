__author__ = "Georg Schmalhofer, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import itertools
import logging
from abc import ABC, abstractmethod

# get logger
msg_logger = logging.getLogger("Message_logger")


class SamplingHandler:
    def __init__(self, dt: float, max_sampling_number: int, t_min: float, horizon: float, delta_d_min: float,
                 delta_d_max: float):
        self.dt = dt
        self.max_sampling_number = max_sampling_number
        self.s_sampling_mode = False

        self.t_min = t_min
        self.horizon = horizon
        self.t_sampling = None

        self.delta_d_min = delta_d_min
        self.delta_d_max = delta_d_max
        self.d_sampling = None

        self.v_sampling = None
        self.s_sampling = None

        self.set_t_sampling()
        self.set_d_sampling()

    def update_static_params(self, t_min: float, horizon: float, delta_d_min: float, delta_d_max: float):
        assert t_min > 0, "t_min cant be <= 0"
        self.t_min = t_min
        self.horizon = horizon
        self.delta_d_min = delta_d_min
        self.delta_d_max = delta_d_max

        self.set_t_sampling()
        self.set_d_sampling()

    def change_max_sampling_level(self, max_samp_lvl):
        self.max_sampling_number = max_samp_lvl

    def set_t_sampling(self):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param horizon: sampled time horizon
        """
        self.t_sampling = TimeSampling(self.t_min, self.horizon, self.max_sampling_number, self.dt)

    def set_d_sampling(self):
        """
        Sets sample parameters of lateral offset
        """
        self.d_sampling = LateralPositionSampling(self.delta_d_min, self.delta_d_max, self.max_sampling_number)

    def set_v_sampling(self, v_min, v_max):
        """
        Sets sample parameters of sampled velocity interval
        """
        self.v_sampling = VelocitySampling(v_min, v_max, self.max_sampling_number)

    def set_s_sampling(self, delta_s_min, delta_s_max):
        """
        Sets sample parameters of lateral offset
        """
        self.s_sampling = LongitudinalPositionSampling(delta_s_min, delta_s_max, self.max_sampling_number)


def generate_sampling_matrix(*, t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range,
                             dd0_range, ddd0_range, d1_range, dd1_range, ddd1_range):
    """
    Generates a sampling matrix with all possible combinations of the given parameter ranges.
    Each row of the matrix is a different combination. Every parameter has to be passed by keyword argument,
    e.g. t0_range=[0, 1, 2], t1_range=[3, 4, 5], etc. to impede errors due to wrong order of arguments.

    Args:
    00: t0_range (np.array or int): Array of possible values for the starting time, or a single integer.
    01: t1_range (np.array or int): Array of possible values for the end time, or a single integer.
    02: s0_range (np.array or int): Array of possible values for the start longitudinal position, or a single integer.
    03: ss0_range (np.array or int): Array of possible values for the start longitudinal velocity, or a single integer.
    04: sss0_range (np.array or int): Array of possible values for the start longitudinal acceleration, or a single integer.
    05: ss1_range (np.array or int): Array of possible values for the end longitudinal velocity, or a single integer.
    06: sss1_range (np.array or int): Array of possible values for the end longitudinal acceleration, or a single integer.
    07: d0_range (np.array or int): Array of possible values for the start lateral position, or a single integer.
    08: dd0_range (np.array or int): Array of possible values for the start lateral velocity, or a single integer.
    09: ddd0_range (np.array or int): Array of possible values for the start lateral acceleration, or a single integer.
    10: d1_range (np.array or int): Array of possible values for the end lateral position, or a single integer.
    11: dd1_range (np.array or int): Array of possible values for the end lateral velocity, or a single integer.
    12: ddd1_range (np.array or int): Array of possible values for the end lateral acceleration, or a single integer.
    13: debug_mode (boolean): If True, print the number of sampled trajectories. default: True

    Returns:
    np.array: 2D array (matrix) where each row is a different combination of parameters.
    """
    # Convert all input ranges to arrays, if they are not already
    ranges = [np.atleast_1d(x) for x in (
        t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range, dd0_range, ddd0_range,
        d1_range, dd1_range, ddd1_range)]

    # Use itertools.product to generate all combinations
    combinations = list(itertools.product(*ranges))

    msg_logger.debug('<ReactivePlanner>: %s trajectories sampled' % len(combinations))
    # Convert the list of combinations to a numpy array and return
    return np.array(combinations)


class Sampling(ABC):
    def __init__(self, minimum: float, maximum: float, max_density: int):

        assert maximum >= minimum
        assert isinstance(max_density, int)
        assert max_density > 0

        self.minimum = minimum
        self.maximum = maximum
        self.max_density = max_density
        self._sampling_vec = list()
        self._initialization()

    @abstractmethod
    def _initialization(self):
        pass

    def to_range(self, sampling_stage: int = 0) -> set:
        """
        Obtain the sampling steps of a given sampling stage
        :param sampling_stage: The sampling stage to receive (>=0)
        :return: The set of sampling steps for the queried sampling stage
        """
        assert 0 <= sampling_stage < self.max_density, '<Sampling/to_range>: Provided sampling stage is' \
                                                           ' incorrect! stage = {}'.format(sampling_stage)
        return self._sampling_vec[sampling_stage]


class VelocitySampling(Sampling):
    def __init__(self, minimum: float, maximum: float, density: int):
        super(VelocitySampling, self).__init__(minimum, maximum, density)

    def _initialization(self):
        n = 3
        for _ in range(self.max_density):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class LateralPositionSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, density: int):
        super(LateralPositionSampling, self).__init__(minimum, maximum, density)

    def _initialization(self):
        n = 3
        for _ in range(self.max_density):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class LongitudinalPositionSampling(Sampling):
    def __init__(self, maximum: float,  minimum: float, density: int):
        super(LongitudinalPositionSampling, self).__init__(maximum, minimum, density)

    def _initialization(self):
        n = 3
        for _ in range(self.max_density):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class TimeSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, density: int, dT: float):
        self.dT = dT
        super(TimeSampling, self).__init__(minimum, maximum, density)

    def _initialization(self):
        for i in range(self.max_density):
            step_size = int((1 / (i + 1)) / self.dT)
            samp = set(np.round(np.arange(self.minimum, self.maximum + self.dT, (step_size * self.dT)), 2))
            samp.discard(elem for elem in list(samp) if elem > round(self.maximum + self.dT, 2))
            self._sampling_vec.append(samp)
