__author__ = "Christian Pek, Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"

from typing import Union, List
import numpy as np
from abc import ABC, abstractmethod
import math
import multiprocessing
from itertools import repeat
from multiprocessing.context import Process
from commonroad.scenario.obstacle import DynamicObstacle
from frenetix_motion_planner.polynomial_trajectory import PolynomialTrajectory


class Sample(ABC):
    """
    Abstract class representing a trajectory sample in a certain coordinate system
    """

    def __init__(self, current_time_step: int):
        """
        :param current_time_step
        """
        self.current_time_step = current_time_step

    @property
    def current_time_step(self):
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, curr_time_step):
        self._current_time_step = curr_time_step

    @abstractmethod
    def length(self) -> int:
        """
        Returns the length of the sample
        :return:
        """
        pass

    @abstractmethod
    def enlarge(self, dt: float):
        """
        Enlarges the sample specified by a certain number of steps
        :param dt: The time step of between each step
        """
        pass


class CartesianSample(Sample):
    """
    Class representing the cartesian trajectory of a given trajectory sample
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, v: np.ndarray, a: np.ndarray, kappa: np.ndarray,
                 kappa_dot: np.ndarray, current_time_step: int):
        super().__init__(current_time_step)
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.a = a
        self.kappa = kappa
        self.kappa_dot = kappa_dot

    @property
    def x(self) -> np.ndarray:
        """
        Returns the x positions of the trajectory in Cartesian space
        :return: x positions as numpy array
        """
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self) -> np.ndarray:
        """
        Returns the y positions of the trajectory in Cartesian space
        :return: y positions as numpy array
        """
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def theta(self) -> np.ndarray:
        """
        Returns the orientations of the trajectory in Cartesian space
        :return: Orientations as numpy array
        """
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def v(self) -> np.ndarray:
        """
        Returns the velocities of the trajectory in Cartesian space
        :return: Velocities as numpy array
        """
        return self._v

    @v.setter
    def v(self, v):
        self._v = v

    @property
    def a(self) -> np.ndarray:
        """
        Returns the accelerations of the trajectory in Cartesian space
        :return: Accelerations as numpy array
        """
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def kappa(self) -> np.ndarray:
        """
        Returns the curvatures of the trajectory in Cartesian space
        :return: Curvatures as numpy array
        """
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        self._kappa = kappa

    @property
    def kappa_dot(self) -> np.ndarray:
        """
        Returns the curvature change of the trajectory in Cartesian space
        :return: Curvature change as numpy array
        """
        return self._kappa_dot

    @kappa_dot.setter
    def kappa_dot(self, kappa_dot):
        self._kappa_dot = kappa_dot

    def length(self) -> int:
        """
        Returns the length of the trajectory
        :return: Length of the trajectory as int
        """
        return len(self.x)

    def enlarge(self, dt: float):
        """
        Enlarges the Cartesian sample specified by the number of steps
        :param dt: The time step between two consecutive steps
        """
        last_time_step = self.current_time_step - 1
        steps = self.length() - self.current_time_step

        # create time index
        t = np.arange(1, steps + 1, 1) * dt
        # enlarge acceleration values
        self.a[self.current_time_step:] = np.repeat(self.a[last_time_step], steps)

        # enlarge velocities by considering acceleration
        # TODO remove a?
        v_temp = self.v[last_time_step] + t * self.a[-1]
        # remove negative velocities
        v_temp = v_temp * np.greater_equal(v_temp, 0)
        self.v[self.current_time_step:] = v_temp

        for i in range(last_time_step + 1, last_time_step + steps):
            radians1 = math.atan2(self.y[i] - self.y[i-1], self.x[i] - self.x[i-1])
            radians2 = math.atan2(self.y[i+1] - self.y[i], self.x[i+1] - self.x[i])
            self.theta[i] = (radians1 + radians2) / 2
        self.theta[-1] = (self.theta[-2]-self.theta[-3]) + self.theta[-2]

        # enlarge curvatures
        self.kappa[self.current_time_step:] = np.repeat(self.kappa[last_time_step], steps)
        # enlarge curvature changes
        self.kappa_dot[self.current_time_step:] = np.repeat(self.kappa_dot[last_time_step], steps)

        # enlarge positions
        # self.x[self.current_time_step:] = self.x[last_time_step] + np.cumsum(dt * v_temp * math.cos(self.theta[last_time_step]))
        # self.y[self.current_time_step:] = self.y[last_time_step] + np.cumsum(dt * v_temp * math.sin(self.theta[last_time_step]))
        self.current_time_step = self.length()


class CurviLinearSample(Sample):
    """
    Class representing the curvilinear trajectory of a given trajectory sample
    """

    def __init__(self, s: np.ndarray, d: np.ndarray, theta: np.ndarray, current_time_step: int, dd=None, ddd=None, ss=None, sss=None):
        super().__init__(current_time_step)
        self.s = s
        self.d = d
        self.theta = theta
        self.d_dot = dd
        self.d_ddot = ddd
        self.s_dot = ss
        self.s_ddot = sss

    @property
    def s(self) -> np.ndarray:
        """
        Returns the s coordinate of the sample
        :return:
        """
        return self._s

    @s.setter
    def s(self, s):
        self._s = s

    @property
    def d(self) -> np.ndarray:
        """
        Returns the d coordinate of the sample
        :return:
        """
        return self._d

    @d.setter
    def d(self, d):
        self._d = d

    @property
    def theta(self) -> np.ndarray:
        """
        Returns the orientations of the sample
        :return:
        """
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def d_dot(self) -> np.ndarray:
        """
        Returns the derivation of the d coordinate
        :return:
        """
        return self._d_dot

    @d_dot.setter
    def d_dot(self, d_dot):
        self._d_dot = d_dot

    @property
    def d_ddot(self) -> np.ndarray:
        """
        Returns the second derivation of the d coordinate
        :return:
        """
        return self._d_ddot

    @d_ddot.setter
    def d_ddot(self, d_ddot):
        self._d_ddot = d_ddot

    @property
    def s_dot(self) -> np.ndarray:
        """
        Returns the derivation of the s coordinate
        :return:
        """
        return self._s_dot

    @s_dot.setter
    def s_dot(self, s_dot):
        self._s_dot = s_dot

    @property
    def s_ddot(self) -> np.ndarray:
        """
        Returns the second derivation of the s coordinate
        :return:
        """
        return self._s_ddot

    @s_ddot.setter
    def s_ddot(self, s_ddot):
        self._s_ddot = s_ddot

    def length(self) -> int:
        return len(self.s)

    def enlarge(self, dt: float):
        """
        Enlarges the curvilinear sample specified by the number of steps
        :param dt: The time step between two v steps
        """
        last_time_step = self._current_time_step - 1
        steps = self.length() - self._current_time_step

        # create time array
        t = np.arange(1, (steps + 1), 1) * dt
        # enlarge velocities by considering acceleration
        #TODO Remove acceleration?
        s_dot_temp = self.s_dot[last_time_step] + t * self.s_ddot[-1]
        # remove negative velocities
        s_dot_temp = s_dot_temp * np.greater_equal(s_dot_temp, 0)
        self.s_dot[self.current_time_step:] = s_dot_temp

        # enlarge velocities by considering acceleration
        # TODO remove acceleration?
        d_dot_temp = self.d_dot[last_time_step] + t * self.d_ddot[-1]
        self.d_dot[self.current_time_step:] = d_dot_temp

        # enlarge accelerations
        self.s_ddot[self.current_time_step:] = np.repeat(self.s_ddot[last_time_step], steps)
        self.d_ddot[self.current_time_step:] = np.repeat(self.d_ddot[last_time_step], steps)

        # enlarge orientations
        # self.theta[self.current_time_step:] = np.repeat(self.theta[last_time_step], steps)

        # enlarge positions
        # self.s[self.current_time_step:] = self.s[last_time_step] + t * self.s_dot[last_time_step]
        # self.d[self.current_time_step:] = self.d[last_time_step] + t * self.d_dot[last_time_step]
        self.current_time_step = self.length()


class TrajectorySample(Sample):
    """
    Class representing a TrajectorySample which stores the information about the polynomials
    for the longitudinal and lateral motion
    """

    def __init__(self, horizon: float, dt: float, trajectory_long: PolynomialTrajectory,
                 trajectory_lat: PolynomialTrajectory, uniqueId: int, costMap: dict):
        self.horizon = horizon
        self.dt = dt
        assert isinstance(trajectory_long,
                          PolynomialTrajectory), '<TrajectorySample/init>: Provided longitudinal trajectory ' \
                                                 'is not feasible! trajectory = {}'.format(
            trajectory_long)
        self._trajectory_long = trajectory_long
        assert isinstance(trajectory_lat,
                          PolynomialTrajectory), '<TrajectorySample/init>: Provided lateral trajectory ' \
                                                 'is not feasible! trajectory = {}'.format(
            trajectory_lat)
        self._trajectory_lat = trajectory_lat

        self._cost = 0
        self._cost_list = list()
        self._cost_function = None
        self._cartesian: CartesianSample
        self._curvilinear: CurviLinearSample
        self._occupancy: DynamicObstacle
        self._ext_cartesian = None
        self._ext_curvilinear = None
        self.feasible = None
        self.valid = None
        self._ego_risk = None
        self._obst_risk = None
        self.boundary_harm = None
        self._coll_detected = None
        self.actual_traj_length = None
        self.harm_occ_module = None

        self.uniqueId = uniqueId
        self.costMap = {x: (0, 0) for x in costMap}

    @property
    def trajectory_long(self) -> PolynomialTrajectory:
        """
        Returns the longitudinal polynomial trajectory
        :return: The longitudinal polynomial trajectory
        """
        return self._trajectory_long

    @trajectory_long.setter
    def trajectory_long(self, trajectory_long):
        pass

    @property
    def trajectory_lat(self) -> PolynomialTrajectory:
        """
        Returns the lateral polynomial trajectory
        :return: The lateral polynomial trajectory
        """
        return self._trajectory_lat

    @trajectory_lat.setter
    def trajectory_lat(self, trajectory_lat):
        pass

    def set_costs(self, cost, cost_list, cost_list_weighted, cost_names):
        """
        Evaluated cost of the trajectory sample
        :return: The cost of the trajectory sample
        """
        self._cost = cost
        for idx, names in enumerate(cost_names):
            self.costMap[names] = (cost_list[idx], cost_list_weighted[idx])

    @property
    def cost(self) -> float:
        """
        Evaluated cost of the trajectory sample
        :return: The cost of the trajectory sample
        """
        return self._cost

    @property
    def cost_list(self) -> list:
        """
        Evaluated cost of the trajectory sample
        :return: The cost list of the trajectory sample
        """
        return self._cost_list

    @property
    def curvilinear(self) -> CurviLinearSample:
        """
        The curvilinear sample of the trajectory sample
        :return: The curvilinear sample
        """
        return self._curvilinear

    @curvilinear.setter
    def curvilinear(self, curvilinear: CurviLinearSample):
        """
        Sets the curvilinear sample of the trajectory sample
        :param curvilinear: The computed curvilinear sample
        """
        assert isinstance(curvilinear, CurviLinearSample)
        self._curvilinear = curvilinear

    @property
    def cartesian(self) -> CartesianSample:
        """
        Returns the Cartesian sample of the trajectory sample
        :return: The Cartesian sample
        """
        return self._cartesian

    @cartesian.setter
    def cartesian(self, cartesian: CartesianSample):
        """
        Sets the Cartesian sample of the trajectory sample
        :param cartesian: The Cartesian sample
        """
        assert isinstance(cartesian, CartesianSample)
        self._cartesian = cartesian

    def length(self) -> int:
        """
        Length of the trajectory sample (number of states)
        :return: The number of states in the trajectory sample
        """
        return self.cartesian.length()

    def set_feasible_status(self, sts: bool):
        self.feasible = sts

    def enlarge(self, dt: float):
        """
        Enlarges the Cartesian and curvilinear sample specified by the given steps
        :param dt: The time step between two consecutive steps
        """
        self._cartesian.enlarge(dt)
        self._curvilinear.enlarge(dt)


class TrajectoryBundle:
    """
    Class that represents a collection of trajectories
    """

    def __init__(self, trajectories: List[TrajectorySample], cost_function, multiproc, num_workers):
        """
        Initializer of a TrajectoryBundle
        :param trajectories: The list of trajectory samples
        :param cost_function: The cost function for the evaluation
        """
        assert isinstance(trajectories, list) and all([isinstance(t, TrajectorySample) for t in trajectories]), \
            '<TrajectoryBundle/init>: ' \
            'Provided list of trajectory samples is not ' \
            'feasible! List = {}'.format(trajectories)

        self.trajectories: List[TrajectorySample] = trajectories
        # self.trajectories_all: List[TrajectorySample] = trajectories
        self._cost_function = cost_function
        self._is_sorted = False
        self._is_sorted_all = False
        self._multiproc = multiproc
        self._num_workers = num_workers

    @property
    def trajectories(self) -> List[TrajectorySample]:
        """
        Returns the list of trajectory samples
        :return: The list of trajectory samples
        """
        return self._trajectory_bundle

    @trajectories.setter
    def trajectories(self, trajectories: List[TrajectorySample]):
        """
        Sets the list of trajectory samples
        :param trajectories: The list of trajectory samples
        """
        self._trajectory_bundle = trajectories

    def _calc_cost(self, trajectories: List[TrajectorySample], queue_1=None):
        self._cost_function.calc_cost(trajectories)
        queue_1.put(trajectories)

    def sort(self, occlusion_module=None):
        """
        Sorts the trajectories within the TrajectoryBundle according to their costs from lowest to highest.
        """
        if not self._is_sorted:
            if self._trajectory_bundle:
                if not self._trajectory_bundle[0].cost > 0:
                    if self._multiproc:
                        # with multiprocessing
                        # divide trajectory_bundle.trajectories into chunks
                        chunk_size = math.ceil(len(self._trajectory_bundle) / self._num_workers)
                        chunks = [self._trajectory_bundle[ii * chunk_size: min(len(self._trajectory_bundle),
                                  (ii + 1) * chunk_size)] for ii in range(0, self._num_workers)]

                        queue_1 = multiprocessing.Queue()
                        list_processes = []
                        trajectories_proc = []

                        for i, chunk in enumerate(chunks):
                            p = Process(target=self._calc_cost, args=(chunk, queue_1))
                            list_processes.append(p)
                            p.start()
                        # # get return values from queue
                        for p in list_processes:
                            trajectories_proc.extend(queue_1.get())
                        for p in list_processes:
                            p.join()

                        self.trajectories = trajectories_proc
                        self._trajectory_bundle = trajectories_proc
                    else:
                        self._cost_function.evaluate(self._trajectory_bundle)

                if occlusion_module is not None:
                    occlusion_module.calc_costs(self._trajectory_bundle)

            self._trajectory_bundle.sort(key=lambda x: x.cost)
            self._is_sorted = True

    def optimal_trajectory(self) -> Union[TrajectorySample, None]:
        """
        Returns the optimal trajectory (the one with minimal costs) from the TrajectoryBundle
        :return: Trajectory in trajectory_bundle with minimal cost. None, if trajectory bundle is empty.
        """
        if not self._trajectory_bundle or not self._is_sorted:
            return None
        return min(self._trajectory_bundle, key=lambda x: x.total_cost)

    def min_costs(self) -> TrajectorySample:
        """
        Returns the trajectory with the minimal costs.
        :return: The trajectory with the minimal costs
        """
        return self._trajectory_bundle[0] if self._is_sorted else None

    def max_costs(self) -> TrajectorySample:
        """
        Returns the trajectory with the maximal costs.
        :return: The trajectory with the maximal costs
        """
        return self._trajectory_bundle[-1] if self._is_sorted else None

    def get_sorted_list(self, occlusion_module=None) -> list:
        """
        Returns an ordered list of the trajectories. The trajectories are ordered according
        to their costs from lowest to highest.
        :return: List of trajectories sorted from lowest to highest cost
        """
        if not self._is_sorted:
            self.sort(occlusion_module=occlusion_module)
        return self._trajectory_bundle

    @property
    def empty(self) -> bool:
        """
        Checks if the TrajectoryBundle is empty, i.e., does not contain any TrajectorySample
        :return:
        """
        return len(self._trajectory_bundle) == 0
