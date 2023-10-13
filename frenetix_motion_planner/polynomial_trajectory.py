__author__ = "Christian Pek, Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"

import numpy as np
import warnings
from abc import ABC, abstractmethod
from methodtools import lru_cache

import commonroad.common.validity as val


class PolynomialTrajectory(ABC):
    """
    Abstract class representing a polynomial trajectory
    """

    def __init__(self, tau_0=0, delta_tau=0, x_0=np.zeros([3, 1]), x_d=np.zeros([3, 1]), power=5):
        """
        Initializer of a polynomial trajectory
        :param tau_0: The start parameter value of the trajectory
        :param delta_tau: The final parameter value of the trajectory
        :param x_0: The initial condition of the trajectory
        :param x_d: The final condition of the trajectory
        :param power:
        """

        super(PolynomialTrajectory, self).__init__()

        self.tau_0 = tau_0
        self.delta_tau = delta_tau
        self.x_0 = x_0
        self.x_d = x_d
        self._cost = None

        # set information about polynomial trajectory
        assert val.is_natural_number(
            power) and power >= 4, '<PolynomialTrajectory/power>: power not valid! power={}'.format(power)
        self._power = power
        if power != 5 and power != 4:
            warnings.warn('Only power of 5 currently supported!')

        # compute coefficients
        self.coeffs = self.calc_coeffs()

    @property
    def power(self) -> float:
        """
        Returns the power of the polynomial trajectory
        :return: The power of the polynomial trajectory
        """
        return self._power

    @property
    def coeffs(self) -> np.ndarray:
        """
        Returns the computed coefficients of the polynomial trajectory
        :return: The computed coefficients of the polynomial trajectory
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, co: np.ndarray):
        """
        Sets the coefficients of the polynomial trajectory
        :param co: The coefficients of the polynomial trajectory
        """
        # todo: remove fixed len = 6 => make modular and adaptable
        assert isinstance(co, np.ndarray) and len(
            co) == 6, '<PolynomialTrajectory/coeffs>: coeffs length not valid! length={}'.format(len(co))
        self._coeffs = co
        # database of already computed queries
        self._db = dict()

    @property
    def delta_tau(self) -> float:
        """
        Returns length of tau interval defining the polynomial trajectory (time duration or arclength)
        :return: Length of tau interval defining the polynomial trajectory (time duration or arclength)
        """
        return self._delta_tau

    @delta_tau.setter
    def delta_tau(self, tau: float):
        assert val.is_positive(tau), '<PolynomialTrajectory/delta_tau>: delta_tau not valid! delta_tau={}'.format(tau)
        self._delta_tau = tau

    @property
    def tau_0(self) -> float:
        """
        Returns initial value of variable describing polynomial trajectory (time or arclength)
        :return: Initial value of variable describing polynomial trajectory
        """
        return self._tau_0

    @tau_0.setter
    def tau_0(self, tau: float):
        """
        Sets initial value of variable describing polynomial trajectory (time or arclength)
        :param tau: New initial value
        """
        assert val.is_real_number(tau) and tau >= 0, '<PolynomialTrajectory/tau_0>: tau_0 not valid! tau_0={}'.format(
            tau)
        self._tau_0 = tau

    @property
    def x_0(self) -> np.ndarray:
        """
        Return initial state of polynomial trajectory
        :return: Initial state of polynomial trajectory as numpy array
        """
        return self._x_0

    @x_0.setter
    def x_0(self, x: np.ndarray):
        """
        Sets initial state of polynomial trajectory
        :param x: New initial state
        """
        assert val.is_real_number_vector(x), '<PolynomialTrajectory/x_0>: x_0 not valid! x_0={}'.format(x)
        self._x_0 = x

    @property
    def x_d(self) -> np.ndarray:
        """
        Return desired state of polynomial trajectory (at the end of planning horizon)
        :return: Desired state of polynomial trajectory as numpy array
        """
        return self._x_d

    @x_d.setter
    def x_d(self, x: np.ndarray):
        """
        Sets desired state of polynomial trajectory (at the end of planning horizon)
        :param x: New desired state
        """
        assert isinstance(x, np.ndarray)
        self._x_d = x

    # calculate coefficients
    @abstractmethod
    def calc_coeffs(self):
        """
        In inherited classes, the coefficients of the polynomial are determined as
        numpy array of shape [1, 6]. Evaluated during initialization.
        :return:
        """
        pass

    @property
    def cost(self) -> float:
        """
        Returns the computed costs of the polynomial trajectory
        :return: The cost of the polynomial trajectory
        """
        return self._cost

    @cost.setter
    def cost(self, cost):
        """
        Sets the cost of the polynomial trajectory
        :param cost: The new cost
        """
        assert val.is_real_number(cost) and cost >= 0, '<PolynomialTrajectory/cost>: cost not valid! cost={}'.format(
            cost)
        self._cost = cost

    def squared_jerk_integral(self, t):
        """
        returns the integral of the squared jerk of a fifth order polynomial in the interval [0, t]
        :param t: time > 0
        :return: evaluated integral
        """
        if self.coeffs is None:
            raise ValueError('Coefficients are not determined')
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        integral_squared_jerk = (36 * self.coeffs[3] * self.coeffs[3] * t + 144 * self.coeffs[3] * self.coeffs[4] * t2 +
                                 240 * self.coeffs[3] * self.coeffs[5] * t3 + 192 * self.coeffs[4] * self.coeffs[
                                     4] * t3 +
                                 720 * self.coeffs[4] * self.coeffs[5] * t4 + 720 * self.coeffs[5] * self.coeffs[
                                     5] * t5)

        return integral_squared_jerk

    def evaluate_state_at_tau(self, tau: float):
        """
        Returns the position, velocity and acceleration at tau of the polynomial with coefficients self.coeffs
        :param tau: point between tau_0 and (tau_0+delta_tau).
        :return: Numpy array of the form [p, p_dot, p_ddot]
        """
        assert self.coeffs is not None, '<PolynomialTrajectory/evaluate_state_at_tau>: Coefficients are not determined!'

        # check if this query has already been processed
        if tau in self._db:
            result = self._db[tau]
        else:
            # normalize query
            tau_prime = tau - self.tau_0

            if tau_prime < 0:
                tau = self.tau_0
            elif tau_prime > self.delta_tau:
                tau = self.delta_tau

            tau2 = np.power(tau, 2)
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau3 * tau2

            # position
            p = self.calc_position(tau, tau2, tau3, tau4, tau5)
            # velocity
            p_d = self.calc_velocity(tau, tau2, tau3, tau4)
            # acceleration
            p_dd = self.calc_acceleration(tau, tau2, tau3)

            result = np.array([p, p_d, p_dd])
            self._db[tau] = result

        return result

    def calc_jerk(self, tau, tau2):
        """
        Computes the jerk at the given parameter values
        :param tau: The values of the query
        :param tau2: The square of the values of the query
        :return: The jerk at the given parameter values
        """
        if self.coeffs is None:
            raise ValueError('Coefficients are not determined')
        return 6 * self.coeffs[3] + 24 * self.coeffs[4] * tau + 60 * self.coeffs[5] * tau2

    def calc_acceleration(self, tau, tau2, tau3):
        """
        Computes the acceleration at the given parameter values
        :param tau: The values of the query
        :param tau2: The square of the values of the query
        :param tau3: The cubic of the values of the query
        :return: The acceleration at the given parameter values
        """
        if self.coeffs is None:
            raise ValueError('Coefficients are not determined')
        return 2 * self.coeffs[2] + 6 * self.coeffs[3] * tau + 12 * self.coeffs[4] * tau2 + 20 * self.coeffs[5] * tau3

    def calc_velocity(self, tau, tau2, tau3, tau4):
        if self.coeffs is None:
            raise ValueError('Coefficients are not determined')
        return (self.coeffs[1] + 2. * self.coeffs[2] * tau + 3. * self.coeffs[3] * tau2 + 4. * self.coeffs[4] * tau3 +
                5. * self.coeffs[5] * tau4)

    def calc_position(self, tau, tau2, tau3, tau4, tau5):
        """
        Calculates the positions of the trajectory
        :param tau: The tau to
        :param tau2: The squared tau
        :param tau3: The cubic tau
        :param tau4: The quartic tau
        :param tau5: The quintic tau
        :return: The positions of the trajectory at the specified tau
        """
        if self.coeffs is None:
            raise ValueError('Coefficients are not determined')
        return (self.coeffs[0] + self.coeffs[1] * tau + self.coeffs[2] * tau2 + self.coeffs[3] * tau3 +
                self.coeffs[4] * tau4 + self.coeffs[5] * tau5)


class QuinticTrajectory(PolynomialTrajectory):
    """
    Class representing a quintic polynomial trajectory
    """

    def __init__(self, tau_0=0, delta_tau=0, x_0=np.zeros([3, 1]), x_d=np.zeros([3, 1])):
        super(QuinticTrajectory, self).__init__(tau_0=tau_0, delta_tau=delta_tau, x_0=x_0, x_d=x_d, power=5)

    def calc_coeffs(self) -> np.ndarray:
        """
        Computes the coefficients of the quintic polynomial trajectory
        :return: The coefficients
        """
        xs, vxs, axs = self.x_0
        xe, vxe, axe = self.x_d
        return QuinticTrajectory._calc_coeffs_static(xs, vxs, axs, xe, vxe, axe,
                                                     self.delta_tau)

    @lru_cache(3024)
    @classmethod
    def _calc_coeffs_static(cls, xs, vxs, axs, xe, vxe, axe, T) -> np.ndarray:
        """
        Initialize a quintic polynomial.

        Args:
            xs (float): Value at the start.
            vxs (float): Value of the first derivation at the start.
            axs (float): Value of the second derivation at the start.
            xe (float): Value at the end.
            vxe (float): Value of the first derivation at the end.
            axe (float): Value of the second derivation at the end.
            T (float): End time.
        """
        # calc coefficient of quintic polynomial
        cls.xs = xs
        cls.vxs = vxs
        cls.axs = axs
        cls.xe = xe
        cls.vxe = vxe
        cls.axe = axe

        cls.a0 = xs
        cls.a1 = vxs
        cls.a2 = axs / 2.0

        A = np.array(
            [
                [T ** 3., T ** 4, T ** 5],
                [3. * T ** 2, 4. * T ** 3, 5. * T ** 4],
                [6. * T, 12. * T ** 2, 20. * T ** 3],
            ]
        )

        b = np.array([xe - xs - vxs * T - .5 * axs * T ** 2,
                      vxe - vxs - axs * T,
                      axe - axs])

        # try to solve linear optimization problem
        try:
            x = np.linalg.solve(A, b)
        except Exception as e:
            print(e)
            return np.empty(0)

        cls.a3 = x[0]
        cls.a4 = x[1]
        cls.a5 = x[2]

        return np.array([xs, vxs, .5 * axs, x[0], x[1], x[2]])

    def calc_point(self, t):
        """
        Calculate position at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Position at time t.
        """
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * np.power(t, 2)
            + self.a3 * np.power(t, 3)
            + self.a4 * np.power(t, 4)
            + self.a5 * np.power(t, 5)
        )

        return xt

    def calc_first_derivative(self, t):
        """
        Calculate first derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: First derivative at time t.
        """
        # xt = (
        #     self.a1
        #     + 2 * self.a2 * t
        #     + 3 * self.a3 * t ** 2
        #     + 4 * self.a4 * t ** 3
        #     + 5 * self.a5 * t ** 4
        # )
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * np.power(t, 2)
            + 4 * self.a4 * np.power(t, 3)
            + 5 * self.a5 * np.power(t, 4)
        )

        return xt

    def calc_second_derivative(self, t):
        """
        Calculate second derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Second derivative at time t.
        """
        # xt = (
        #     2 * self.a2
        #     + 6 * self.a3 * t
        #     + 12 * self.a4 * t ** 2
        #     + 20 * self.a5 * t ** 3
        # )
        xt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * np.power(t, 2)
            + 20 * self.a5 * np.power(t, 3)
        )

        return xt

    def calc_third_derivative(self, t):
        """
        Calculate third derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Third derivative at time t.
        """
        # xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * np.power(t, 2)

        return xt


class QuarticTrajectory(PolynomialTrajectory):
    """
    Class representing a quartic polynomial trajectory
    """

    def __init__(self, tau_0=0, delta_tau=0, x_0=np.zeros([3, 1]), x_d=np.zeros([2, 1])):
        self._desired_velocity = x_d[0]
        super(QuarticTrajectory, self).__init__(tau_0=tau_0, delta_tau=delta_tau, x_0=x_0, x_d=x_d, power=4)

    def calc_coeffs(self) -> np.ndarray:
        """
        Computes the coefficients of the quartic polynomial trajectory
        :return: The coefficients
        """
        xs, vxs, axs = self.x_0
        return QuarticTrajectory._calc_coeffs_static_(xs, vxs, axs,
                                                      self.delta_tau, self._desired_velocity)

    @lru_cache(3024)
    @classmethod
    def _calc_coeffs_static_(cls, xs, vxs, axs, T, _desired_velocity):
        """
        Initialize a quartic polynomial.

        Args:
            xs (float): Start value.
            vxs (float): Value of the first derivation at the start.
            axs (float): Value of the second derivation at the start.
            vxe (float): Value of the first derivation at the end.
            axe (float): Value of the second derivation at the end
            T (float): End time.
        """
        # calc coefficient of quintic polynomial
        cls.xs = xs
        cls.vxs = vxs
        cls.axs = axs
        cls.a0 = xs
        cls.a1 = vxs
        cls.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])

        b = np.array([_desired_velocity - vxs - axs * T, - axs])

        # try to solve linear optimization problem
        try:
            x = np.linalg.solve(A, b)
        except Exception as e:
            print(e)
            return np.empty(0)

        cls.a3 = x[0]
        cls.a4 = x[1]

        return np.array([xs, vxs, .5 * axs, x[0], x[1], 0.])

    def calc_point(self, t):
        """
        Calculate position at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Position at time t.
        """
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t ** 2
            + self.a3 * t ** 3
            + self.a4 * t ** 4
        )

        return xt

    def calc_first_derivative(self, t):
        """
        Calculate first derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: First derivative at time t.
        """
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        """
        Calculate second derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Second derivative at time t.
        """
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        """
        Calculate third derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Third derivative at time t.
        """
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt
