__author__ = "Moritz Ellermann, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# general imports
import numpy as np
import logging

# commonroad imports

# project imports
import behavior_planner.utils.helper_functions as hf

# get logger
behavior_message_logger = logging.getLogger("Behavior_logger")


class VelocityPlanner(object):
    """
    Velocity Planner: Used by the Behavior Planner to determine the situation dependent goal velocity.
    Chooses either a calculated Max velocity (MAX) or a Time To Collision velocity (TTC).

    TODO: include rule to not overtake on the right
    TODO: Run time optimization for helper functions
    TODO: include configuration file for ttc_norm
    TODO: get mathematical model for conditions factor
    """

    def __init__(self, BM_state):
        """ Init Velocity Planner

        Args:
        BM_state (BehaviorModuleState)
        """

        # input variables
        self.BM_state = BM_state
        self.VP_state = BM_state.VP_state
        self.PP_state = BM_state.PP_state
        self.FSM_state = BM_state.FSM_state

        # submodules
        self.driving_dynamics = DrivingDynamicsConditions()
        self.visibility = VisibilityConditions()

        # actions
        self._set_default_speed_limit()
        self.VP_state.final_velocity_interval, self.VP_state.final_velocity_center = hf.goal_velocity_interval(
            self.BM_state.planning_problem.goal.state_list,
            BM_state.goal_index)

    def execute(self):
        """ Execute velocity planner each time step"""

        # calculate comfortable stopping distance
        self._calc_comfortable_stopping_distance()

        # calculate driving conditions factor
        self._get_condition_factor()

        # calculate MAX velocity
        self._set_default_speed_limit()
        self._calc_max()

        # calculate TTC velocity
        self._calc_ttc()

        # calculate goal velocity
        self._get_goal_velocity()
        self._set_desired_velocity()
        # maybe recommend overtaking previous car if ego vehicle needs to slow down to follow previous car and
        # previous car is driving 10 or more km/h slower than allowed

    def _set_desired_velocity(self):
        # clip velocity to maximal accelerating and braking values
        self.VP_state.desired_velocity = self._clip_velocity()

        # slow car to find gap for lane change maneuvers
        if self.FSM_state.change_velocity_for_lane_change:
            self.VP_state.desired_velocity = \
                self.BM_state.ego_state.velocity + self.FSM_state.free_space_offset * 0.75
            behavior_message_logger.debug("BP slowing vehicle for lane change maneuver, recommended velocity is: " +
                                          str(self.VP_state.desired_velocity))
            self.FSM_state.change_velocity_for_lane_change = False

        # no strong acceleration while preparing lane changes
        # maybe no strong acceleration when changing to the right & no strong deceleration when changing to the left
        # if self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeLeft' \
        #         or self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeRight':
        #     if self.VP_state.desired_velocity > self.BM_state.ego_state.velocity * 1.05:
        #         self.VP_state.desired_velocity = self.BM_state.ego_state.velocity * 1.05
        #         behavior_message_logger.debug("BP no strong vehicle acceleration while lane change maneuvers, "
        #                                       "recommended velocity is: " + str(self.VP_state.desired_velocity))

        # stopping for traffic light
        # if self.FSM_state.slowing_car_for_traffic_light:
        #     self.VP_state.desired_velocity = 0

        if self.VP_state.desired_velocity <= self.BM_state.config.behavior.zero_velocity_threshold:
            self.VP_state.desired_velocity = 0

    def _clip_velocity(self):
        """
        Clips the velocity to keep it
            1. between the minimal and maximal velocity.
            2. inside the maximal possible acceleration.

        This method also clips the velocity according to the constraints if the current velocity
        is outside the minimal and maximal velocity interval. \n
        Negative velocities are also handled appropriately.

        # :param input_vel: The desired velocity
        # :param v_ego: Current velocity of the vehicle
        # :param a_max: Maximal acceleration of the vehicle
        # :param v_max: Maximal velocity of the vehicle
        # :param v_min: Minimal velocity of the vehicle
        # :param delta: factor by which to multiply the maximal acceleration

        :returns: clipped velocity
        """
        input_vel = self.VP_state.goal_velocity
        v_ego = self.BM_state.ego_state.velocity
        a_max = self.BM_state.config.vehicle.a_max
        v_max = self.BM_state.config.vehicle.v_max
        v_min = 0.0  # TODO hard coded
        delta = self.BM_state.config.behavior.a_max_delta

        return min(max(
            input_vel,

            # this is the old velocity with maximal possible deceleration
            (v_ego + (-2 * a_max * delta)) if v_ego > 0 else (v_ego + (-1 * a_max * delta)),
            # make sure that if the velocity is outside the bounds the deceleration is still correctly clipped
            v_min if v_min <= v_ego else v_ego + a_max * delta),

            # this is the old velocity with maximal possible acceleration
            (v_ego + (a_max * delta)) if v_ego >= 0 else (v_ego + (2 * a_max * delta)),
            # make sure that if the velocity is outside the bounds the acceleration is still correctly clipped
            v_max if v_max >= v_ego else v_ego + (-2 * a_max * delta))

    def _get_goal_velocity(self):
        """Compare TTC and MAX velocities and set the final goal velocity"""

        if not (self.VP_state.MAX is None and self.VP_state.TTC is None):

            if self.VP_state.MAX is None:
                self.VP_state.goal_velocity = self.VP_state.TTC
                self.VP_state.velocity_mode = 'TTC'
            elif self.VP_state.TTC is None:
                self.VP_state.goal_velocity = self.VP_state.MAX
                self.VP_state.velocity_mode = 'MAX'

            else:
                if self.VP_state.MAX <= self.VP_state.TTC:
                    self.VP_state.goal_velocity = self.VP_state.MAX
                    self.VP_state.velocity_mode = 'MAX'
                elif self.VP_state.MAX > self.VP_state.TTC:
                    self.VP_state.goal_velocity = self.VP_state.TTC
                    self.VP_state.velocity_mode = 'TTC'
        else:
            self.VP_state.goal_velocity = None
            self.VP_state.velocity_mode = None

        # aim towards the velocity of the final goal when stop points aims at it
        if str(self.BM_state.stop_point_mode).endswith("final goal") and (
                self.VP_state.TTC is None or self.BM_state.desired_velocity_stop_point < self.VP_state.TTC):
            self.VP_state.goal_velocity = self.BM_state.desired_velocity_stop_point
            self.VP_state.velocity_mode = 'final'


    def _calc_safety_distance(self):
        """
        relation between the distance it takes the ego vehicle to a come to a stop
        and the distance it takes the other vehicle to a come to a stop \n
        after calculating the minimal safety distance between the two vehicles a
        safety buffer equivalent to the distance the trailing vehicle travels in 2s
        is added to the safety distance \n

        there are eight scenarios: (all variables are relative to the s-position)
            1. dist >= 0 and v_ego >= 0 and v_other >= 0 (2) \n
            2. dist >= 0 and v_ego >= 0 and v_other <  0 (1) \n
            3. dist >= 0 and v_ego <  0 and v_other >= 0 (4) \n
            4. dist >= 0 and v_ego <  0 and v_other <  0 (3) \n
            5. dist <  0 and v_ego >= 0 and v_other >= 0 (3) \n
            6. dist <  0 and v_ego >= 0 and v_other <  0 (4) \n
            7. dist <  0 and v_ego <  0 and v_other >= 0 (1) \n
            8. dist <  0 and v_ego <  0 and v_other <  0 (2) \n

        this results in four situations:
            1. ego -><- other \n
            2. ego ->-> other \n
            3. ego <-<- other \n
            4. ego <--> other \n

        resulting values:
            1. safety_dist: is the safety distance \n
            2. relevant: is a boolean indicating that the ego vehicle is heading towards the other vehicle \n

        # :param v_ego: velocity of the ego vehicle
        # :param v_other: velocity of the other vehicle
        # :param a_max: the maximal acceleration
        # :param delta: reaction time in seconds
        # :param dist: current distance between the ego vehicle and the other vehicle

        :returns: relevant
        :return type: boolean
        """
        # get base values
        v_ego = self.BM_state.ego_state.velocity
        v_other = self.VP_state.vel_preceding_veh
        a_max_ego = self.BM_state.vehicle_params.a_max
        a_max_other = a_max_ego  # assume that the other vehicle has the same properties
        len_ego = self.BM_state.vehicle_params.length
        dist = self.VP_state.dist_preceding_veh
        delta = self.BM_state.dt * self.BM_state.config.behavior.replanning_frequency
        safety_dist_sec = self.BM_state.config.behavior.safety_distance_buffer

        # calculate reaction and stopping distances
        ego_react_dist = v_ego * delta
        other_react_dist = v_other * delta
        ego_stop_dist = hf.stop_distance(v_ego, a_max_ego)
        other_stop_dist = hf.stop_distance(v_other, a_max_other)
        self.VP_state.stop_dist_preceding_veh = abs(other_stop_dist)

        # set result values
        safety_dist = len_ego / 2 + 0.5
        relevant = True

        # ego vehicle and other vehicle drive towards each other (1)
        if (dist >= 0 and v_ego >= 0 and v_other < 0) or (dist <  0 and v_ego < 0 and v_other >= 0):
            safety_dist += abs(ego_react_dist) + abs(ego_stop_dist) + abs(other_stop_dist)
            self.VP_state.min_safety_dist = safety_dist
            safety_dist += max(v_ego * safety_dist_sec, v_other * safety_dist_sec)

        # ego vehicle drives behind other vehicle (2)
        elif (dist >= 0 and v_ego >= 0 and v_other >= 0) or (dist < 0 and v_ego < 0 and v_other < 0):
            safety_dist += abs(ego_react_dist) + abs(ego_stop_dist) - abs(other_stop_dist)
            self.VP_state.min_safety_dist = safety_dist
            safety_dist += (v_other + v_ego) / 2 * safety_dist_sec

        # ego vehicle drives in front of other vehicle (3)
        elif (dist >= 0 and v_ego < 0 and v_other < 0) or (dist < 0 and v_ego >= 0 and v_other >= 0):
            safety_dist += abs(other_react_dist) + abs(other_stop_dist) - abs(ego_stop_dist)
            self.VP_state.min_safety_dist = safety_dist
            safety_dist += (v_other + v_ego) / 2 * safety_dist_sec
            relevant = False

        # ego vehicle and other vehicle drive away from each other (4)
        elif (dist >= 0 and v_ego < 0 and v_other >= 0) or (dist <  0 and v_ego >= 0 and v_other <  0):
            safety_dist += -np.inf
            self.VP_state.min_safety_dist = safety_dist
            relevant = False

        self.VP_state.safety_dist = safety_dist
        return relevant

    def _calc_ttc(self):
        """Calculate Time To Collision velocity (TTC)"""
        # TODO make use of the 'relevant' flag from _calc_safety_distance

        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.vel_preceding_veh is not None:
                self._calc_safety_distance()
                self.VP_state.ttc_relative = ((self.VP_state.dist_preceding_veh - self.VP_state.safety_dist)
                                              / self.VP_state.ttc_norm)
                self.VP_state.TTC_unconditioned = self.VP_state.vel_preceding_veh + self.VP_state.ttc_relative
                self.VP_state.TTC = self.VP_state.TTC_unconditioned * self.VP_state.condition_factor
            else:
                self.VP_state.TTC = None
                self.VP_state.stop_dist_preceding_veh = None
                self.VP_state.min_safety_dist = None
        else:
            self.VP_state.TTC = None
            self.VP_state.stop_dist_preceding_veh = None
            self.VP_state.min_safety_dist = None

    def _calc_comfortable_stopping_distance(self):
        ego_react_dist = (self.BM_state.ego_state.velocity
                          * self.BM_state.dt
                          * self.BM_state.config.behavior.replanning_frequency)
        self.VP_state.comfortable_stopping_distance = (
                ego_react_dist
                + hf.stop_distance(self.BM_state.ego_state.velocity,
                                      self.BM_state.config.behavior.comfortable_deceleration_rate))

    def _set_default_speed_limit(self):
        """Setting the default speed limit according to the street setting state from FSM"""
        # TODO make it Country dependent: DEU/ESP/USA/ZAM etc.

        if self.FSM_state.street_setting == 'Highway':
            self.VP_state.speed_limit_default = 130 / 3.6
        elif self.FSM_state.street_setting == 'Country':
            self.VP_state.speed_limit_default = 100 / 3.6
        elif self.FSM_state.street_setting == 'Urban':
            self.VP_state.speed_limit_default = 50 / 3.6
        else:
            self.VP_state.speed_limit_default = 30 / 3.6

    def _calc_max(self):
        """Calculate Max velocity (MAX)"""

        if self.BM_state.speed_limit is not None:
            self.VP_state.MAX = self.BM_state.speed_limit * self.VP_state.condition_factor
        else:
            self.VP_state.MAX = self.VP_state.speed_limit_default * self.VP_state.condition_factor

    def _get_condition_factor(self):
        """Calculate factor to express the driving conditions of the vehicle. Factor ∈ [0,1]"""

        self._get_lon_dyn_cond_factor()
        self._get_lat_dyn_cond_factor()
        self._get_visual_cond_factor()

        # calculate factor. TODO: formula for combination of the condition factors
        self.VP_state.condition_factor = \
            self.VP_state.lon_dyn_cond_factor * self.VP_state.lat_dyn_cond_factor * self.VP_state.visual_cond_factor

    def _get_lon_dyn_cond_factor(self):
        """Calculate factor to express influences on longitudinal dynamic behavior of the vehicle. Factor ∈ [0,1]
        Conditions: Grip, Vehicle Parameters, Maintenance, Street Setting
        """

        self.VP_state.lon_dyn_cond_factor = self.driving_dynamics.execute()[0]

    def _get_lat_dyn_cond_factor(self):
        """Calculate factor to express influences on lateral dynamic behavior of the vehicle. Factor ∈ [0,1]
        Conditions: Grip, Vehicle Parameters, Maintenance, Street Setting, Curve
        """

        self.VP_state.lat_dyn_cond_factor = self.driving_dynamics.execute()[1]

    def _get_visual_cond_factor(self):
        """Calculate factor to express influences on the visibility. Factor ∈ [0,1]
        Conditions: Human Visibility, Sensor Range, Unobstructed View, Curve View
        """

        self.VP_state.visual_cond_factor = self.visibility.execute()


class DrivingDynamicsConditions(object):
    """Class to estimate the driving conditions influencing the longitudinal and lateral dynamic performance of the
    vehicle.

    TODO: get sufficient mathematical model for driving dynamics estimation
    TODO: check for possible integration of street setting and curve
    """

    def __init__(self):
        """Init the driving dynamics module"""

        self.grip = None
        self.vehicle_parameters = None
        self.maintenance = None
        self.street_setting = None
        self.curve = None

        self.longitudinal_conditions = None
        self.lateral_conditions = None

    def execute(self):
        """Execute the estimation of the driving dynamic conditions.
        Returns [longitudinal conditions factor, lateral conditions factor]"""

        self._estimate_grip()
        self._estimate_vehicle_parameters()
        self._estimate_maintenance()
        self._estimate_street_setting()
        self._estimate_curve()

        self._calc_longitudinal_conditions()
        self._calc_lateral_conditions()

        return [self.longitudinal_conditions, self.lateral_conditions]

    def _calc_longitudinal_conditions(self):
        """Estimate the longitudinal driving conditions factor"""

        # TODO: missing model to estimate longitudinal conditions
        self.longitudinal_conditions = 1.0

    def _calc_lateral_conditions(self):
        """Estimate the lateral driving conditions factor"""

        # TODO: missing model to estimate lateral conditions
        self.lateral_conditions = 1.0

    def _estimate_grip(self):
        """Estimate the grip based on camera images, temperature, gps data, etc."""
        self.grip = 1  # not included in this model

    def _estimate_vehicle_parameters(self):
        """Estimate vehicle parameters like weight, wheel parameters, center of mass, etc."""
        self.vehicle_parameters = []  # not included in this model

    def _estimate_maintenance(self):
        """Estimate maintenance vehicle parameters like brakes, tire profile, etc"""
        self.maintenance = []  # not included in this model

    def _estimate_street_setting(self):
        """Estimate the street setting based on gps data, camera images etc."""

        # TODO: check for information withing the scenarios
        self.street_setting = "Highway"  # e.g: highway, roadworks, village

    def _estimate_curve(self):
        """Estimate the curve conditions based on gps data, camera images etc."""

        # TODO: check for information withing the scenarios
        self.curve = [None, None]  # [curvature in rad/s, slope in degrees]


class VisibilityConditions(object):
    """Class to estimate the visibility conditions influencing of the vehicle.

    TODO: get sufficient mathematical model for visibility estimation
    TODO: integrate unobstructed view estimation
    TODO: check for integration of curve
    """

    def __init__(self):
        """Init the visibility module"""

        self.human_vis = None
        self.sensor_vis = None
        self.unobstructed_view = None
        self.curve_view = None

        self.visibility_conditions = None

    def execute(self):
        """Execute the estimation visibility conditions.
        Returns visibility conditions factor"""
        self._calc_visibility_conditions()

        return self.visibility_conditions

    def _calc_visibility_conditions(self):
        """Estimate the visibility conditions factor"""

        self.visibility_conditions = 1.0  # TODO: missing model to estimate visibility conditions

    def _estimate_human_vis(self):
        """Estimate the human visibility in m based on camera images"""
        self.human_vis = 30  # not included in this model

    def _estimate_sensor_vis(self):
        """Estimate the sensor visibility in m based on radar (lidar/infrared/ultra sonic)"""
        self.sensor_vis = 100  # not included in this model

    def _estimate_unobstructed_view(self):
        """Estimate the percentage of unobstructed view of a predefined view area"""

        # TODO: include function to get percentage if unobstructed view
        self.unobstructed_view = 70  # in %

    def _estimate_curve_view(self):
        """Estimate road visibility due to road curvature"""

        # TODO: check for information within scenario
        self.curve_view = 60  # in m
