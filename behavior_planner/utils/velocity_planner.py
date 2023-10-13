__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import behavior_planner.utils.helper_functions as hf
import logging

# get logger
msg_logger = logging.getLogger("Message_logger")


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

    def execute(self):
        """ Execute velocity planner each time step"""

        self.VP_state.closest_preceding_vehicle, self.VP_state.dist_preceding_veh, self.VP_state.vel_preceding_veh =\
            hf.get_closest_preceding_obstacle(predictions=self.BM_state.predictions,
                                              lanelet_network=self.BM_state.scenario.lanelet_network,
                                              coordinate_system=self.PP_state.cl_ref_coordinate_system,
                                              lanelet_id=self.BM_state.current_lanelet_id,
                                              ego_position_s=self.BM_state.ref_position_s,
                                              ego_state=self.BM_state.ego_state)

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

    def _set_desired_velocity(self):
        if self.BM_state.time_step < 3 or self.VP_state.goal_velocity is None:  # wait till predictions are stabilized
            self.VP_state.desired_velocity = self.BM_state.init_velocity
        else:
            self.VP_state.desired_velocity = self.VP_state.goal_velocity

            # slow car to find gap for lane change maneuvers
            if self.FSM_state.change_velocity_for_lane_change:
                self.VP_state.desired_velocity = \
                    self.BM_state.ego_state.velocity + self.FSM_state.free_space_offset * 0.75
                msg_logger.debug("BP slowing vehicle for lane change maneuver, recommended velocity is: " +
                                 str(self.VP_state.desired_velocity))
                self.FSM_state.change_velocity_for_lane_change = False

            # no strong acceleration while preparing lane changes
            if self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeLeft' \
                    or self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeRight':
                if self.VP_state.desired_velocity > self.BM_state.ego_state.velocity * 1.05:
                    self.VP_state.desired_velocity = self.BM_state.ego_state.velocity * 1.05
                    msg_logger.debug("BP no strong vehicle acceleration while lane change maneuvers, "
                                     "recommended velocity is: " + str(self.VP_state.desired_velocity))

            # stopping for traffic light
            # if self.FSM_state.slowing_car_for_traffic_light:
            #     self.VP_state.desired_velocity = 0

            # prevent large velocity jumps
            if self.BM_state.ego_state.velocity > 8.333:
                if self.VP_state.desired_velocity > self.BM_state.ego_state.velocity * 1.33:
                    msg_logger.debug("BP planner velocity too high, recommended velocity is: " +
                                     str(self.BM_state.ego_state.velocity * 1.33))
                    self.VP_state.desired_velocity = self.BM_state.ego_state.velocity * 1.33
                elif self.VP_state.desired_velocity < self.BM_state.ego_state.velocity * 0.67:
                    msg_logger.debug("BP planner velocity too low, recommended velocity is: " +
                                     str(self.BM_state.ego_state.velocity * 0.67))
                    self.VP_state.desired_velocity = self.BM_state.ego_state.velocity * 0.67

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

    def _calc_safety_distance(self):
        """Calculate the minimum safety distance to a preceding vehicle (TTC)"""
        a_max_ego = self.BM_state.config.vehicle.a_max
        a_max_lead = a_max_ego
        delta = 0.3
        v_lead = self.VP_state.vel_preceding_veh
        v_ego = self.BM_state.ego_state.velocity
        # d_safe_1 = ((((v_lead - (abs(a_max_lead)*delta) - v_ego)**2) / (-2*(abs(a_max_lead)-abs(a_max_ego))))
        #             - (v_lead*delta) + (0.5*abs(a_max_lead)*(delta**2)) + (v_ego*delta))

        d_safe_2 = ((v_lead**2) / (-2*a_max_lead)) - ((v_ego**2) / (-2*a_max_ego)) + v_ego*delta
        if d_safe_2 > 0:
            self.VP_state.safety_dist = d_safe_2 + self.BM_state.config.vehicle.length / 2 + \
                                        self.VP_state.closest_preceding_vehicle.get('shape').get('length') / 2
        else:
            self.VP_state.safety_dist = self.BM_state.config.vehicle.length * 1.5

    def _calc_ttc(self):
        """Calculate Time To Collision velocity (TTC)"""

        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.vel_preceding_veh is not None:
                self._calc_safety_distance()
                self.VP_state.ttc_relative = ((self.VP_state.dist_preceding_veh - self.VP_state.safety_dist)
                                              / self.VP_state.ttc_norm)
                self.VP_state.TTC_unconditioned = self.VP_state.vel_preceding_veh + self.VP_state.ttc_relative
                self.VP_state.TTC = self.VP_state.TTC_unconditioned * self.VP_state.condition_factor
            else:
                self.VP_state.TTC = None
        else:
            self.VP_state.TTC = None

    def _set_default_speed_limit(self):
        """Setting the default speed limit according to the street setting state from FSM"""

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
