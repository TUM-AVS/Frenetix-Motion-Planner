__author__ = "Marc Kaufeld,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
import pandas as pd
from commonroad.geometry.shape import Circle
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline, compute_pathlength_from_polyline
from commonroad_dc.geometry.util import compute_orientation_from_polyline
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import cr_scenario_handler.utils.helper_functions as hf
from cr_scenario_handler.utils.utils_coordinate_system import interpolate_angle


class Measures:
    """
    Class containing the following criticality metrics:
    - HW, THW,
    - TTC, TIT, TET
    - DCE, TTCE
    - BTN, STN, required a_long, required a_lat
    - ET, PET
    - MSD, PSD
    - Jerk, Acc, Vel
    """

    def __init__(self, agent_id, scenario, a_max_long, a_max_lat, radius, tau, t_start, t_end, msg_logger=None):
        """
        Initialization of the measures
        :param agent_id: id of agent to evaluate
        :param scenario: scenario to evaluate
        :param a_max_long: max longitudinal acceleration
        :param a_max_lat: max lateral acceleration
        :param radius: max distance considered to other vehicles
        :param tau: threshold value for tit, tet
        :param t_start: start time step of evaluation
        :param t_end: end time step of evaluation
        :param msg_logger: msg-logger
        """
        # scenario
        self.scenario = scenario
        self.dt = scenario.dt
        self.t_start = t_start
        self.t_end = t_end
        # agent vehicle
        self.agent_id = agent_id
        self.ego = scenario.obstacle_by_id(agent_id)
        self.a_max = a_max_long
        self._a_lat_max = a_max_lat
        # radius to consider vehicles within
        self.radius = radius
        # obstacles in the surrounding
        self.other_obstacles = self._set_other_obstacles()
        # coordinate systems
        self.cosys = self._update_clcs(self.radius)

        # threshold value for TIT and TET
        self.tau_ttc = tau

        self.msg_logger = msg_logger

        self._dce = dict()
        self._ttc = None
        self._a_long_req = None
        self._a_lat_req = None
        self._msd = None

    def _update_clcs(self, radius):
        """
        Updates the curvilinear coordinate system in the configuration setting using the reference path from the lanelet
        where the ego vehicle is currently located to the end of the lanelet.
        """
        cosy = {}
        _, lanes = self._get_successor_lanelets(self.ego.initial_state.position)
        for lane in lanes:
            reference_path = np.array(chaikins_corner_cutting(lane.center_vertices))
            reference_path = resample_polyline(reference_path)
            cosy[lane.lanelet_id] = CurvilinearCoordinateSystem(reference_path, radius, 0.1)

        return cosy

    def _set_other_obstacles(self):
        """
        get other obstacles in scenario
        :return: list with obstacles
        """
        other_obstacles = [obs for obs in self.scenario.obstacles
                           if obs.obstacle_id is not self.agent_id]
        return other_obstacles

    def _get_obstacles_in_proximity(self, time_step):
        """get obstacles in proximity in specified time step"""

        obstacles_within_radius = []
        ego_state = self.ego.state_at_time(time_step)
        for obs in self.other_obstacles:
            # do not consider the ego vehicle
            if obs.obstacle_id != self.ego.obstacle_id:
                obs_state = obs.state_at_time(time_step)
                # if the obstacle is not in the lanelet network at the given time, its occupancy is None
                if obs_state is not None:
                    # calculate the distance between the two obstacles
                    dist = hf.distance(
                        pos1=ego_state.position,
                        pos2=obs_state.position,
                    )
                    # add obstacles that are close enough
                    if dist < self.radius:
                        obstacles_within_radius.append(obs)
        return obstacles_within_radius

    def _get_local_orientation(self, obj_state, cosy):
        """calculate orientation in curvilinear coordinates"""
        try:
            [s, _], s_idx = cosy.convert_to_curvilinear_coords_and_get_segment_idx(obj_state.position[0],
                                                                                   obj_state.position[1])
            # factor for interpolation
            ref_pos = cosy.segments_longitudinal_coordinates()
            ref_theta = np.unwrap(compute_orientation_from_polyline(np.array(cosy.reference_path())))
            theta_cl = obj_state.orientation - interpolate_angle(
                s,
                ref_pos[s_idx],
                ref_pos[s_idx + 1],
                ref_theta[s_idx],
                ref_theta[s_idx + 1])
            while theta_cl < -np.pi:
                theta_cl += 2 * np.pi
            while theta_cl > np.pi:
                theta_cl -= 2 * np.pi
        except ValueError:
            theta_cl = 0
        return theta_cl


    def _sd_distance(self, cosy, other_position, ego_position):
        """
        calculates the distances in curvilinear coordinates
        :param cosy: curvilinear coordinate system
        :param other_position: position 1
        :param ego_position: position 0
        :return: [lat. distance, long. distance], absolute distance
        """
        try:
            dist_sd = (cosy.convert_to_curvilinear_coords(other_position[0], other_position[1]) -
                       cosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1]))
            dist = np.linalg.norm(dist_sd)

        except ValueError:
            # if Coordinate outside of projection domain it's not considered
            dist = np.inf
            dist_sd = [np.inf, np.inf]
        return dist_sd, dist

    def _get_successor_lanelets(self, position):
        """obtain lanelets merged with successors at current position """
        init_lane_id_list = self.scenario.lanelet_network.find_lanelet_by_position([position])[0]
        successors = []
        lanelets = []
        # try:
        for init_lane_id in init_lane_id_list:
            init_lane = self.scenario.lanelet_network.find_lanelet_by_id(init_lane_id)
            (long_lanelets, long_lanelets_ids) = init_lane.all_lanelets_by_merging_successors_from_lanelet(init_lane,
                                                                                                           self.scenario.lanelet_network,
                                                                                                           self.radius)
            successors.extend(set([j for i in long_lanelets_ids for j in i]))
            lanelets.extend(long_lanelets)
        # except:
        #     pass

        return successors, lanelets

    def _va_obs(self, obs, t):
        """get velocity and acceleration of obstacle"""
        if isinstance(obs, DynamicObstacle):
            obs_state = obs.state_at_time(t)
            if hasattr(obs_state, "velocity_y"):
                v = (np.sqrt(obs_state.velocity ** 2 + obs_state.velocity_y ** 2)
                     * np.cos(obs_state.orientation))
            else:
                v = obs_state.velocity
            if hasattr(obs_state, "acceleration_y"):
                a = (np.sqrt(obs_state.acceleration ** 2 + obs_state.acceleration_y ** 2) *
                     np.cos(obs_state.orientation))
            elif hasattr(obs_state, "acceleration"):
                a = obs_state.acceleration
            else:
                try:
                    obs_next_state = obs.state_at_time(t + 1)
                    if obs_next_state is None:
                        obs_next_state = obs_state
                    if hasattr(obs_next_state, "velocity_y"):
                        v_next = (np.sqrt(
                            obs_next_state.velocity ** 2 + obs_next_state.velocity_y ** 2))

                    else:
                        v_next = obs_next_state.velocity
                except:
                    raise ValueError
                a = np.gradient([v, v_next], self.dt)[0]
        else:
            v = 0.0
            a = 0.0
        return v, a

    def _ca_times(self, ca, t, veh):
        """
        calculates the time steps a vehicle enters and leaves a critical area
        :param ca: critical area
        :param t: starting timestep
        :param veh: vehicle considered
        :return: entering time step, exit time step
        """
        enter_time = np.inf
        exit_time = np.inf
        already_in = False
        try:
            for tt in range(t, self.t_end + 1):
                veh_shape = veh.occupancy_at_time(tt).shape.shapely_object
                if veh_shape.intersects(ca) and already_in is False:
                    # if the vehicle is already within the conflict area, then the enter time is set to 0
                    enter_time = max(tt - 1, 0)
                    already_in = True
                if not veh_shape.intersects(ca) and already_in is True:
                    exit_time = tt
                    break
        except AttributeError:
            # if vehicle.occupancy does not exist at time tt
            pass
        return enter_time, exit_time


    def _obs_lane_poly(self, obs, intersec_lanelets):
        """
        obtains the polygon of the of the intersecting lanelet a vehicle is driving one
        :param obs: vehicle driving on a lanelet
        :param intersec_lanelets: lanelets of an intersection
        :return: polygon of intersecting lanelet occupied by obstacle
        """
        obs_lane_poly = None
        obs_state = obs.state_at_time(self.t_start)
        obs_lanelet_ids, lanelet_list = self._get_successor_lanelets(obs_state.position)
        obs_bool_contains = [l in obs_lanelet_ids for l in intersec_lanelets]
        if any(obs_bool_contains):
            for lanelet in lanelet_list:
                point_list = np.array([i.position for i in obs.prediction.trajectory.state_list])
                if len(point_list) < 2:
                    point_list = np.append(point_list, point_list, axis=0)
                if all(lanelet.contains_points(point_list)):
                    obs_lane_poly = lanelet.polygon.shapely_object
                    break
        return obs_lane_poly


    def hw(self):
        """
        Head way:
        iterate each timestep and obstacle in proximity and calculate the head way
         """
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating HW of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        for t in range(self.t_start, self.t_end + 1):
            veh_results = [np.inf]
            hw_obs = [None]

            ego_state = self.ego.state_at_time(t)

            obstacles = self._get_obstacles_in_proximity(time_step=t)

            for obs in obstacles:

                obs_state = obs.state_at_time(t)
                obs_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([obs_state.position])[0]

                for lane_id, cosy in self.cosys.items():

                    if any([str(i) in str(lane_id) for i in obs_lanelet_ids]):
                        theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                        theta_cl_obs = self._get_local_orientation(obs_state, cosy)

                        dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_state.position)
                        l_obs = (abs(obs.obstacle_shape.length / 2 * np.cos(theta_cl_obs)) +
                                 abs(obs.obstacle_shape.width / 2 * np.sin(theta_cl_obs)))
                        l_ego = (abs(self.ego.obstacle_shape.length / 2 * np.cos(theta_cl_ego)) +
                                 abs(self.ego.obstacle_shape.width / 2 * np.sin(theta_cl_ego)))
                        hw = dist_sd[0] - l_obs - l_ego

                        if hw > 0:
                            veh_results.append(hw)
                            hw_obs.append(obs.obstacle_id)
            results[t] = min(veh_results)

        return results

    def thw(self):
        """
        Time head way
        iterate each timestep and obstacle in proximity and calculate the time head way
        """
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating THW of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        for t in range(self.t_start, self.t_end + 1):

            veh_results = [np.inf]

            ego_state = self.ego.state_at_time(t)

            obstacles = self._get_obstacles_in_proximity(time_step=t)
            for obs in obstacles:

                obs_state = obs.state_at_time(t)
                obs_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([obs_state.position])[0]

                for lane_id, cosy in self.cosys.items():

                    if any([str(i) in str(lane_id) for i in obs_lanelet_ids]):
                        theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                        theta_cl_obs = self._get_local_orientation(obs_state, cosy)
                        dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_state.position)

                        l_obs = (abs(obs.obstacle_shape.length / 2 * np.cos(theta_cl_obs)) +
                                 abs(obs.obstacle_shape.width / 2 * np.sin(theta_cl_obs)))
                        l_ego = (abs(self.ego.obstacle_shape.length / 2 * np.cos(theta_cl_ego)) +
                                 abs(self.ego.obstacle_shape.width / 2 * np.sin(theta_cl_ego)))
                        hw = dist_sd[0] - l_obs - l_ego

                        if hw > 0 and ego_state.velocity > 0:
                            # approximate thw with current velocity
                            thw = hw / ego_state.velocity
                            # check if position of obstacle is reached in the future, if so take correct time to obstacle
                            for ts in range(t + 1, self.t_end + 1):
                                ego_position = self.ego.state_at_time(
                                    ts).position  # + [self.ego.obstacle_shape.length / 2, 0]
                                dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_position)
                                hw = dist_sd[0] - obs.obstacle_shape.length / 2 - self.ego.obstacle_shape.length / 2
                                if hw <= 0:
                                    thw = min(thw, (ts - t) * self.dt)
                                    break
                            veh_results.append(thw)

            results[t] = min(veh_results)

        return results

    def ttc(self):
        """Time-to-collision
        iterate each timestep and obstacle in proximity and calculate the time tp collision"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating TTC of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):

            veh_results = [np.inf]

            ego_state = self.ego.state_at_time(t)

            obstacles = self._get_obstacles_in_proximity(time_step=t)

            for obs in obstacles:

                obs_state = obs.state_at_time(t)
                obs_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([obs_state.position])[0]

                for lane_id, cosy in self.cosys.items():

                    if any([str(i) in str(lane_id) for i in obs_lanelet_ids]):
                        theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                        theta_cl_obs = self._get_local_orientation(obs_state, cosy)
                        dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_state.position)

                        l_obs = (abs(obs.obstacle_shape.length / 2 * np.cos(theta_cl_obs)) +
                                 abs(obs.obstacle_shape.width / 2 * np.sin(theta_cl_obs)))
                        l_ego = (abs(self.ego.obstacle_shape.length / 2 * np.cos(theta_cl_ego)) +
                                 abs(self.ego.obstacle_shape.width / 2 * np.sin(theta_cl_ego)))
                        hw = dist_sd[0] - l_obs - l_ego

                        if hw > 0:
                            # actual velocity and acceleration of both vehicles along the lanelet

                            # v_ego = ego_state.velocity
                            # a_ego = ego_state.acceleration
                            v_ego, a_ego = self._va_obs(self.ego, t)

                            v_other, a_other = self._va_obs(obs, t)

                            delta_v = v_other - v_ego
                            delta_a = a_other - a_ego

                            if delta_v < 0 and abs(delta_a) <= 0.1:
                                veh_results.append(-(hw / delta_v))
                            elif delta_v ** 2 - 2 * hw * delta_a < 0:
                                veh_results.append(np.inf)
                            elif (delta_v < 0 and delta_a != 0) or (delta_v >= 0 > delta_a):
                                first = -(delta_v / delta_a)
                                second = np.sqrt(delta_v ** 2 - 2 * hw * delta_a) / delta_a
                                veh_results.append(first - second)
                            else:  # delta_v >= 0 and delta_a >= 0
                                veh_results.append(np.inf)

            results[t] = min(veh_results)
        self._ttc = results
        return results

    def dce(self):
        """
         Calculates the minimum distance of closest encounter between the ego vehicle
         and the surrounding obstacles at each time step
         :return: (pd.Series): dce
         """
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating DCE of {self.agent_id} ")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        min_dce = np.inf
        t_min_dce = np.inf
        for t in reversed(range(self.t_start, self.t_end + 1)):

            veh_results = [np.inf]

            ego_state = self.ego.state_at_time(t)
            ego_shape = self.ego.occupancy_at_time(t).shape.shapely_object

            obstacles = self._get_obstacles_in_proximity(time_step=t)
            for obs in obstacles:
                obs_shape = obs.occupancy_at_time(t).shape.shapely_object
                dist = np.round(ego_shape.distance(obs_shape), 3)
                veh_results.append(dist)

            dce = min(veh_results)
            if dce < min_dce:
                min_dce = dce
                t_min_dce = t

            self._dce[t] = {t_min_dce: min_dce}
            results[t] = min_dce

        return results

    def ttce(self):
        """
         Calculates the minimum time-to-closest-encounter between the ego vehicle
         and the surrounding obstacles at each time step
         :return: (pd.Series): dce
         """
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating TTCE of {self.agent_id} ")
        if not self._dce:
            self.dce()
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            results[t] = (list(self._dce[t].keys())[0] - t) * self.dt
        return results

    def tit(self):
        """ Time integrated TTC starting from each timestep"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating TIT of {self.agent_id} ")
        tau = self.tau_ttc
        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))
        if self._ttc is None:
            self.ttc()
        tit = 0.0
        for t in reversed(range(self.t_start, self.t_end + 1)):
            if self._ttc[t] <= tau:
                tit += (tau - self._ttc[t]) * self.dt
            results[t] = tit
        return results

    def tet(self):
        """Time exposed TTC starting from each timestep"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating TET of {self.agent_id} ")
        tau = self.tau_ttc
        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))
        if self._ttc is None:
            self.ttc()
        tet = 0.0
        for t in reversed(range(self.t_start, self.t_end + 1)):
            if self._ttc[t] <= tau:
                tet += self.dt
            results[t] = tet / (self.t_end - self.t_start)
        return results

    def a_long_req(self):
        """Required longitudinal deceleration to avoid a collision at each time step"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating a_long_req of {self.agent_id} ")

        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            veh_results = [0]

            ego_state = self.ego.state_at_time(t)

            obstacles = self._get_obstacles_in_proximity(time_step=t)
            for obs in obstacles:

                obs_state = obs.state_at_time(t)
                obs_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([obs_state.position])[0]

                for lane_id, cosy in self.cosys.items():

                    if any([str(i) in str(lane_id) for i in obs_lanelet_ids]):
                        theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                        theta_cl_obs = self._get_local_orientation(obs_state, cosy)
                        dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_state.position)

                        l_obs = (abs(obs.obstacle_shape.length / 2 * np.cos(theta_cl_obs)) +
                                 abs(obs.obstacle_shape.width / 2 * np.sin(theta_cl_obs)))
                        l_ego = (abs(self.ego.obstacle_shape.length / 2 * np.cos(theta_cl_ego)) +
                                 abs(self.ego.obstacle_shape.width / 2 * np.sin(theta_cl_ego)))
                        hw = dist_sd[0] - l_obs - l_ego

                        v_ego, _ = self._va_obs(self.ego, t)
                        v_ego_long = v_ego * np.cos(theta_cl_ego)

                        v_other, a_other = self._va_obs(obs, t)
                        v_obs_long = v_other * np.cos(theta_cl_obs)
                        a_obs_long = a_other * np.cos(theta_cl_obs)

                        v_rel = v_ego_long - v_obs_long

                        veh_results.append(min(a_obs_long - v_rel ** 2 / (2 * hw), 0.0))
            results[t] = np.round(min(veh_results), 3)
        self._a_long_req = results
        return results

    def btn(self):
        """Max. Break threat number at each time step"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating BTN of {self.agent_id} ")
        if self._a_long_req is None:
            self.a_long_req()
        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            results[t] = abs(self._a_long_req[t] / self.a_max)
        return results

    def a_lat_req(self):
        """Required lateral acceleration to avoid a collision at each time step"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating a_lat_req of {self.agent_id} ")
        if self._ttc is None:
            self.ttc()

        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            veh_results = [np.inf]
            if self._ttc[t] == np.inf:
                continue
            ego_state = self.ego.state_at_time(t)

            obstacles = self._get_obstacles_in_proximity(time_step=t)
            for obs in obstacles:

                obs_state = obs.state_at_time(t)
                obs_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([obs_state.position])[0]

                for lane_id, cosy in self.cosys.items():

                    if any([str(i) in str(lane_id) for i in obs_lanelet_ids]):
                        theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                        theta_cl_obs = self._get_local_orientation(obs_state, cosy)
                        dist_sd, _ = self._sd_distance(cosy, obs_state.position, ego_state.position)
                        d = dist_sd[1]

                        v_ego, _ = self._va_obs(self.ego, t)
                        v_ego_lat = v_ego * np.sin(theta_cl_ego)

                        v_other, a_other = self._va_obs(obs, t)
                        v_obs_lat = v_other * np.sin(theta_cl_obs)
                        a_obs_lat = a_other * np.sin(theta_cl_obs)

                        if isinstance(obs.obstacle_shape, Circle):
                            obs_width = obs.obstacle_shape.radius * 2
                        else:
                            obs_width = obs.obstacle_shape.width

                        w = (obs_width + self.ego.obstacle_shape.width)

                        v_rel = v_obs_lat - v_ego_lat

                        first = a_obs_lat + (2 * v_rel * self._ttc[t] + 2 * d) / self._ttc[t] ** 2
                        second = w / self._ttc[t] ** 2
                        a_lat_req = abs(min(first + second, first - second))
                        veh_results.append(a_lat_req)
            results[t] = min(veh_results)
        self._a_lat_req = results
        return results

    def stn(self):
        "Steer threat number at each time step"
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating STN of {self.agent_id} ")
        if self._a_lat_req is None:
            self.a_lat_req()
        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            results[t] = abs(self._a_lat_req[t] / self._a_lat_max)
        return results


    def et(self):
        """Encroachment Time
        calculates the time, a vehicle needs to cross a critical area"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating ET of {self.agent_id} ")

        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        res = []
        if len(self.scenario.lanelet_network.intersections) != 0:
            intersec_lanelets = []
            for intersection in self.scenario.lanelet_network.intersections:
                for inc in intersection.incomings:
                    intersec_lanelets.extend(
                        list(inc.successors_left) + list(inc.successors_right) + list(inc.successors_straight))
                ego_lane_poly = self._obs_lane_poly(self.ego, intersec_lanelets)
                veh_results = [[np.inf, np.inf]]
                if ego_lane_poly:
                    # veh_results = []
                    for obs in self.scenario.obstacles:
                        if obs.obstacle_id == self.ego.obstacle_id:
                            continue
                        obs_lane_poly = self._obs_lane_poly(obs, intersec_lanelets)
                        if obs_lane_poly:
                            ca = ego_lane_poly.intersection(obs_lane_poly)
                            ca.buffer(0)

                            ego_enter_time, ego_exit_time = self._ca_times(ca, self.t_start, self.ego)
                            et = ego_exit_time - ego_enter_time
                            if not (np.isnan(et) or np.isinf(et)):
                                veh_results.append([et, ego_enter_time])
                res.append(min(veh_results))

            for t in range(self.t_start, self.t_end + 1):
                if res:
                    res1 = min(res)
                    t_start = res1[1]
                    if t <= t_start:
                        results[t] = res1[0]
                    elif t <= t_start + res1[0]:
                        results[t] = res1[0] - (t - t_start)


                    else:
                        results[t] = np.inf

                else:
                    results[t] = np.inf

        else:
            if self.msg_logger is not None:
                self.msg_logger.debug(f"ET: No intersection as conflict area in the current scenario ")
        return results

    def pet(self):
        """ Post Encroachment Time
        calculates the time between two vehicles crossing a critical area"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating PET of {self.agent_id} ")

        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        res = []
        if len(self.scenario.lanelet_network.intersections) != 0:
            intersec_lanelets = []
            for intersection in self.scenario.lanelet_network.intersections:
                for inc in intersection.incomings:
                    intersec_lanelets.extend(
                        list(inc.successors_left) + list(inc.successors_right) + list(inc.successors_straight))
                ego_lane_poly = self._obs_lane_poly(self.ego, intersec_lanelets)
                veh_results = [[np.inf, np.inf]]
                if ego_lane_poly:

                    for obs in self.scenario.obstacles:
                        if obs.obstacle_id == self.ego.obstacle_id:
                            continue
                        obs_lane_poly = self._obs_lane_poly(obs, intersec_lanelets)
                        if obs_lane_poly:
                            ca = ego_lane_poly.intersection(obs_lane_poly)
                            ca.buffer(0)

                            ego_enter_time, ego_exit_time = self._ca_times(ca, self.t_start, self.ego)
                            obs_enter_time, obs_exit_time = self._ca_times(ca, self.t_start, obs)

                            if any(i == np.inf for i in [ego_exit_time, ego_enter_time, obs_enter_time, obs_exit_time]):
                                dt = np.inf
                                tstep = np.inf
                            elif obs_enter_time > ego_exit_time:
                                # ego encroaches ca first (if at all)
                                dt = abs(obs_enter_time - ego_exit_time)
                                tstep = ego_exit_time
                            elif ego_enter_time > obs_exit_time:
                                # obs encroaches ca first (if at all)
                                dt = abs(ego_enter_time - obs_exit_time)
                                tstep = obs_exit_time
                            else:
                                # vehicles are in ca at same time
                                dt = 0
                                tstep = 0
                            veh_results.append([dt, tstep])
                res.append(min(veh_results))

            for t in range(self.t_start, self.t_end + 1):
                if res:
                    res1 = min(res)
                    if t <= res1[1]:
                        results[t] = res1[0]
                    else:
                        results[t] = np.inf

                else:
                    results[t] = np.inf

        else:
            if self.msg_logger is not None:
                self.msg_logger.debug(f"PET: No intersection as conflict area in the current scenario ")
        return results

    def msd(self):
        """Minimum Stopping Distance to bring the vehicle to a halt"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating MSD of {self.agent_id} ")

        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))

        v = self.v()
        for t in range(self.t_start, self.t_end + 1):
            results[t] = v[t] ** 2 / (2 * abs(self.a_max))
        self._msd = results
        return results

    def psd(self):
        """Proportion of Stopping Distance to next critical area"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating PSD of {self.agent_id} ")

        results = pd.Series(0.0, index=list(range(self.t_start, self.t_end + 1)))
        if self._msd is None:
            self.msd()

        if len(self.scenario.lanelet_network.intersections) != 0:
            intersec_lanelets = []
            veh_results = [[np.inf]]
            for intersection in self.scenario.lanelet_network.intersections:
                for inc in intersection.incomings:
                    intersec_lanelets.extend(
                        list(inc.successors_left) + list(inc.successors_right) + list(inc.successors_straight))
                ego_lane_poly = self._obs_lane_poly(self.ego, intersec_lanelets)

                if ego_lane_poly:

                    for obs in self.scenario.obstacles:

                        if obs.obstacle_id == self.ego.obstacle_id:
                            continue

                        obs_lane_poly = self._obs_lane_poly(obs, intersec_lanelets)
                        if obs_lane_poly:
                            ca = ego_lane_poly.intersection(obs_lane_poly)
                            ca.buffer(0)
                            ego_enter_time, ego_exit_time = self._ca_times(ca, self.t_start, self.ego)
                            if ego_enter_time is not np.inf and ego_enter_time != 0:
                                # use reversed list to get distance from ca
                                pos_list = np.asarray(
                                    [state.position for state in reversed(self.ego.prediction.trajectory.state_list[
                                                                          self.t_start: ego_enter_time + 1])])
                                pathlength_list = compute_pathlength_from_polyline(pos_list)
                                # undo list reversing
                                pathlength_list = [i for i in reversed(pathlength_list)]
                                veh_results.append(pathlength_list)

            for t in range(self.t_start, self.t_end + 1):
                dist_min = np.inf
                for res in reversed(veh_results):
                    if len(res) <= t:
                        veh_results.remove(res)
                    else:
                        dist_min = min(dist_min, res[t])
                results[t] = dist_min / self._msd[t]
        else:
            if self.msg_logger is not None:
                self.msg_logger.debug(f"PSD: No intersection as conflict area in the current scenario ")
        return results


    def v(self):
        """velocity profile of agent"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating velocity of {self.agent_id}")
        v = [self.ego.state_at_time(t).velocity for t in range(self.t_start, self.t_end + 1)]
        results = pd.Series(v, index=list(range(self.t_start, self.t_end + 1)))
        return results

    def v_long(self):
        """longitudinal velocity profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating v_long of {self.agent_id}")

        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        v = self.v()
        vlong = 0

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    vlong = max(vlong, abs(v[t] * np.cos(theta_cl_ego)))
            results[t] = vlong
        return results

    def v_lat(self):
        """lateral velocity profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating v_lat of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        v = self.v()
        vlat = np.inf

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    vlat = min(vlat, abs(v[t] * np.sin(theta_cl_ego)))
            results[t] = vlat
        return results

    def acc(self):
        """acceleration profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating acceleration of {self.agent_id}")
        try:
            acc = [self.ego.state_at_time(t).acceleration for t in range(self.t_start, self.t_end + 1)]
        except AttributeError:
            v = self.v()  # [self.ego.state_at_time(t).velocity for t in range(self.t_start, self.t_end + 1)]
            acc = np.gradient(v, self.dt)
        results = pd.Series(acc, index=list(range(self.t_start, self.t_end + 1)))
        return results

    def a_lat(self):
        """lateral acceleration profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating a_lat of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        acc = self.acc()
        alat = np.inf

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    alat = min(alat, abs(acc[t] * np.sin(theta_cl_ego)))
            results[t] = alat
        return results

    def a_long(self):
        """longitudinal acceleration profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating a_long of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        acc = self.acc()
        along = 0

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    along = max(along, abs(acc[t] * np.cos(theta_cl_ego)))
            results[t] = along
        return results

    def jerk(self):
        """jerk profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk of {self.agent_id}")
        try:
            jerk = [self.ego.state_at_time(t).jerk for t in range(self.t_start, self.t_end + 1)]
        except AttributeError:
            ego_acc = self.acc()
            jerk = np.gradient(ego_acc, self.dt)

        results = pd.Series(jerk, index=list(range(self.t_start, self.t_end + 1)))
        return results

    def jerk_lat(self):
        """lateral jerk profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk_lat of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        jerk = self.jerk()

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            jlat = np.inf
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    jlat = min(jlat, abs(jerk[t] * np.sin(theta_cl_ego)))
            results[t] = jlat

        return results

    def jerk_long(self):
        """longitudinal jerk profile of agent along calculated trajectory"""
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk_long of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        jerk = self.jerk()

        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            jlong = 0
            for lane_id, cosy in self.cosys.items():
                if any([str(i) in str(lane_id) for i in ego_lanelet_ids]):
                    theta_cl_ego = self._get_local_orientation(ego_state, cosy)
                    jlong = max(jlong, abs(jerk[t] * np.cos(theta_cl_ego)))
            results[t] = jlong

        return results
