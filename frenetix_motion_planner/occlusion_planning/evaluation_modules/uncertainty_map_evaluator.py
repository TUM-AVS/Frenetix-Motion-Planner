__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as ohf
from scipy.spatial.distance import cdist


class OccUncertaintyMapEvaluator:
    def __init__(self, vis_module=None, occ_map=None, occ_plot=None):
        self.trajectories = None
        self.vis_module = vis_module
        self.occlusion_map = occ_map
        self.occ_plot = occ_plot
        self.min_dist = 0.25
        self.max_dist = 3.0
        self.dist_mode = 1
        self.w_distance = 1
        self.w_velocity = 4
        self.costs = None
        self.traj_length_m = None
        self.traj_speeds_occ = None
        self.max_cost = 10
        self.range = 35

    def evaluate_trajectories(self, trajectories, plot_uncertainty_map=False, plot_traj=False):
        # step uncertainty map
        self.occlusion_map.step()

        # store trajectories
        self.trajectories = trajectories

        # store costs and additional information
        self.costs, self.traj_length_m, self.traj_speeds_occ = self._evaluate(trajectories,
                                                                              min_dist=self.min_dist,
                                                                              max_dist=self.max_dist,
                                                                              dist_mode=self.dist_mode,
                                                                              w_distance=self.w_distance,
                                                                              w_velocity=self.w_velocity)
        # plot uncertainty map if activated
        if plot_uncertainty_map and self.occ_plot is not None:
            self.occ_plot.plot_uncertainty_map(self.occlusion_map.map_detail)

        # plot trajectories with color coded costs
        if plot_traj and self.occ_plot is not None:
            self.occ_plot.plot_trajectories_cost_color(trajectories, self.costs)

        return np.zeros(len(trajectories)) if self.costs is None else self.costs

    def _evaluate(self, trajectories, min_dist=0.25, max_dist=3.0, dist_mode=1, w_distance=1, w_velocity=1):

        costs = []
        length = []
        traj_v_speeds = []

        # only use uncertainty map where occlusion value is greater than 0 (not visible)
        occ_map = self.occlusion_map.map[self.occlusion_map.map[:, 2] > 0]

        # calculate the distance of each point in the uncertainty map from the ego position
        ego_dist = np.sqrt(np.sum((occ_map[:, :2] - self.vis_module.ego_pos) ** 2, axis=1))

        # only consider points that are closer than range in m
        occ_map = occ_map[ego_dist < self.range]

        if len(occ_map) == 0:
            return None, None, None

        for traj in trajectories:
            # get x and y coordinates from trajectory
            traj_coords = np.array([traj.cartesian.x, traj.cartesian.y]).T

            # calculate the distance weights specified with dist_mode --> 1 / (distance^dist_mode)
            distance_weights = np.power(cdist(traj_coords, occ_map[:, :2]), -dist_mode)
            distance = 1 / distance_weights

            # only consider points where the distance is smaller than max_dist (2.5m)
            distance_weights[distance_weights < np.power(max_dist, -dist_mode)] = 0

            # limit min distance to min_dist (0.25m) to avoid large factors in weight calculation (1/r^dist_mode)
            distance_weights[distance_weights > np.power(min_dist, -dist_mode)] = np.power(min_dist, -dist_mode)
            distance_weights *= w_distance
            distance[distance > max_dist] = 0

            # calculate the costs based on the distance weights and the occlusion values
            cost = np.matmul(distance_weights, occ_map[:, 2])
            cost_vector = cost

            # if the velocity shall be considered, the distance costs are multiplied with the corresponding speed
            velocity_with_w = w_velocity * np.array(traj.cartesian.v).T
            cost = np.matmul(cost, velocity_with_w)

            # calculate length of trajectory
            traj_length_total = np.sum(np.sqrt(np.sum(np.diff(traj_coords, axis=0) ** 2, axis=1)))
            traj_length_occ = np.sum(np.sqrt(np.sum(np.diff(traj_coords[cost_vector > 0], axis=0) ** 2, axis=1)))
            if traj_length_occ > 0:
                traj_v_occ = np.mean(traj.cartesian.v[cost_vector > 0])
            else:
                traj_v_occ = np.nan

            # normalize costs with trajectory length near to occlusion
            # if traj_length_occ > 0:
            #     cost = cost/(traj_length_occ/3)
            # else:
            #     cost = 0

            # save costs
            costs.append(cost)
            length.append([traj_length_total, traj_length_occ])
            traj_v_speeds.append(traj_v_occ)

        # return if all costs are 0
        if max(costs) == 0:
            return costs, None, None

        # normalize costs from 0 to max_costs using z transformation
        costs_norm_z = ohf.normalize_costs_z(costs, max_costs=self.max_cost)

        # debug plot costs
        # self.occ_plot.debug_trajectory_point_distances(occ_uncertainty_map, traj, traj_coords, distance, distance_weights)

        return costs_norm_z, length, traj_v_speeds

# eof
