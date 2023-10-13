__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
This module is the main module of the Phantom obstacle generation of the reactive planner.

"""

# imports
import numpy as np
import logging
from scipy.spatial import distance
from shapely.geometry import Point, LineString
import copy

# commonroad imports
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as ohf
import frenetix_motion_planner.occlusion_planning.utils.vis_helper_functions as vhf
from frenetix_motion_planner.occlusion_planning.basic_modules.occlusion_obstacles import OccPhantomObstacle

# risk assessment and harm imports
from risk_assessment.risk_costs import calc_risk

# get logger
msg_logger = logging.getLogger("Message_logger")


class OccPhantomModule:
    def __init__(self, config=None, occ_scenario=None, vis_module=None,
                 occ_visible_area=None, occ_plot=None, params_risk=None, params_harm=None):
        self.config = config.occlusion
        self.ego_vehicle_params = config.vehicle
        self.occ_scenario = occ_scenario
        self.vis_module = vis_module
        self.occ_visible_area = occ_visible_area
        self.occ_plot = occ_plot
        self.params_risk = params_risk
        self.params_harm = params_harm
        self.max_number_of_phantom_peds = 2
        self.ped_width = 0.5
        self.ped_length = 0.3
        self.ped_velocity = 1.2
        self.sorted_static_obstacles = None
        self.phantom_peds = None
        self.added_phantom = {}
        self.reactivate_at_s = None
        self.ego_pos_s = None
        self.max_s_trajectories = None
        self.costs = None
        self.max_costs = 10
        self.w_risk = 3
        self.w_harm = 25
        self.max_accepted_harm = 0
        self.max_accepted_risk = 0
        self.last_phantom_id = None

        # variable to store commonroad like predictions
        self.cr_predictions = dict()

        # calculate the maximum distance from a vehicle corner point to the vehicle center point
        self.ego_diag = np.sqrt(
            np.power(self.occ_scenario.ego_length, 2) + np.power(self.occ_scenario.ego_width, 2))

    ################################
    # Main Function of phantom module
    ################################

    def evaluate_trajectories(self, trajectories, max_harm=0.5, plot_traj=False, plot_ped=False):

        # calculate spawn points and create phantom obstacles
        self._evaluation_preprocessing(trajectories)

        # if no phantom obstacles exist
        if self.phantom_peds is None or len(self.phantom_peds) == 0:
            return np.zeros(len(trajectories))

        # calc harm for each trajectory
        harm, risk, count_invalid_trajectories = self._calc_harm(trajectories, max_harm)

        # debug message
        msg_logger.debug('Rejected {} invalid trajectories due to phantom pedestrian harm'
                         .format(count_invalid_trajectories))

        # calc_costs
        self.costs = np.array(self._calc_costs(trajectories, harm, risk))

        # plot phantom peds and their trajectories if plot is activated
        if plot_ped and self.occ_plot is not None:
            self.occ_plot.plot_phantom_ped_trajectory(self.phantom_peds)

        # visualize costs
        if plot_traj and self.occ_plot is not None:
            self.occ_plot.plot_trajectories_cost_color(trajectories, self.costs, harm=harm, risk=risk)

        return self.costs

    def analyze_trajectory_harm(self, trajectory):
        # function to analyze the accepted risk and harm of the selected trajectory
        if self.phantom_peds is not None and len(self.phantom_peds) > 0:

            # calc risk and harm,
            ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, ego_risk, obst_risk, obst_harm_occ = \
                calc_risk(scenario=self,
                          traj=trajectory,
                          predictions=self.cr_predictions,
                          vehicle_params=self.ego_vehicle_params,
                          params_risk=self.params_risk,
                          params_harm=self.params_harm,
                          ego_id=24,
                          ego_state=self.vis_module.ego_state
                          )

            # store values in object
            self.max_accepted_harm = obst_harm_occ
            self.max_accepted_risk = obst_risk

            # plot final trajectory
            if self.occ_plot is not None:
                self.occ_plot.plot_trajectories(trajectory)

        else:
            self.max_accepted_harm = 0
            self.max_accepted_risk = 0

    def add_phantom_ped_to_cr(self, cr_scenario, planner, min_ped_distance=10):
        condition = any(abs(self.ego_pos_s - value) <= 1 for value in self.config.create_real_pedestrians)
        ped = None

        if condition and self.phantom_peds is not None and len(self.phantom_peds) > 0:

            # search for candidate that has desired distance to already added ped
            for candidate in self.phantom_peds:
                if len(self.added_phantom) == 0 or not any(abs(v['ped'].s - candidate.s) <= min_ped_distance for
                                                           v in self.added_phantom.values()):
                    ped = candidate
                    break

            # return if no suited pedestrian could be found
            if ped is None:
                return

            # add phantom pedestrian (dynamic obstacle) to commonroad scenario
            cr_scenario.add_objects(ped.commonroad_dynamic_obstacle)

            # add object to vis_module obstacles for visible_area_calculation
            self.vis_module.add_occ_obstacle(ped.commonroad_dynamic_obstacle)

            # add to added_phantoms
            self.added_phantom[ped.obstacle_id] = {'ped': copy.deepcopy(ped),
                                                   'added_at_timestep': self.vis_module.time_step}

            # update collision checker (otherwise collision with pedestrian won't be recognized)
            planner.set_collision_checker(cr_scenario)

            msg_logger.debug("Phantom Pedestrian {} added to scenario at timestep {} at position  s {}."
                  .format(ped.obstacle_id, self.vis_module.time_step, ped.s))

    def _calc_costs(self, trajectories, harm, risk):
        # calc trajectory costs based on risk and harm
        costs = []
        for i in range(0, len(trajectories)):
            # risk is almost 0 -> costs are 0
            if np.isclose(risk[i], 0):
                costs.append(0)
            else:
                # calc costs depending on risk and potential harm
                cost = self.w_risk * np.power(2 * risk[i], 2) + self.w_harm * np.power(2 * harm[i], 2)
                costs.append(cost)

        return costs

    def _calc_harm(self, trajectories, max_harm, plot_risk_harm=False):
        # calc risk and harm of each trajectory using the risk model
        # init temp variables
        count_invalid_trajectories = 0
        traj_harm = []
        traj_risk = []

        # create commonroad like predictions for harm estimation
        self._combine_commonroad_predictions()

        # iterate over trajectories and calc harm
        for traj in trajectories:

            # calc risk and harm
            ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, ego_risk, obst_risk, obst_harm_occ = \
                calc_risk(scenario=self,
                          traj=traj,
                          predictions=self.cr_predictions,
                          vehicle_params=self.ego_vehicle_params,
                          params_risk=self.params_risk,
                          params_harm=self.params_harm,
                          ego_id=24,
                          ego_state=self.vis_module.ego_state
                          )

            # find maximum risk and harm of trajectory
            traj_harm.append(obst_harm_occ)
            traj_risk.append(obst_risk)

            # store harm information in trajectory sample
            setattr(traj, 'harm_occ_module', obst_harm_occ)

            # if risk is greater than max harm, reject trajectory
            if obst_harm_occ > max_harm:
                setattr(traj, 'valid', False)
                count_invalid_trajectories += 1

            if self.occ_plot is not None and plot_risk_harm:
                # show risk and harm in plot
                label = None
                if not np.isclose(obst_risk, 0):
                    label = "obst_risk_max: " + str(round(obst_risk, 3)) + \
                            " -- obst_harm_max: " + str(round(max(obst_harm_max.values()), 3))
                self.occ_plot.plot_trajectories(traj, label=label)

        return traj_harm, traj_risk, count_invalid_trajectories

    ################################
    # create phantom obstacles
    ################################

    def _evaluation_preprocessing(self, trajectories):

        # only consider static obstacles (ped won't be in front of a moving car)
        self.sorted_static_obstacles = [obst for obst in self.vis_module.obstacles if obst.obstacle_role == 'STATIC']

        # quit here, if no static obstacle exists
        if len(self.sorted_static_obstacles) == 0:
            return

        # sort obstacles by distance to ego pos
        self.sorted_static_obstacles = sorted(self.sorted_static_obstacles, key=lambda obstacle:
                                              distance.euclidean(obstacle.pos, self.vis_module.ego_pos))

        # find max s in trajectories
        self.max_s_trajectories = ohf.find_max_s_in_trajectories(trajectories)

        # check, which obstacles area along the driving direction
        self._check_obstacles()

        # find spawn points (dict with xy and cl) and points on ref path for phantom pedestrians
        spawn_points = self._find_spawn_points()

        # create phantom pedestrians
        self._create_phantom_peds(spawn_points)

        # calculate trajectories of phantom peds
        self._calc_trajectories()

    def _create_phantom_peds(self, spawn_points):
        # create empty list for phantom pedestrians
        if self.last_phantom_id is None:
            last_id = 110
        else:
            last_id = self.last_phantom_id + 1

        self.last_phantom_id = last_id

        phantom_peds = []

        # iterate over spawn points
        for i, spawn_point in enumerate(spawn_points):
            # create vector to calculate orientation
            vector = spawn_point['rp_xy'] - spawn_point['xy']
            # calculate orientation with vector and reference vector [1, 0]
            orientation = vhf.angle_between_positive(np.array([1, 0]), vector)

            # find needed parameters for phantom ped
            create_cr_obst = True

            # create phantom pedestrian and add to list
            ped = OccPhantomObstacle(i + last_id, spawn_point['xy'], orientation, self.ped_length,
                                     self.ped_width, vector, s=spawn_point['cl'][0],
                                     time_step=self.vis_module.time_step,
                                     calc_ped_traj_polygons=False,  # ToDO remove
                                     create_cr_obst=create_cr_obst)

            # last check if spawn point is valid
            if ped.polygon.intersects(self.vis_module.visible_area_timestep.buffer(-0.1)) or not \
                    ped.polygon.within(self.occ_scenario.lanelets_combined_with_sidewalk.buffer(0.1)):
                msg_logger.debug('removed spawn point')
                continue

            phantom_peds.append(ped)

        # assign local variable to object
        self.phantom_peds = phantom_peds

    def _calc_trajectories(self):
        for ped in self.phantom_peds:
            ped.set_velocity(self.ped_velocity)

            ped.calc_goal_position(self.occ_scenario.sidewalk_combined)
            ped.calc_trajectory(dt=self.occ_scenario.dt)

    def _check_obstacles(self):

        for obst in self.sorted_static_obstacles:
            if obst.polygon.intersects(self.occ_visible_area.poly.buffer(0.01)):
                obst.visible_in_driving_direction = True
            else:
                obst.visible_in_driving_direction = False

    ################################
    # find spawn points
    ################################

    def _find_spawn_points(self, shift_spawn_points=True):

        self.ego_pos_s = self.occ_scenario.ref_path_ls.project(Point(self.vis_module.ego_pos))
        spawn_points = []

        # iterate over obstacles (sorted by distance to ego position)
        for obst in self.sorted_static_obstacles:

            # if the obstacle is visible (and not behind the vehicle), find relevant corner points
            if obst.visible_in_driving_direction:

                # find relevant corner points (cp) for each obstacle and the corresponding vectors
                # to ref path (needed later)
                relevant_cp = self._spawn_point_preprocessing(obst, offset=self.ped_width / 2 * 1.2)

                # calc curvilinear coordinates for relevant_cp and remove points that are too far away
                relevant_cp = self._convert_to_cl_and_remove_far_points(relevant_cp)

                # check if more than one possible spawn point exists for each obstacle
                if len(relevant_cp) > 1:

                    # if more than one relevant corner point exists, find critical spawn point and vector_to_ref_path
                    spawn_point = self._find_relevant_spawn_point(relevant_cp)

                    # add spawn point to spawn_points array
                    spawn_points.append(spawn_point)

                elif len(relevant_cp) == 1:
                    # only one relevant corner exists -> add relevant corner to spawn_points array
                    spawn_points.append(relevant_cp[0])

                else:
                    continue

        # filter and sort spawn points according to their s coordinate
        sorted_spawn_points = self._remove_close_spawn_points(spawn_points, min_s_distance=3)

        # limit spawn points to max_number_of_phantom_peds
        spawn_points = sorted_spawn_points[:self.max_number_of_phantom_peds]

        # shift spawn_points to border of visible and occluded area if activated
        if shift_spawn_points and len(spawn_points) > 0:
            ref_path_ls = self.occ_scenario.ref_path_ls
            for spawn_point in spawn_points:
                # create line from spawn point to point on ref path to find intersection with visible area
                line = np.stack([spawn_point['xy'], spawn_point['rp_xy']])
                line_ls = LineString(line)

                # find shifted spawn point as shapely point
                spawn_point_shifted_point = self.vis_module.visible_area_timestep.buffer(self.ped_length/2 * 1.2). \
                    exterior.intersection(line_ls)

                # try to shift the point again
                if spawn_point_shifted_point.is_empty:
                    length = line_ls.length
                    extension_length = 2
                    vector = spawn_point['rp_xy'] - spawn_point['xy']
                    end_point = np.array([spawn_point['rp_xy'][0] + vector[0] * (length + extension_length),
                                          spawn_point['rp_xy'][1] + vector[1] * (length + extension_length)])

                    extended_line = np.stack([spawn_point['rp_xy'], end_point])
                    line_ls = LineString(extended_line)

                    # find shifted spawn point as shapely point
                    spawn_point_shifted_point = self.vis_module.visible_area_timestep.buffer(
                        self.ped_length / 2 * 1.2).exterior.intersection(line_ls)

                # if more than one shifted spawn point is available (happens when visible area border is close to
                # original spawn point) use the point that is closer to the reference path
                if spawn_point_shifted_point.geom_type == "MultiPoint":
                    dist = []

                    # create shapely point for point on ref path
                    point_rp = Point(spawn_point['rp_xy'])

                    # calc distance to each point and find index of min distance
                    dist = [point_rp.distance(point) for point in spawn_point_shifted_point.geoms]
                    min_dist_index = np.argmin(dist)

                    # select point
                    spawn_point_shifted_point = spawn_point_shifted_point.geoms[min_dist_index]

                # if no shifted spawn point could be found, use spawn point at corner
                if not spawn_point_shifted_point.is_empty:
                    # calc curvilinear coordinates of shifted spawn point
                    s = ref_path_ls.project(spawn_point_shifted_point)
                    d = ref_path_ls.distance(spawn_point_shifted_point)

                    # overwrite values in spawn point
                    spawn_point['xy'] = np.array(spawn_point_shifted_point.coords).flatten()
                    spawn_point['cl'] = np.array([s, d])

        return spawn_points

    def _convert_to_cl_and_remove_far_points(self, relevant_cp):

        # initialize variables
        ref_path_ls = self.occ_scenario.ref_path_ls
        relevant_cp_new = []

        # iterate through corner points and calculate curvilinear s and d coordinate for each corner
        for cp in relevant_cp:
            s = ref_path_ls.project(Point(cp['xy']))
            d = ref_path_ls.distance(Point(cp['xy']))

            # only add points that can be reached by a trajectory and that are not to close to ego vehicle
            if not s >= self.max_s_trajectories and not s < self.ego_pos_s + 4:
                relevant_cp_dict = {'xy': cp['xy'], 'cl': np.array([s, d]), 'rp_xy': cp['rp_xy']}
                relevant_cp_new.append(relevant_cp_dict)

        return relevant_cp_new

    def _find_relevant_spawn_point(self, relevant_cp) -> dict:
        """
        This function uses the list of relevant corner points and finds the most critical spawn point for a phantom
        pedestrian. Only one spawn point will be determined for each obstacle.
        Args:
            relevant_cp: list of dicts with relevant corner points {'xy' - corner points in xy coordinates,
                                                                    'cl' - corner points in curvilinear coordinates,
                                                                    'rp_xy' - point on ref path in xy coordinates}


        Returns: dict of spawn point {xy, cl, rp_xy}

        """
        # convert list of curvilinear coordinates to numpy array [s,d]
        relevant_cp_cl = np.array([d['cl'] for d in relevant_cp])

        # threshold, that points are "close" (only s considered)
        threshold = 0.2

        # calc distance of s coordinates between corner_cl and ego pos (projected on ref path)
        distances = abs(relevant_cp_cl[:, 0] - self.ego_pos_s)

        # find index of the closest point to ego pos (only s considered)
        min_distance_index = np.argmin(distances)

        # check whether there is a point, that has a similar s coordinate
        similar_indices = np.where(np.isclose(distances, distances[min_distance_index], rtol=0, atol=threshold))[0]

        # if two points have similar s coordinates, the point with the smaller d coordinate shall be used
        if len(similar_indices) > 1:
            # find point with smaller d coordinate
            min_d_index = np.argmin(relevant_cp_cl[similar_indices, 1])

            # update index
            min_distance_index = similar_indices[min_d_index]

        # assign spawn point
        spawn_point = relevant_cp[min_distance_index]

        return spawn_point

    def _spawn_point_preprocessing(self, obst, offset=0.25) -> list:
        """
        This function calculates relevant corner points and returns them as a list (for each obstacle)
        Args:
            obst: obstacle (one obstacle of type OcclusionObstacle)

        Returns: - list of dicts with relevant corner points [xy, cl, rp](potential spawn points for phantom peds)

        """
        # load ref path linestring from occ scenario
        ref_path_ls = self.occ_scenario.ref_path_ls

        # initialize list for relevant corner points
        relevant_corner_points = []

        # if offset shall be used, calculate corner points from polygon, otherwise use corner points of obstacle
        if offset > 0:
            corners = np.array(obst.polygon.buffer(offset, join_style=2).exterior.coords[:-1])
        else:
            corners = obst.corner_points

        # iterate over corner points
        for corner in corners:

            # prepare needed variables to calculate points and vectors
            cp = Point(corner)
            point_on_ref_path = np.array(ref_path_ls.interpolate(ref_path_ls.project(cp)).coords).flatten()
            vector = point_on_ref_path - corner
            normal_vector = ohf.unit_vector(vector)

            # calc points towards and in the opposite direction to ref path
            point_to_rp = corner + 0.1 * normal_vector
            point_opp_rp = corner - 0.1 * normal_vector

            # if the opposite point is within the visible area, the corner is not relevant
            if Point(point_opp_rp).within(self.vis_module.visible_area_timestep.buffer(0.01)):
                continue

            # if the opposite point is within the obstacle, the corner is not relevant
            if Point(point_opp_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point towards the ref path is within the obstacle, the corner is not relevant
            if Point(point_to_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point is outside the lanelet network (including sidewalk), the corner is not relevant
            if not Point(point_opp_rp).within(self.occ_scenario.lanelets_combined_with_sidewalk):
                continue

            # check if phantom pedestrian would be within another obstacle
            if cp.buffer(self.ped_width / 2).intersects(self.vis_module.obstacles_polygon):
                continue

            # if corner point is a potential spawn point for phantom peds, add to list
            relevant_cp_dict = {'xy': corner, 'cl': None, 'rp_xy': point_on_ref_path}
            relevant_corner_points.append(relevant_cp_dict)

        return relevant_corner_points

    ################################
    # Functions for harm model
    ################################

    def obstacle_by_id(self, obstacle_id):
        for ped in self.phantom_peds:
            if ped.obstacle_id == obstacle_id:
                return ped.commonroad_dynamic_obstacle

        return None

    def _combine_commonroad_predictions(self):
        # clear prediction dict
        self.cr_predictions = {}

        # add ped predictions to dict
        for ped in self.phantom_peds:
            self.cr_predictions[ped.obstacle_id] = ped.commonroad_predictions

    ################################
    # helper functions
    ################################

    def _remove_close_spawn_points(self, spawn_points, min_s_distance=3):
        # function to remove close spawn_points
        sorted_spawn_points = sorted(spawn_points, key=lambda x: x['cl'][0])
        filtered_spawn_points = []

        # iterate over spawn points and use first spawn point, that is not close to "real"
        # phantom ped (added to scenario)
        for i in range(0, len(sorted_spawn_points)):
            cur_s = sorted_spawn_points[i]['cl'][0]
            close_to_added_phantom = any(abs(v['ped'].s - cur_s) <= 10 for v in self.added_phantom.values())
            if not close_to_added_phantom:
                filtered_spawn_points.append(sorted_spawn_points[i])
                index = i + 1
                break

        # if no point could be found return empty list
        if len(filtered_spawn_points) == 0:
            return []

        # iterate over remaining spawn points
        for i in range(index, len(sorted_spawn_points)):
            current_point = sorted_spawn_points[i]
            previous_point = filtered_spawn_points[-1]

            # calc s distance
            dist = current_point['cl'][0] - previous_point['cl'][0]

            # if distance is larger than min distance -> add to list
            if dist >= min_s_distance:
                filtered_spawn_points.append(current_point)

        return filtered_spawn_points
