__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import logging

import frenetix_motion_planner.occlusion_planning.utils.vis_helper_functions as vhf
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as ohf
from frenetix_motion_planner.occlusion_planning.basic_modules.occlusion_obstacles import OcclusionObstacle
import numpy as np

# get logger
msg_logger = logging.getLogger("Message_logger")


class VisibilityModule:
    def __init__(self, scenario=None, lanelets=None, sensor_radius=50, occlusions=True, wall_buffer=0.0):
        self.ego_state = None
        self.lanelets = lanelets
        self.sensor_radius = sensor_radius
        self.occlusions = occlusions
        self.wall_buffer = wall_buffer
        self.obstacles = convert_obstacles(scenario.obstacles)
        self.ego_pos = None
        self.time_step = None
        self.visible_area_timestep = None
        self.visible_objects_timestep = None
        self.obstacles_polygon = None

    def update_obstacles_at_time_step(self, time_step):
        for obst in self.obstacles:
            obst.update_at_timestep(time_step)

    def get_visible_area_and_objects(self, time_step, ego_state=None, debug=True):

        if ego_state is None:
            raise ValueError("Ego state must be provided for the calculation of visible area and visible objects!")

        # initialize ego position and current time step
        self.ego_state = ego_state
        self.ego_pos = ego_state.initial_state.position
        self.time_step = time_step

        # initialize list to store visible obstacles
        visible_objects_timestep = []

        # update corner points and polygons for dynamic obstacles
        if self.time_step > 0:  # initial values are already set
            self.update_obstacles_at_time_step(self.time_step)

        # calculate visible area only considering the lanelet geometry
        visible_area = vhf.calc_visible_area_from_lanelet_geometry(self.lanelets, self.ego_pos, self.sensor_radius)

        # if obstacle occlusions shall be considered, subtract polygon from visible_area
        if self.occlusions:

            # update visible_area due to obstacle occlusion
            visible_area, obst_polygon = vhf.calc_visible_area_from_obstacle_occlusions(visible_area, self.ego_pos,
                                                                                        self.obstacles,
                                                                                        self.sensor_radius,
                                                                                        return_shapely_obstacles=True)

            # assign multipolygon of all obstacles to variable (needed for phantom pedestrian calculation)
            self.obstacles_polygon = obst_polygon

        # get visible obstacles and add to list
        visible_area_check = visible_area.buffer(0.01, join_style=2)
        for obst in self.obstacles:
            if obst.pos is not None:
                if obst.polygon.intersects(visible_area_check):
                    visible_objects_timestep.append(obst.obstacle_id)
                    obst.visible_at_timestep = True

                    if obst.last_visible_timestep != self.time_step:
                        obst.visibility_time += 1

                    obst.last_visible_timestep = self.time_step

                    # set time step when obstacle was visible for the first time
                    if obst.first_time_visible is None:
                        obst.first_time_visible = self.time_step
                        obst.first_time_visible_distance_ego = np.linalg.norm(obst.current_pos - self.ego_pos)
                        msg_logger.debug("Obstacle {} first time visible at timestep {} with distance {} from ego"
                                  .format(obst.obstacle_id, self.time_step, obst.first_time_visible_distance_ego))

                else:
                    obst.visibility_time = 0

                # print visibility time
                if obst.visibility_time != 0 and debug:
                    msg_logger.debug("Obstacle {} visible for {} timesteps ".format(obst.obstacle_id, obst.visibility_time))

        # remove linestrings from visible_area
        visible_area = ohf.remove_unwanted_shapely_elements(visible_area)

        # remove unwanted areas if visible_area is multipolygon
        if visible_area.geom_type == "MultiPolygon":
            visible_area = ohf.remove_small_areas(visible_area)

        # save visible_objects and visible area in VisibilityModule object
        self.visible_objects_timestep = visible_objects_timestep
        self.visible_area_timestep = visible_area

        return visible_objects_timestep, visible_area

    def add_occ_obstacle(self, obstacle):
        new_occ_obstacle = OcclusionObstacle(obstacle)
        self.obstacles.append(new_occ_obstacle)


def convert_obstacles(obstacles):
    occ_obstacles = []
    for obst in obstacles:
        occ_obst = OcclusionObstacle(obst)
        occ_obstacles.append(occ_obst)
    return occ_obstacles

