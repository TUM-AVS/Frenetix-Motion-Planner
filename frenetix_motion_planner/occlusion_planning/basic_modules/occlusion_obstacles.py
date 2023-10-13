__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
This module contains obstacle types used in the occlusion planning

"""

# imports
import numpy as np
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as ohf
import frenetix_motion_planner.occlusion_planning.utils.vis_helper_functions as vhf
from shapely.geometry import Point, Polygon, LineString

# commonroad imports
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState, CustomState
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

# helper for tv collision object creation
from risk_assessment.helpers.collision_helper_function import create_tvobstacle


class OcclusionObstacle:
    def __init__(self, obst):
        self.obstacle_id = obst.obstacle_id
        self.obstacle_type = obst.obstacle_type
        self.obstacle_role = obst.obstacle_role.name
        self.obstacle_shape = obst.obstacle_shape
        self.initial_time_step = obst.initial_state.time_step
        self.first_time_visible = None
        self.first_time_visible_distance_ego = None
        self.visibility_time = 0
        self.visible_at_timestep = False
        self.last_visible_timestep = None
        self.visible_in_driving_direction = None

        if self.obstacle_role == "STATIC":
            self.pos = obst.initial_state.position
            self.current_pos = obst.initial_state.position
            self.orientation = obst.initial_state.orientation
            self.corner_points = vhf.calc_corner_points(self.pos, self.orientation, self.obstacle_shape)
            self.polygon = Polygon(self.corner_points)
            self.pos_point = Point(self.pos)
        else:
            self.pos = [state.position for state in obst.prediction.trajectory.state_list]
            self.current_pos = obst.initial_state.position
            self.orientation = [state.orientation for state in obst.prediction.trajectory.state_list]
            # corner points and polygon will be updated each timestep, no history saved
            self.corner_points = vhf.calc_corner_points(self.pos[0], self.orientation[0], self.obstacle_shape)
            self.polygon = Polygon(self.corner_points)
            self.pos_point = Point(self.pos[0])

    def update_at_timestep(self, time_step):
        if not self.obstacle_role == "STATIC" and self.pos is not None:
            try:
                ts = time_step - self.initial_time_step
                self.corner_points = vhf.calc_corner_points(self.pos[ts], self.orientation[ts],
                                                            self.obstacle_shape)
                self.polygon = Polygon(self.corner_points)
                self.pos_point = Point(self.pos[ts])
                self.current_pos = self.pos[ts]
            except IndexError:
                self.pos = None
                self.current_pos = None
                self.orientation = None
                self.corner_points = None
                self.polygon = None
                self.pos_point = None

        self.visible_at_timestep = False

        # needed for phantom pedestrian creation - will be calculated in OccPhantomModule
        self.visible_in_driving_direction = None


class OccBasicObstacle:
    """
    OccBasicObstacles are similar to OcclusionObstacles, but contain less information. They are the super class to
    EstimationObstacle and OccPhantomObstacle.
    """
    def __init__(self, obst_id, pos, orientation, length, width):
        self.obstacle_id = obst_id
        self.pos = pos
        self.orientation = orientation
        self.length = length
        self.width = width

        # create rectangle object from commonroad.geometry.shape (vertices are needed for further calculation)
        self.shape = Rectangle(length, width, center=np.array([0.0, 0.0]), orientation=0.0)

        self.corner_points = vhf.calc_corner_points(self.pos, self.orientation, self.shape)
        self.polygon = Polygon(self.corner_points)
        self.pos_point = Point(self.pos)


class EstimationObstacle(OccBasicObstacle):
    """
    obstacles are essential for accurately estimating visible and occluded areas.
    These obstacles are represented at their predicted positions at timestep x. These obstacles have to be created,
    so that the already implemented function calc_visible_area_from_obstacle_occlusions can be reused.
    """

    def __init__(self, obst_id, prediction, timestep):
        pos = prediction['pos_list'][timestep]
        orientation = prediction['orientation_list'][timestep]
        length = prediction['shape']['length']
        width = prediction['shape']['width']

        super().__init__(obst_id, pos, orientation, length, width)


class OccPhantomObstacle(OccBasicObstacle):
    def __init__(self, phantom_id, pos, orientation, length=0.3, width=0.5, vector=None, s=None,
                 calc_ped_traj_polygons=False, create_cr_obst=False, time_step=0):

        # initialize phantom obstacle using OccBasicObstacle class
        super().__init__(phantom_id, pos, orientation, length, width)

        # assign additional variables
        self.time_step = time_step
        self.calc_ped_traj_polygons = calc_ped_traj_polygons
        self.create_cr_obst = create_cr_obst
        self.orientations = []
        self.orientation_vector = vector
        self.v = 0
        self.trajectory = None
        self.goal_position = None
        self.trajectory_length = None
        self.traj_polygons = None
        self.s = s
        self.diag = np.sqrt(np.power(length, 2) + np.power(width, 2))

        # create commonroad like variables for further use (e.g. collision checking, harm estimation)
        self.commonroad_dynamic_obstacle = None
        self.cr_collision_object = None
        self.cr_tv_collision_object = None
        self.commonroad_predictions = None

    def create_cr_predictions(self):
        self._create_cr_predictions()
        return self.commonroad_predictions

    def _create_cr_predictions(self):
        # create dict like a commonroad prediction (e.g. walenet) -> needed in harm estimation
        shape = {'length': self.commonroad_dynamic_obstacle.obstacle_shape.length * 1.2,
                 'width': self.commonroad_dynamic_obstacle.obstacle_shape.width * 1.2}

        trajectory_ped = self.commonroad_dynamic_obstacle.prediction.trajectory.state_list

        self.commonroad_predictions = {'orientation_list': np.array([states.orientation for states in trajectory_ped]),
                                       'v_list': np.array([states.velocity for states in trajectory_ped]),
                                       'pos_list': np.array([states.position for states in trajectory_ped]),
                                       'shape': shape,
                                       'cov_list': np.full([len(trajectory_ped), 2, 2], 0.2)}

    def _create_cr_collision_object(self):
        # function is currently not active
        # can be used by collision = tvo.collide(cr_collision_object)
        self.cr_collision_object = create_collision_object(self.commonroad_dynamic_obstacle)

    def _create_cr_tv_collision_object(self):
        # create pycrcc time variant collision object for dynamic collision checking
        self.cr_tv_collision_object = create_tvobstacle(traj_list=np.array([self.trajectory[:, 0],
                                                                            self.trajectory[:, 1],
                                                                            self.orientations]).transpose().tolist(),
                                                        box_length=self.shape.length / 2,
                                                        box_width=self.shape.width / 2,
                                                        start_time_step=0)

    def create_cr_obstacle(self, time_step=0):
        # create commonroad dynamic obstacle for harm estimation (also for further usage if needed)
        # create initial state
        initial_state = InitialState(time_step=time_step,
                                     position=self.pos,
                                     orientation=self.orientation,
                                     velocity=self.v,
                                     acceleration=0.0,
                                     yaw_rate=0.0,
                                     slip_angle=0.0)

        # initialize state list and append custom states
        state_list = []
        for i in range(0, len(self.trajectory)):
            custom_state = CustomState(orientation=self.orientation,
                                       velocity=self.v,
                                       position=self.trajectory[i],
                                       time_step=time_step + i)

            state_list.append(custom_state)

        # create trajectory from state list
        trajectory = Trajectory(initial_time_step=time_step, state_list=state_list)

        # create trajectory prediction from trajectory
        trajectory_prediction = TrajectoryPrediction(trajectory=trajectory,
                                                     shape=self.shape)

        # combine all information to commonroad dynamic obstacle
        self.commonroad_dynamic_obstacle = DynamicObstacle(obstacle_id=self.obstacle_id,
                                                           obstacle_type=ObstacleType('pedestrian'),
                                                           obstacle_shape=self.shape,
                                                           initial_state=initial_state,
                                                           prediction=trajectory_prediction)

    def set_velocity(self, v):
        self.v = v

    def calc_trajectory(self, dt=0.1, duration=None):
        """
        Computes the positions reached by an object in a given time,
        based on its starting point, orientation, speed, and time step.

        dt: time step
        duration: time duration

        return: numpy array with the positions reached by the object
        """
        # load variables
        v = self.v
        orientation = self.orientation
        start_point = self.pos

        # calculate duration
        if duration is None:
            duration = self.trajectory_length / v

        # Compute the x and y components of the velocity
        vx = round(v * np.cos(orientation), 3)
        vy = round(v * np.sin(orientation), 3)

        # Compute the number of time steps
        num_steps = int(duration / dt) + 1

        # Create a matrix of time steps and velocities
        t = np.arange(num_steps + 1)[:, np.newaxis] * dt
        v_matrix = np.array([vx, vy])

        # Compute the trajectory
        trajectory = start_point + t * v_matrix.T

        self.trajectory = trajectory

        # calc ped orientation array
        self.orientations = np.ones(len(self.trajectory)) * self.orientation

        if self.calc_ped_traj_polygons:
            self.calc_traj_polygons()

        if self.create_cr_obst:
            self.create_cr_obstacle(time_step=self.time_step)
            self._create_cr_predictions()
            self._create_cr_tv_collision_object()

    def calc_goal_position(self, sidewalk):
        # define length of linestring (only used to calculate final destination)
        length = 10

        # calculate start side of linestring
        start_x = self.pos_point.x + length * np.cos(np.pi + self.orientation)
        start_y = self.pos_point.y + length * np.sin(np.pi + self.orientation)
        start_point = Point(start_x, start_y)

        # calculate end of linestring
        end_x = self.pos_point.x + length * np.cos(self.orientation)
        end_y = self.pos_point.y + length * np.sin(self.orientation)
        end_point = Point(end_x, end_y)

        # create linestring
        # line = LineString([self.pos_point, end_point])
        line = LineString([start_point, end_point])

        # calc intersection between line and sidewalk (goal position)
        goal_position = line.intersection(sidewalk.interiors)[0]

        # use further point if geom type is MultiPoint
        if goal_position.geom_type == 'MultiPoint':

            # convert multipoint to list of np.arrays
            points = [np.array(p.coords).flatten() for p in goal_position.geoms]

            # calc distances between points and ego pos
            distances = np.linalg.norm(np.stack(points) - self.pos, axis=1)

            # find index of max distance
            max_distance_index = np.argmax(distances)

            # assign point to object
            self.goal_position = points[max_distance_index]
        else:
            self.goal_position = np.array(goal_position.coords).flatten()

        # update orientation
        vector = self.goal_position - self.pos
        self.orientation = vhf.angle_between_positive(np.array([1, 0]), vector)

        # calculate length of trajectory
        self.trajectory_length = np.linalg.norm(self.goal_position - self.pos)

    def calc_traj_polygons(self):

        # calculate pedestrian trajectory polygons for each timestep
        self.traj_polygons = ohf.compute_vehicle_polygons(self.trajectory[:, 0], self.trajectory[:, 1],
                                                          self.orientations, self.width, self.length)

# eof
