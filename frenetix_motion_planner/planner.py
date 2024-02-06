__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from abc import abstractmethod
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import math
import logging

from commonroad.planning.planning_problem import PlanningProblem, GoalRegion
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState, InputState
from commonroad.scenario.trajectory import Trajectory

from frenetix_motion_planner.state import ReactivePlannerState
from frenetix_motion_planner.sampling_matrix import SamplingHandler
from frenetix_motion_planner.utility.logging_helpers import DataLoggingCosts
from frenetix_motion_planner.trajectories import TrajectorySample

from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem, interpolate_angle
from cr_scenario_handler.utils import helper_functions as hf
from cr_scenario_handler.utils.collision_check import collision_check_prediction

from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum, \
                                                                          trajectories_collision_static_obstacles

from risk_assessment.risk_costs import calc_risk
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle

from frenetix_motion_planner.utility.load_json import (
    load_harm_parameter_json,
    load_risk_json
)

# get logger
msg_logger = logging.getLogger("Message_logger")


class Planner:

    def __init__(self, config_plan, config_sim, scenario: Scenario,
                 planning_problem: PlanningProblem,
                 log_path: str, work_dir: str, msg_logger):
        """Wrappers providing a consistent interface for different planners.
        To be implemented for every specific planner.

        :param config: Object containing the configuration of the planner.
        :param scenario: The commonroad scenario to be simulated.
        :param planning_problem: The PlanningProblem for this planner.
        :param log_path: Path the planner's log files will be written to.
        :param work_dir: Working directory for the planner.
        """
        self.config_plan, self.config_sim = config_plan, config_sim
        self.horizon = config_plan.planning.planning_horizon
        self.dT = config_plan.planning.dt
        self.N = int(config_plan.planning.planning_horizon / config_plan.planning.dt)
        self._check_valid_settings()
        self.vehicle_params = config_sim.vehicle
        self._low_vel_mode_threshold = config_plan.planning.low_vel_mode_threshold
        self.msg_logger = msg_logger
        # Multiprocessing & Settings
        self._multiproc = config_plan.debug.multiproc
        self._num_workers = config_plan.debug.num_workers

        # Initial State
        self.x_0: Optional[ReactivePlannerState] = None
        self.x_cl: Optional[Tuple[List, List]] = None
        self.reference_path = None

        self.record_state_list: List[ReactivePlannerState] = list()
        self.record_input_list: List[InputState] = list()

        self.ego_vehicle_history = list()
        self._LOW_VEL_MODE = False

        # Scenario
        self.coordinate_system = None
        self.scenario = None
        self.road_boundary = None
        self.set_scenario(scenario)
        self.planning_problem = planning_problem
        self.predictions = None
        self.reach_set = None
        self.behavior = None
        self.set_new_ref_path = None
        self.cost_function = None
        self.goal_status = False
        self.full_goal_status = None
        self.goal_area = None
        self.occlusion_module = None
        self.goal_message = "Planner is in time step 0!"

        self.desired_velocity = None
        self._desired_d = 0.
        # self.max_seen_costs = 1

        self.cost_weights = OmegaConf.to_object(config_plan.cost.cost_weights)

        # **************************
        # Extensions Initialization
        # **************************
        self.use_prediction = False
        self._collision_counter = 0

        # **************************
        # Statistics Initialization
        # **************************
        self._total_count = 0
        self._infeasible_count_collision = 0
        self._infeasible_count_kinematics = None
        self.infeasible_kinematics_percentage = None
        # self._optimal_cost = 0

        # **************************
        # Sampling Initialization
        # **************************
        # Set Sampling Parameters#
        self._sampling_min = config_plan.planning.sampling_min
        self._sampling_max = config_plan.planning.sampling_max
        self.sampling_handler = SamplingHandler(dt=self.dT, max_sampling_number=config_plan.planning.sampling_max,
                                                t_min=config_plan.planning.t_min, horizon=self.horizon,
                                                delta_d_max=config_plan.planning.d_max, delta_d_min=config_plan.planning.d_min)

        self.stopping_s = None

        # *****************************
        # Debug & Logger Initialization
        # *****************************
        self.log_risk = self.config_plan.debug.log_risk
        self.save_all_traj = self.config_plan.debug.save_all_traj
        self.all_traj = None
        self.optimal_trajectory = None
        self.trajectory_pair = None
        self.use_occ_model = False
        if config_plan.debug.activate_logging:
            self.logger = DataLoggingCosts(
                config=self.config_plan,
                scenario=scenario,
                planning_problem=planning_problem,
                path_logs=log_path,
                save_all_traj=self.save_all_traj,
                cost_params=self.config_plan.cost.cost_weights
            )
        else:
            self.logger = None
        self._draw_traj_set = self.config_plan.debug.draw_traj_set
        self._kinematic_debug = self.config_plan.debug.kinematic_debug

        # **************************
        # Risk & Harm Initialization
        # **************************
        self.params_harm = load_harm_parameter_json(work_dir)
        self.params_risk = load_risk_json(work_dir)

    @property
    def infeasible_count_collision(self):
        """Number of colliding trajectories"""
        return self._collision_counter

    def update_externals(self, scenario: Scenario = None, reference_path: np.ndarray = None,
                         planning_problem: PlanningProblem = None, goal_area: GoalRegion = None,
                         x_0: ReactivePlannerState = None, x_cl: Optional[Tuple[List, List]] = None,
                         cost_weights=None, occlusion_module=None, desired_velocity: float = None,
                         predictions=None, reach_set=None, behavior=None):
        """
        Sets all external information in reactive planner
        :param scenario: Commonroad scenario
        :param reference_path: reference path as polyline
        :param planning_problem: reference path as polyline
        :param goal_area: commonroad goal area
        :param x_0: current ego vehicle state in global coordinate system
        :param x_cl: current ego vehicle state in curvilinear coordinate system
        :param cost_weights: current used cost weights
        :param occlusion_module: occlusion module setup
        :param desired_velocity: desired velocity in mps
        :param predictions: external calculated predictions of other obstacles
        :param reach_set: external calculated reach_sets
        :param behavior: behavior planner setup
        """
        if scenario is not None:
            self.set_scenario(scenario)
        if reference_path is not None:
            self.reference_path = reference_path
            self.set_reference_and_coordinate_system(reference_path)
        if planning_problem is not None:
            self.set_planning_problem(planning_problem)
        if goal_area is not None:
            self.set_goal_area(goal_area)
        if x_0 is not None:
            self.set_x_0(x_0)
            self.set_x_cl(x_cl)
        if cost_weights is not None:
            self.set_cost_function(cost_weights)
        if occlusion_module is not None:
            self.set_occlusion_module(occlusion_module)
        if desired_velocity is not None:
            self.set_desired_velocity(desired_velocity, x_0.velocity)
        if predictions is not None:
            self.set_predictions(predictions)
        if reach_set is not None:
            self.set_reach_set(reach_set)
        if behavior is not None:
            self.set_behavior(behavior)

    def set_reach_set(self, reach_set):
        self.reach_set = reach_set

    def set_x_0(self, x_0: ReactivePlannerState):
        # set Cartesian initial state
        self.x_0 = x_0
        if self.x_0.velocity < self._low_vel_mode_threshold:
            self._LOW_VEL_MODE = True
            self.msg_logger.debug("Plan Timestep in Low-Velocity Mode!")
        else:
            self._LOW_VEL_MODE = False

    def set_x_cl(self, x_cl):
        # set curvilinear initial state
        if self.x_cl is not None and not self.set_new_ref_path:
            self.x_cl = x_cl
        else:
            self.x_cl = self._compute_initial_states(self.x_0)
            self.set_new_ref_path = False

    def set_ego_vehicle_state(self, current_ego_vehicle):
        self.ego_vehicle_history.append(current_ego_vehicle)

    def set_behavior(self, behavior):
        self.behavior = behavior

    def record_state_and_input(self, state: ReactivePlannerState):
        """
        Adds state to list of recorded states
        Adds control inputs to list of recorded inputs
        """
        # append state to state list
        self.record_state_list.append(state)

        # compute control inputs and append to input list
        if len(self.record_state_list) > 1:
            steering_angle_speed = (state.steering_angle - self.record_state_list[-2].steering_angle) / self.dT
        else:
            steering_angle_speed = 0.0

        input_state = InputState(time_step=state.time_step,
                                 acceleration=state.acceleration,
                                 steering_angle_speed=steering_angle_speed)
        self.record_input_list.append(input_state)

    def set_goal_area(self, goal_area: GoalRegion):
        """
        Sets the planning problem
        :param goal_area: Goal Area of Planning Problem
        """
        self.goal_area = goal_area

    def set_occlusion_module(self, occ_module):
        self.use_occ_model = True
        self.occlusion_module = occ_module

    def set_planning_problem(self, planning_problem: PlanningProblem):
        """
        Sets the planning problem
        :param planning_problem: PlanningProblem
        """
        self.planning_problem = planning_problem

    def set_sampling_parameters(self, t_min: float, horizon: float, delta_d_min: float, delta_d_max: float):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param horizon: sampled time horizon
        :param delta_d_min: min lateral sampling
        :param delta_d_max: max lateral sampling
        """
        self.sampling_handler.update_static_params(t_min, horizon, delta_d_min, delta_d_max)

    def set_desired_velocity(self, desired_velocity: float, current_speed: float = None, stopping: bool = False,
                             v_limit: float = 36):
        """
        Sets desired velocity and calculates velocity for each sample
        :param desired_velocity: velocity in m/s
        :param current_speed: velocity in m/s
        :param stopping
        :param v_limit: limit velocity due to behavior planner in m/s
        :return: velocity in m/s
        """
        self.desired_velocity = desired_velocity

        min_v = max(0.01, current_speed - 0.5 * self.vehicle_params.a_max * self.horizon)
        max_v = min(min(current_speed + (self.vehicle_params.a_max / 7.0) * self.horizon, v_limit),
                    self.vehicle_params.v_max)

        self.sampling_handler.set_v_sampling(min_v, max_v)

        self.msg_logger.info('Sampled interval of velocity: {} m/s - {} m/s'.format(min_v, max_v))

    def set_risk_costs(self, trajectory):

        ego_risk_dict, obst_risk_dict, ego_harm_dict, obst_harm_dict, ego_risk, obst_risk, obst_harm_occ = calc_risk(
            traj=trajectory,
            ego_state=self.x_0,
            predictions=self.predictions,
            scenario=self.scenario,
            ego_id=24,
            vehicle_params=self.vehicle_params,
            road_boundary=self.road_boundary,
            params_harm=self.params_harm,
            params_risk=self.params_risk,
        )
        trajectory._ego_risk = ego_risk
        trajectory._obst_risk = obst_risk
        return trajectory

    def trajectory_collision_check(self, feasible_trajectories):
        """
        Checks feasible trajectories for collisions with static obstacles
        :param feasible_trajectories: feasible trajectories list
        :return trajectory: optimal feasible trajectory or None
        """
        # go through sorted list of sorted trajectories and check for collisions
        for trajectory in feasible_trajectories:
            # skip trajectory if occ module is activated and trajectory is invalid (harm exceeds max harm)
            if self.use_occ_model and trajectory.valid is False:
                continue

            # Add Occupancy of Trajectory to do Collision Checks later
            cart_traj = self._compute_cart_traj(trajectory)
            occupancy = self.convert_state_list_to_commonroad_object(cart_traj.state_list)
            # get collision_object
            coll_obj = self.create_coll_object(occupancy, self.vehicle_params, self.x_0)

            # TODO: Check kinematic checks in cpp. no feasible traj available

            if self.use_prediction:
                collision_detected = collision_check_prediction(
                    predictions=self.predictions,
                    scenario=self.scenario,
                    ego_co=coll_obj,
                    frenet_traj=trajectory,
                    time_step=self.x_0.time_step,
                )
                if collision_detected:
                    self._collision_counter += 1
            else:
                collision_detected = False

            leaving_road_at = trajectories_collision_static_obstacles(
                trajectories=[coll_obj],
                static_obstacles=self.road_boundary,
                method="grid",
                num_cells=32,
                auto_orientation=True,
            )
            if leaving_road_at[0] != -1:
                coll_time_step = leaving_road_at[0] - self.x_0.time_step
                coll_vel = trajectory.cartesian.v[coll_time_step]

                boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
                    velocity=coll_vel, coeff=self.params_harm
                )

            else:
                boundary_harm = 0

            # Save Status of Trajectory to sort for alternative
            trajectory.boundary_harm = boundary_harm
            trajectory._coll_detected = collision_detected

            if not collision_detected and boundary_harm == 0:
                if self.use_occ_model:
                    metric, safety_check = self.occlusion_module.trajectory_safety_assessment(trajectory)
                    if safety_check is False:
                        continue

                return trajectory

        return None

    def _compute_trajectory_pair(self, trajectory: TrajectorySample) -> tuple:
        """
        Computes the output required for visualizing in CommonRoad framework
        :param trajectory: the optimal trajectory
        :return: (CartesianTrajectory, FrenetTrajectory, lon sample, lat sample)
        """
        # go along state list
        cart_list = list()
        cl_list = list()

        lon_list = list()
        lat_list = list()
        for i in range(len(trajectory.cartesian.x)):
            # create Cartesian state
            cart_states = dict()
            cart_states['time_step'] = self.x_0.time_step+i
            cart_states['position'] = np.array([trajectory.cartesian.x[i], trajectory.cartesian.y[i]])
            cart_states['orientation'] = trajectory.cartesian.theta[i]
            cart_states['velocity'] = trajectory.cartesian.v[i]
            cart_states['acceleration'] = trajectory.cartesian.a[i]
            if i > 0:
                cart_states['yaw_rate'] = (trajectory.cartesian.theta[i] - trajectory.cartesian.theta[i-1]) / self.dT
            else:
                cart_states['yaw_rate'] = self.x_0.yaw_rate
            # TODO Check why computation with yaw rate was faulty ??
            cart_states['steering_angle'] = np.arctan2(self.vehicle_params.wheelbase *
                                                       trajectory.cartesian.kappa[i], 1.0)
            cart_list.append(ReactivePlannerState(**cart_states))

            # create curvilinear state
            # TODO: This is not correct
            cl_states = dict()
            cl_states['time_step'] = self.x_0.time_step+i
            cl_states['position'] = np.array([trajectory.curvilinear.s[i], trajectory.curvilinear.d[i]])
            cl_states['velocity'] = trajectory.cartesian.v[i]
            cl_states['acceleration'] = trajectory.cartesian.a[i]
            cl_states['orientation'] = trajectory.cartesian.theta[i]
            cl_states['yaw_rate'] = trajectory.cartesian.kappa[i]
            cl_list.append(CustomState(**cl_states))

            lon_list.append(
                [trajectory.curvilinear.s[i], trajectory.curvilinear.s_dot[i], trajectory.curvilinear.s_ddot[i]])
            lat_list.append(
                [trajectory.curvilinear.d[i], trajectory.curvilinear.d_dot[i], trajectory.curvilinear.d_ddot[i]])

        # make Cartesian and Curvilinear Trajectory
        cartTraj = Trajectory(self.x_0.time_step, cart_list)
        cvlnTraj = Trajectory(self.x_0.time_step, cl_list)

        # correct orientations of cartesian output trajectory
        cartTraj_corrected = self.shift_orientation(cartTraj, interval_start=self.x_0.orientation - np.pi,
                                                    interval_end=self.x_0.orientation + np.pi)

        return cartTraj_corrected, cvlnTraj, lon_list, lat_list

    def _compute_cart_traj(self, trajectory: TrajectorySample) -> Trajectory:
        """
        Computes the output required for visualizing in CommonRoad framework
        :param trajectory: the optimal trajectory
        :return: (CartesianTrajectory, FrenetTrajectory, lon sample, lat sample)
        """
        # Cache attributes for quicker access
        cartesian = trajectory.cartesian
        x_0 = self.x_0
        dT = self.dT
        vehicle_params = self.vehicle_params

        # Precompute values that are static or can be vectorized
        time_steps = x_0.time_step + np.arange(len(cartesian.x))
        positions = np.vstack((cartesian.x, cartesian.y)).T
        orientations = cartesian.theta
        velocities = cartesian.v
        accelerations = cartesian.a
        yaw_rates = np.gradient(cartesian.theta) / dT
        yaw_rates[0] = x_0.yaw_rate  # set the first yaw_rate to initial condition
        steering_angles = np.arctan2(vehicle_params.wheelbase * cartesian.kappa, 1.0)

        # Use a list comprehension to create ReactivePlannerState instances
        cart_list = [
            ReactivePlannerState(
                time_step=int(time_step),  # Convert numpy int64 to Python int
                position=position,
                orientation=orientation,
                velocity=velocity,
                acceleration=acceleration,
                yaw_rate=yaw_rate,
                steering_angle=steering_angle
            )
            for time_step, position, orientation, velocity, acceleration, yaw_rate, steering_angle in zip(
                time_steps, positions, orientations, velocities, accelerations, yaw_rates, steering_angles)
        ]

        return Trajectory(x_0.time_step, cart_list)

    def convert_state_list_to_commonroad_object(self, state_list: List[ReactivePlannerState], obstacle_id: int = 42):
        """
        Converts a CR trajectory to a CR dynamic obstacle with given dimensions
        :param state_list: trajectory state list of reactive planner
        :param obstacle_id: [optional] ID of ego vehicle dynamic obstacle
        :return: CR dynamic obstacle representing the ego vehicle
        """
        # shift trajectory positions to center
        new_state_list = list()
        for state in state_list:
            new_state_list.append(state.shift_positions_to_center(self.vehicle_params.wb_rear_axle))

        trajectory = Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)
        # get shape of vehicle
        shape = Rectangle(self.vehicle_params.length, self.vehicle_params.width)
        # get trajectory prediction
        prediction = TrajectoryPrediction(trajectory, shape)
        return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, trajectory.state_list[0], prediction)

    def create_coll_object(self, trajectory, vehicle_params, ego_state):
        """Create a collision_object of the trajectory for collision checking with road
        boundary and with other vehicles."""

        collision_object_raw = hf.create_tvobstacle_trajectory(
            traj_list=trajectory,
            box_length=vehicle_params.length / 2,
            box_width=vehicle_params.width / 2,
            start_time_step=ego_state.time_step,
        )
        # if the preprocessing fails, use the raw trajectory
        collision_object, err = trajectory_preprocess_obb_sum(
            collision_object_raw
        )
        if err:
            collision_object = collision_object_raw

        return collision_object

    def shift_orientation(self, trajectory: Trajectory, interval_start=-np.pi, interval_end=np.pi):
        for state in trajectory.state_list:
            while state.orientation < interval_start:
                state.orientation += 2 * np.pi
            while state.orientation > interval_end:
                state.orientation -= 2 * np.pi
        return trajectory

    def _check_valid_settings(self):
        """Checks validity of provided dt and horizon"""
        assert self.dT > 0, 'provided dt is not correct! dt = {}'.format(self.dT)
        assert self.N > 0 and isinstance(self.N, int), 'N is not correct!'
        assert self.horizon > 0, 'provided t_h is not correct! dt = {}'.format(self.horizon)

    def set_scenario(self, scenario: Scenario):
        """Update the scenario to synchronize between agents"""
        self.scenario = scenario

        if not self.road_boundary:
            try:
                (
                    _,
                    self.road_boundary,
                ) = create_road_boundary_obstacle(
                    scenario=self.scenario,
                    method="aligned_triangulation",
                    axis=2,
                )
            except:
                raise RuntimeError("Road Boundary can not be created")

    def _compute_initial_states(self, x_0: ReactivePlannerState) -> (np.ndarray, np.ndarray):
        """
        Computes the curvilinear initial states for the polynomial planner based on a Cartesian CommonRoad state
        :param x_0: The CommonRoad state object representing the initial state of the vehicle
        :return: A tuple containing the initial longitudinal and lateral states (lon,lat)
        """
        # compute curvilinear position
        try:
            s, d = self.coordinate_system.convert_to_curvilinear_coords(x_0.position[0], x_0.position[1])
        except ValueError:
            self.msg_logger.critical("Initial state could not be transformed.")
            raise ValueError("Initial state could not be transformed.")

        # factor for interpolation
        s_idx = np.argmax(self.coordinate_system.ref_pos > s) - 1
        s_lambda = (s - self.coordinate_system.ref_pos[s_idx]) / (
                self.coordinate_system.ref_pos[s_idx + 1] - self.coordinate_system.ref_pos[s_idx])

        # compute orientation in curvilinear coordinate frame
        ref_theta = np.unwrap(self.coordinate_system.ref_theta)
        theta_cl = x_0.orientation - interpolate_angle(s, self.coordinate_system.ref_pos[s_idx], self.coordinate_system.ref_pos[s_idx + 1],
                                                       ref_theta[s_idx], ref_theta[s_idx + 1])

        # compute reference curvature
        kr = (self.coordinate_system.ref_curv[s_idx + 1] - self.coordinate_system.ref_curv[s_idx]) * s_lambda + self.coordinate_system.ref_curv[
            s_idx]
        # compute reference curvature change
        kr_d = (self.coordinate_system.ref_curv_d[s_idx + 1] - self.coordinate_system.ref_curv_d[s_idx]) * s_lambda + self.coordinate_system.ref_curv_d[s_idx]

        # compute initial ego curvature from initial steering angle
        kappa_0 = np.tan(x_0.steering_angle) / self.vehicle_params.wheelbase

        # compute d' and d'' -> derivation after arclength (s): see Eq. (A.3) and (A.5) in Diss. Werling
        d_p = (1 - kr * d) * np.tan(theta_cl)
        d_pp = -(kr_d * d + kr * d_p) * np.tan(theta_cl) + ((1 - kr * d) / (math.cos(theta_cl) ** 2)) * (
                kappa_0 * (1 - kr * d) / math.cos(theta_cl) - kr)

        # compute s dot (s_velocity) and s dot dot (s_acceleration) -> derivation after time
        s_velocity = x_0.velocity * math.cos(theta_cl) / (1 - kr * d)
        if s_velocity < 0:
            raise Exception("Initial state or reference incorrect! Curvilinear velocity is negative which indicates"
                            "that the ego vehicle is not driving in the same direction as specified by the reference")

        s_acceleration = x_0.acceleration
        s_acceleration -= (s_velocity ** 2 / math.cos(theta_cl)) * (
                (1 - kr * d) * np.tan(theta_cl) * (kappa_0 * (1 - kr * d) / (math.cos(theta_cl)) - kr) -
                (kr_d * d + kr * d_p))
        s_acceleration /= ((1 - kr * d) / (math.cos(theta_cl)))

        # compute d dot (d_velocity) and d dot dot (d_acceleration)
        if self._LOW_VEL_MODE:
            # in LOW_VEL_MODE: d_velocity and d_acceleration are derivatives w.r.t arclength (s)
            d_velocity = d_p
            d_acceleration = d_pp
        else:
            # in HIGH VEL MODE: d_velocity and d_acceleration are derivatives w.r.t time
            d_velocity = x_0.velocity * math.sin(theta_cl)
            d_acceleration = s_acceleration * d_p + s_velocity ** 2 * d_pp

        x_0_lon: List[float] = [s, s_velocity, s_acceleration]
        x_0_lat: List[float] = [d, d_velocity, d_acceleration]

        self.msg_logger.debug(f'Initial state for planning is {x_0}')
        self.msg_logger.debug(f'Initial x_0 lon = {x_0_lon}')
        self.msg_logger.debug(f'Initial x_0 lat = {x_0_lat}')

        return x_0_lon, x_0_lat

    def plan_postprocessing(self, optimal_trajectory, planning_time):
        # **************************
        # Logging
        # **************************
        if optimal_trajectory is not None and self.logger:
            self.logger.log(optimal_trajectory, time_step=self.x_0.time_step,
                            infeasible_kinematics=self._infeasible_count_kinematics,
                            percentage_kinematics=self.infeasible_kinematics_percentage, planning_time=planning_time,
                            ego_vehicle=self.ego_vehicle_history[-1], desired_velocity=self.desired_velocity)
            self.logger.log_predicition(self.predictions)
        if self.save_all_traj and self.logger:
            self.logger.log_all_trajectories(self.all_traj, self.x_0.time_step)

        # **************************
        # Check Cost Status
        # **************************
        # if optimal_trajectory is not None:
        #     self._optimal_cost = optimal_trajectory.cost
        #
        #     if self._optimal_cost is not None:
        #         self.msg_logger.debug('Found optimal trajectory with {}% of maximum seen costs'
        #                               .format(int((self._optimal_cost/self.max_seen_costs)*100)))
        #
        #         if self.max_seen_costs < self._optimal_cost:
        #             self.max_seen_costs = self._optimal_cost

    def set_stopping_point(self, stop_s_coordinate):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        """
        self.stopping_s = stop_s_coordinate

    @abstractmethod
    def plan(self):
        """Planner step function.

        To be implemented for every specific planner.

        :returns: Exit code of the planner step,
                  The planned trajectory.
        """
        raise NotImplementedError()

    # @abstractmethod
    # def check_collision(self, ego_obstacle):
    #     """Planner collision check function.
    #
    #     To be implemented for every specific planner.
    #
    #     :returns: Exit code of the collision check,
    #     """
    #     raise NotImplementedError()

    @abstractmethod
    def set_reference_and_coordinate_system(self, *args, **kwargs):
        """Set reference path
        To be implemented for every specific planner.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_cost_function(self, cost_function):
        """Set cost function
        To be implemented for every specific planner.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_predictions(self, predictions: dict):
        """Set predictions
        To be implemented for every specific planner.
        """
        raise NotImplementedError()