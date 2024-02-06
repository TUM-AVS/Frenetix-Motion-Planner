__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import time
import numpy as np
import copy
from itertools import product
from typing import List

# frenetix_motion_planner imports
from frenetix_motion_planner.sampling_matrix import generate_sampling_matrix

from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem
from cr_scenario_handler.utils.visualization import visualize_scenario_and_pp

import frenetix
import frenetix.trajectory_functions
import frenetix.trajectory_functions.feasability_functions as ff
import frenetix.trajectory_functions.cost_functions as cf

from frenetix_motion_planner.planner import Planner

# get logger
# msg_logger = logging.getLogger("Message_logger")


class ReactivePlannerCpp(Planner):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """
    def __init__(self, config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger):
        """
        Constructor of the reactive planner
        : param config: Configuration object holding all planner-relevant configurations
        """
        super().__init__(config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger)

        self.predictionsForCpp = {}

        # *****************************
        # C++ Trajectory Handler Import
        # *****************************

        self.handler: frenetix.TrajectoryHandler = frenetix.TrajectoryHandler(dt=self.config_plan.planning.dt)
        self.coordinate_system_cpp: frenetix.CoordinateSystemWrapper
        self.trajectory_handler_set_constant_cost_functions()
        self.trajectory_handler_set_constant_feasibility_functions()

    def set_predictions(self, predictions: dict):
        self.use_prediction = True
        self.predictions = predictions
        for key, pred in self.predictions.items():
            num_steps = pred['pos_list'].shape[0]
            predicted_path: List[frenetix.PoseWithCovariance] = [None] * num_steps

            for time_step in range(num_steps):
                # Ensure the position is in float64 format
                position = np.append(pred['pos_list'][time_step].astype(np.float64), [0.0]).astype(np.float64)

                # Preallocate orientation array and fill in the values
                orientation = np.zeros(4, dtype=np.float64)
                orientation[2:] = np.array([np.sin(pred['orientation_list'][time_step] / 2.0),
                                            np.cos(pred['orientation_list'][time_step] / 2.0)], dtype=np.float64)

                # Symmetrize the covariance matrix if necessary and convert to float64
                covariance = pred['cov_list'][time_step].astype(np.float64)
                if not np.array_equal(covariance, covariance.T):
                    covariance = ((covariance + covariance.T) / 2).astype(np.float64)

                # Create the covariance matrix for PoseWithCovariance
                covariance_matrix = np.zeros((6, 6), dtype=np.float64)
                covariance_matrix[:2, :2] = covariance

                # Create PoseWithCovariance object and add to predicted_path
                pwc = frenetix.PoseWithCovariance(position, orientation, covariance_matrix)
                predicted_path[time_step] = pwc

            # Store the resulting predicted path
            self.predictionsForCpp[key] = frenetix.PredictedObject(int(key), predicted_path)

    def set_cost_function(self, cost_weights):
        self.config_plan.cost.cost_weights = cost_weights
        self.trajectory_handler_set_constant_cost_functions()
        self.trajectory_handler_set_constant_feasibility_functions()
        self.trajectory_handler_set_changing_functions()
        if self.logger:
            self.logger.set_logging_header(self.config_plan.cost.cost_weights)

    def trajectory_handler_set_constant_feasibility_functions(self):
        self.handler.add_feasability_function(ff.CheckYawRateConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                        wheelbase=self.vehicle_params.wheelbase,
                                                                        wholeTrajectory=False
                                                                        ))
        self.handler.add_feasability_function(ff.CheckAccelerationConstraint(switchingVelocity=self.vehicle_params.v_switch,
                                                                             maxAcceleration=self.vehicle_params.a_max,
                                                                             wholeTrajectory=False)
                                                                             )
        self.handler.add_feasability_function(ff.CheckCurvatureConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                          wheelbase=self.vehicle_params.wheelbase,
                                                                          wholeTrajectory=False
                                                                          ))
        self.handler.add_feasability_function(ff.CheckCurvatureRateConstraint(wheelbase=self.vehicle_params.wheelbase,
                                                                              velocityDeltaMax=self.vehicle_params.v_delta_max,
                                                                              wholeTrajectory=False
                                                                              ))

    def trajectory_handler_set_constant_cost_functions(self):
        name = "acceleration"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateAccelerationCost(name, self.cost_weights[name]))

        name = "jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateJerkCost(name, self.cost_weights[name]))

        name = "lateral_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLateralJerkCost(name, self.cost_weights[name]))

        name = "longitudinal_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLongitudinalJerkCost(name, self.cost_weights[name]))

        name = "orientation_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateOrientationOffsetCost(name, self.cost_weights[name]))

        name = "lane_center_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLaneCenterOffsetCost(name, self.cost_weights[name]))

        name = "distance_to_reference_path"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateDistanceToReferencePathCost(name, self.cost_weights[name]))

    def trajectory_handler_set_changing_functions(self):
        self.handler.add_function(frenetix.trajectory_functions.FillCoordinates(
            lowVelocityMode=self._LOW_VEL_MODE,
            initialOrientation=self.x_0.orientation,
            coordinateSystem=self.coordinate_system_cpp,
            horizon=int(self.config_plan.planning.planning_horizon)
        ))

        name = "prediction"
        if name in self.cost_weights.keys():
            self.handler.add_cost_function(
                cf.CalculateCollisionProbabilityFast(name, self.cost_weights[name], self.predictionsForCpp,
                                                     self.vehicle_params.length*2.5, self.vehicle_params.width*2))

        name = "distance_to_obstacles"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            obstacle_positions = np.zeros((len(self.scenario.obstacles), 2))
            for i, obstacle in enumerate(self.scenario.obstacles):
                state = obstacle.state_at_time(self.x_0.time_step)
                if state is not None:
                    obstacle_positions[i, 0] = state.position[0]
                    obstacle_positions[i, 1] = state.position[1]

            self.handler.add_cost_function(cf.CalculateDistanceToObstacleCost(name, self.cost_weights[name], obstacle_positions))

        name = "velocity_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateVelocityOffsetCost(name, self.cost_weights[name], self.desired_velocity))

    def set_reference_and_coordinate_system(self, reference_path: np.ndarray):
        """
        Automatically creates a curvilinear coordinate system from a given reference path
        :param reference_path: reference_path as polyline
        """
        self.coordinate_system = CoordinateSystem(reference=reference_path, config_sim=self.config_sim)

        # For manual debugging reasons:
        if self.config_sim.visualization.ref_path_debug:
            visualize_scenario_and_pp(scenario=self.scenario, planning_problem=self.planning_problem,
                                      save_path=self.config_sim.simulation.log_path, cosy=self.coordinate_system)

        self.coordinate_system_cpp: frenetix.CoordinateSystemWrapper = frenetix.CoordinateSystemWrapper(reference_path)
        self.set_new_ref_path = True
        if self.logger:
            self.logger.sql_logger.write_reference_path(reference_path)

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """
        self._infeasible_count_kinematics = np.zeros(11)
        self._collision_counter = 0
        self.infeasible_kinematics_percentage = 0
        # **************************************
        # Initialization of Cpp Frenet Functions
        # **************************************
        self.trajectory_handler_set_changing_functions()
        x_0_lon = None
        x_0_lat = None
        if self.x_cl is None:
            initial_state = frenetix.TrajectorySample(
                x0=np.float64(self.x_0.position[0]),
                y0=np.float64(self.x_0.position[1]),
                orientation0=self.x_0.orientation,
                acceleration0=self.x_0.acceleration,
                velocity0=self.x_0.velocity
            )

            initial_state_computation = frenetix.trajectory_functions.ComputeInitialState(
                coordinateSystem=self.coordinate_system_cpp,
                wheelBase=self.vehicle_params.wheelbase,
                steeringAngle=self.x_0.steering_angle,
                lowVelocityMode=self._LOW_VEL_MODE
            )

            initial_state_computation.evaluate_trajectory(initial_state)

            x_0_lat = [np.float64(initial_state.curvilinear.d), np.float64(initial_state.curvilinear.d_dot),
                       np.float64(initial_state.curvilinear.d_ddot)]

            x_0_lon = [np.float64(initial_state.curvilinear.s), np.float64(initial_state.curvilinear.s_dot),
                       np.float64(initial_state.curvilinear.s_ddot)]
        else:
            x_0_lat = self.x_cl[1]
            x_0_lon = self.x_cl[0]

        self.msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(x_0_lon, x_0_lat))
        self.msg_logger.debug('Desired velocity is {} m/s'.format(self.desired_velocity))

        # Initialization of while loop
        optimal_trajectory = None
        sampling_matrix = None
        feasible_trajectories = []
        infeasible_trajectories = []
        t0 = time.time()

        # Initial index of sampling set to use
        samp_level = self._sampling_min

        # sample until trajectory has been found or sampling sets are empty
        while optimal_trajectory is None and samp_level < self._sampling_max:

            # *************************************
            # Create & Evaluate Trajectories in Cpp
            # *************************************
            t1_range = np.array(list(self.sampling_handler.t_sampling.to_range(samp_level).union({self.N*self.dT})))
            ss1_range = np.array(list(self.sampling_handler.v_sampling.to_range(samp_level).union({x_0_lon[1]})))
            d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]})))

            sampling_matrix = generate_sampling_matrix(t0_range=0.0,
                                                       t1_range=t1_range,
                                                       s0_range=x_0_lon[0],
                                                       ss0_range=x_0_lon[1],
                                                       sss0_range=x_0_lon[2],
                                                       ss1_range=ss1_range,
                                                       sss1_range=0,
                                                       d0_range=x_0_lat[0],
                                                       dd0_range=x_0_lat[1],
                                                       ddd0_range=x_0_lat[2],
                                                       d1_range=d1_range,
                                                       dd1_range=0.0,
                                                       ddd1_range=0.0)

            self.handler.reset_Trajectories()
            self.handler.generate_trajectories(sampling_matrix, self._LOW_VEL_MODE)

            if not self.config_plan.debug.multiproc or (self.config_sim.simulation.use_multiagent and
                                                        self.config_sim.simulation.multiprocessing):
                self.handler.evaluate_all_current_functions(True)
            else:
                self.handler.evaluate_all_current_functions_concurrent(True)

            feasible_trajectories = []
            infeasible_trajectories = []
            for trajectory in self.handler.get_sorted_trajectories():
                # check if trajectory is feasible
                if trajectory.feasible:
                    feasible_trajectories.append(trajectory)
                elif trajectory.valid:
                    infeasible_trajectories.append(trajectory)

            if len(feasible_trajectories) + len(infeasible_trajectories) < 1:
                self.msg_logger.critical("No Valid Trajectories!")
            else:
                self.infeasible_kinematics_percentage = float(len(feasible_trajectories)
                                                        / (len(feasible_trajectories) + len(infeasible_trajectories))) * 100

            # print size of feasible trajectories and infeasible trajectories
            self.msg_logger.debug('Found {} feasible trajectories and {} infeasible trajectories'.format(feasible_trajectories.__len__(), infeasible_trajectories.__len__()))
            self.msg_logger.debug(
                'Percentage of valid & feasible trajectories: %s %%' % str(self.infeasible_kinematics_percentage))

            # ******************************************
            # Check Feasible Trajectories for Collisions
            # ******************************************
            optimal_trajectory = self.trajectory_collision_check(feasible_trajectories)

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            samp_level += 1

        planning_time = time.time() - t0

        self.transfer_infeasible_logging_information(infeasible_trajectories)

        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
            self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
            self.infeasible_count_collision))

        # *******************************************
        # Find alternative Optimal Trajectory if None
        # *******************************************
        if optimal_trajectory is None and feasible_trajectories:
            if self.config_plan.planning.emergency_mode == "stopping":
                optimal_trajectory = self._select_stopping_trajectory(feasible_trajectories, sampling_matrix, x_0_lat[0])
                self.msg_logger.warning("No optimal trajectory available. Select stopping trajectory!")
            else:
                for traje in feasible_trajectories:
                    self.set_risk_costs(traje)
                sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk, reverse=False)
                self.msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
                optimal_trajectory = sort_risk[0]

        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        self.trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None
        if self.trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list,
                                                                               self.config_sim.simulation.ego_agent_id)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        # ************************************
        # Set Risk Costs to Optimal Trajectory
        # ************************************
        if optimal_trajectory is not None and self.log_risk:
            optimal_trajectory = self.set_risk_costs(optimal_trajectory)

        self.optimal_trajectory = optimal_trajectory

        # **************************
        # Logging
        # **************************
        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj:
            self.all_traj = feasible_trajectories + infeasible_trajectories

        self.plan_postprocessing(optimal_trajectory=optimal_trajectory, planning_time=planning_time)

        return self.trajectory_pair

    @staticmethod
    def _select_stopping_trajectory(trajectories, sampling_matrix, d_pos):

        min_v_list = np.unique(sampling_matrix[:, 5])
        min_t_list = np.unique(sampling_matrix[:, 1])

        min_d_list = np.unique(sampling_matrix[:, 10])
        sorted_d_indices = np.argsort(np.abs(min_d_list - d_pos))
        min_d_list = min_d_list[sorted_d_indices]

        # Create a dictionary for quick lookups
        trajectory_dict = {}
        for traj in trajectories:
            v, t, d = traj.sampling_parameters[5], traj.sampling_parameters[1], traj.sampling_parameters[10]
            if v not in trajectory_dict:
                trajectory_dict[v] = {}
            if t not in trajectory_dict[v]:
                trajectory_dict[v][t] = {}
            trajectory_dict[v][t][d] = traj

        # Check combinations of v, t, d values for valid trajectories
        for v, t, d in product(min_v_list, min_t_list, min_d_list):
            if v in trajectory_dict and t in trajectory_dict[v] and d in trajectory_dict[v][t]:
                return trajectory_dict[v][t][d]

    def transfer_infeasible_logging_information(self, infeasible_trajectories):

        feas_list = [i.feasabilityMap['Curvature Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[5] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Yaw rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[6] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Curvature Rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[7] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Acceleration Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[8] = int(sum(acc_feas))

        self._infeasible_count_kinematics[0] = int(sum(self._infeasible_count_kinematics))
