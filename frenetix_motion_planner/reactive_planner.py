__author__ = "Rainer Trauth, Gerald Würsching, Christian Pek"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import math
import time

import numpy as np
from typing import List
import multiprocessing
from multiprocessing.context import Process

# frenetix_motion_planner imports
from frenetix_motion_planner.polynomial_trajectory import QuinticTrajectory, QuarticTrajectory
from frenetix_motion_planner.trajectories import TrajectoryBundle, TrajectorySample, CartesianSample, CurviLinearSample
from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem, interpolate_angle
from frenetix_motion_planner.cost_functions.cost_function import AdaptableCostFunction

from frenetix_motion_planner.planner import Planner

# precision value
_EPS = 1e-5

# get logger
# msg_logger = logging.getLogger("Message_logger")


class ReactivePlannerPython(Planner):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """
    def __init__(self, config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger):
        """
        Constructor of the reactive planner
        : param config: Configuration object holding all planner-relevant configurations
        """
        super().__init__(config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger)

        # **************************
        # Cost Function Setting
        # **************************
        cost_function = AdaptableCostFunction(rp=self, configuration=self.config_plan)
        self.set_cost_function(cost_function=cost_function)

    def set_predictions(self, predictions: dict):
        self.use_prediction = True
        self.predictions = predictions

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
        # self.logger.set_logging_header(self.config_plan.cost.cost_weights)

    def set_reference_and_coordinate_system(self, reference_path: np.ndarray = None):
        """
        Automatically creates a curvilinear coordinate system from a given reference path or sets a given
        curvilinear coordinate system for the planner to use
        :param reference_path: reference path as polyline
        :param coordinate_system: given CoordinateSystem object which is used by the planner
        """
        self.coordinate_system = CoordinateSystem(reference=reference_path, config_sim=self.config_sim)
        self.set_new_ref_path = True

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """

        self.msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(self.x_cl[0], self.x_cl[1]))
        self.msg_logger.debug('Desired velocity is {} m/s'.format(self.desired_velocity))

        # Initialization of while loop
        optimal_trajectory = None
        t0 = time.time()

        # Initial index of sampling set to use
        samp_level = self._sampling_min

        # sample until trajectory has been found or sampling sets are empty
        while optimal_trajectory is None and samp_level < self._sampling_max:

            self.cost_function.update_state(scenario=self.scenario, rp=self,
                                            predictions=self.predictions, reachset=self.reach_set)

            bundle = self._create_trajectory_bundle(self.x_cl[0], self.x_cl[1], self.cost_function, samp_level=samp_level)

            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step

            optimal_trajectory = self._get_optimal_trajectory(bundle, samp_level)

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            samp_level += 1

        planning_time = time.time() - t0

        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
            self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
            self.infeasible_count_collision))

        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        self.trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None
        if self.trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        if optimal_trajectory is None and self.x_0.velocity <= 0.1:
            self.msg_logger.warning('Planning standstill for the current scenario')
            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step
            optimal_trajectory = self._compute_standstill_trajectory()

        # ************************************
        # Set Risk Costs to Optimal Trajectory
        # ************************************
        if optimal_trajectory is not None and self.log_risk:
            optimal_trajectory = self.set_risk_costs(optimal_trajectory)

        self.optimal_trajectory = optimal_trajectory

        self.plan_postprocessing(optimal_trajectory=optimal_trajectory, planning_time=planning_time)

        return self.trajectory_pair

    def _create_trajectory_bundle(self, x_0_lon: np.array, x_0_lat: np.array, cost_function, samp_level: int) -> TrajectoryBundle:
        """
        Plans trajectory samples that try to reach a certain velocity and samples in this domain.
        Sample in time (duration) and velocity domain. Initial state is given. Longitudinal end state (s) is sampled.
        Lateral end state (d) is always set to 0.
        :param x_0_lon: np.array([s, s_dot, s_ddot])
        :param x_0_lat: np.array([d, d_dot, d_ddot])
        :param samp_level: index of the sampling parameter set to use
        :return: trajectory bundle with all sample trajectories.

        NOTE: Here, no collision or feasibility check is done!
        """
        # reset cost statistic
        self._min_cost = 10 ** 9
        self._max_cost = 0

        trajectories = list()
        for t in self.sampling_handler.t_sampling.to_range(samp_level):
            # Longitudinal sampling for all possible velocities
            for v in self.sampling_handler.v_sampling.to_range(samp_level):
                # end_state_lon = np.array([t * v + x_0_lon[0], v, 0.0])
                # trajectory_long = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=end_state_lon)
                trajectory_long = QuarticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=np.array([v, 0]))

                # Sample lateral end states (add x_0_lat to sampled states)
                if trajectory_long.coeffs is not None:
                    for d in self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]}):
                        end_state_lat = np.array([d, 0.0, 0.0])
                        # SWITCHING TO POSITION DOMAIN FOR LATERAL TRAJECTORY PLANNING
                        if self._LOW_VEL_MODE:
                            s_lon_goal = trajectory_long.evaluate_state_at_tau(t)[0] - x_0_lon[0]
                            if s_lon_goal <= 0:
                                s_lon_goal = t
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=s_lon_goal, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)

                        # Switch to sampling over t for high velocities
                        else:
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)
                        if trajectory_lat.coeffs is not None:
                            trajectory_sample = TrajectorySample(self.horizon, self.dT, trajectory_long, trajectory_lat,
                                                                 len(trajectories), costMap=self.cost_function.cost_weights)
                            trajectories.append(trajectory_sample)

        # perform pre-check and order trajectories according their cost
        trajectory_bundle = TrajectoryBundle(trajectories, cost_function=cost_function,
                                             multiproc=self._multiproc, num_workers=self._num_workers)
        self._total_count = len(trajectory_bundle._trajectory_bundle)
        self.msg_logger.debug('%s trajectories sampled' % len(trajectory_bundle._trajectory_bundle))
        return trajectory_bundle

    def _get_optimal_trajectory(self, trajectory_bundle: TrajectoryBundle, samp_lvl):
        """
        Computes the optimal trajectory from a given trajectory bundle
        :param trajectory_bundle: The trajectory bundle
        :return: The optimal trajectory if exists (otherwise None)
        """

        # reset statistics
        self._collision_counter = 0
        self._infeasible_count_kinematics = None
        infeasible_count_kinematics = [0] * 11

        # check kinematics of each trajectory
        if self._multiproc:
            # with multiprocessing
            # divide trajectory_bundle.trajectories into chunks
            chunk_size = math.ceil(len(trajectory_bundle.trajectories) / self._num_workers)
            chunks = [trajectory_bundle.trajectories[ii * chunk_size: min(len(trajectory_bundle.trajectories),
                                                                          (ii+1)*chunk_size)] for ii in range(0, self._num_workers)]

            # initialize list of Processes and Queues
            list_processes = []
            trajectories_all = []
            queue_1 = multiprocessing.Queue()
            queue_2 = multiprocessing.Queue()

            for chunk in chunks:
                p = Process(target=self.check_feasibility, args=(chunk, queue_1, queue_2))
                list_processes.append(p)
                p.start()

            # get return values from queue
            for p in list_processes:
                trajectories_all.extend(queue_1.get())
                if self._kinematic_debug:
                    temp = queue_2.get()
                    infeasible_count_kinematics = [x + y for x, y in zip(infeasible_count_kinematics, temp)]

            # wait for all processes to finish
            for p in list_processes:
                p.join()
        else:
            # without multiprocessing
            trajectories_all = self.check_feasibility(trajectory_bundle.trajectories)

        feasible_trajectories = [obj for obj in trajectories_all if obj.valid is True and obj.feasible is True]
        infeasible_trajectories = [obj for obj in trajectories_all if obj.valid is False or obj.feasible is False]

        # update number of infeasible trajectories
        self._infeasible_count_kinematics = infeasible_count_kinematics
        self._infeasible_count_kinematics[0] = len(infeasible_trajectories)
        self.infeasible_kinematics_percentage = float(len(feasible_trajectories) / len(trajectories_all)) * 100

        # print size of feasible trajectories and infeasible trajectories
        self.msg_logger.debug(
            'Found {} feasible trajectories and {} infeasible trajectories'.format(str(len(feasible_trajectories)),
                                                                                   str(len(infeasible_trajectories))))
        self.msg_logger.debug('Percentage of valid & feasible trajectories: %s %%' % str(self.infeasible_kinematics_percentage))

        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj:
            trajectory_bundle.trajectories = trajectories_all
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)
            self.all_traj = trajectory_bundle.trajectories
            trajectory_bundle.trajectories = list(filter(lambda x: x.feasible is True, trajectory_bundle.trajectories))
        else:
            # set feasible trajectories in bundle
            trajectory_bundle.trajectories = feasible_trajectories
            # sort trajectories according to their costs
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)

        # ******************************************
        # Check Feasible Trajectories for Collisions
        # ******************************************
        optimal_trajectory = self.trajectory_collision_check(feasible_trajectories=
                                                             trajectory_bundle.get_sorted_list(
                                                                 occlusion_module=self.occlusion_module))

        if samp_lvl >= self._sampling_max - 1 and optimal_trajectory is None and feasible_trajectories:
            for traje in feasible_trajectories:
                self.set_risk_costs(traje)
            sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk,
                               reverse=False)
            self.msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
            optimal_trajectory = sort_risk[0]
            return optimal_trajectory

        else:
            return optimal_trajectory

    def check_feasibility(self, trajectories: List[TrajectorySample], queue_1=None, queue_2=None):
        """
        Checks the kinematics of given trajectories in a bundle and computes the cartesian trajectory information
        Lazy evaluation, only kinematically feasible trajectories are evaluated further

        :param trajectories: The list of trajectory samples to check
        :param queue_1: Multiprocessing.Queue() object for storing trajectories
        :param queue_2: Multiprocessing.Queue() object for storing reason for infeasible trajectory in list
        :return: The list of output trajectories
        """
        # initialize lists for output trajectories
        # infeasible trajectory list is only used for visualization when self._draw_traj_set is True
        infeasible_invalid_count_kinematics = np.zeros(11)
        trajectory_list = list()

        # loop over list of trajectories
        for trajectory in trajectories:

            trajectory.feasible = True
            trajectory.valid = True

            # create time array and precompute time interval information
            t = np.round(np.arange(0, trajectory.trajectory_long.delta_tau + trajectory.dt, trajectory.dt), 5)
            t2 = np.round(np.power(t, 2), 10)
            t3 = np.round(np.power(t, 3), 10)
            t4 = np.round(np.power(t, 4), 10)
            t5 = np.round(np.power(t, 5), 10)

            # length of the trajectory sample (i.e., number of time steps. can be smaller than planning horizon)
            traj_len = len(t)

            # initialize long. (s) and lat. (d) state vectors
            s = np.zeros(self.N + 1)
            s_velocity = np.zeros(self.N + 1)
            s_acceleration = np.zeros(self.N + 1)
            d = np.zeros(self.N + 1)
            d_velocity = np.zeros(self.N + 1)
            d_acceleration = np.zeros(self.N + 1)

            # compute longitudinal position, velocity, acceleration from trajectory sample
            s[:traj_len] = trajectory.trajectory_long.calc_position(t, t2, t3, t4, t5)  # lon pos
            s_velocity[:traj_len] = trajectory.trajectory_long.calc_velocity(t, t2, t3, t4)  # lon velocity
            s_acceleration[:traj_len] = trajectory.trajectory_long.calc_acceleration(t, t2, t3)  # lon acceleration

            # s-enlargement of t-sampled trajectories
            for ext in range(traj_len, self.N + 1):
                s[ext] = s[ext-1] + trajectory.dt * s_velocity[traj_len-1]
            s_velocity[traj_len:] = s_velocity[traj_len - 1]
            s_acceleration[traj_len:] = 0.0

            # At low speeds, we have to sample the lateral motion over the travelled distance rather than time.
            if not self._LOW_VEL_MODE:
                d[:traj_len] = trajectory.trajectory_lat.calc_position(t, t2, t3, t4, t5)  # lat pos
                d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(t, t2, t3, t4)  # lat velocity
                d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(t, t2, t3)  # lat acceleration
            else:
                # compute normalized travelled distance for low velocity mode of lateral planning
                s1 = s[:traj_len] - s[0]
                s2 = np.square(s1)
                s3 = s2 * s1
                s4 = np.square(s2)
                s5 = s4 * s1

                # compute lateral position, velocity, acceleration from trajectory sample
                d[:traj_len] = trajectory.trajectory_lat.calc_position(s1, s2, s3, s4, s5)  # lat pos
                # in LOW_VEL_MODE d_velocity is actually d' (see Diss. Moritz Werling  p.124)
                d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(s1, s2, s3, s4)  # lat velocity
                d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(s1, s2, s3)  # lat acceleration

            # d-enlargement of t-sampled trajectories
            d[traj_len:] = d[traj_len - 1]
            d_velocity[traj_len:] = 0.0
            d_acceleration[traj_len:] = 0.0

            # precision for near zero velocities from evaluation of polynomial coefficients
            # set small velocities to zero
            if np.any(s_velocity < - _EPS):
                trajectory.valid = False
                infeasible_invalid_count_kinematics[10] += 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    continue
            s_velocity[np.abs(s_velocity) < _EPS] = 0.0
            # d_velocity[np.abs(d_velocity) < _EPS] = 0.0

            # Initialize trajectory state vectors
            # (Global) Cartesian positions x, y
            x = np.zeros(self.N + 1)
            y = np.zeros(self.N + 1)
            # (Global) Cartesian velocity v and acceleration a
            v = np.zeros(self.N + 1)
            a = np.zeros(self.N + 1)
            # Orientation theta: Cartesian (gl) and Curvilinear (cl)
            theta_gl = np.zeros(self.N + 1)
            theta_cl = np.zeros(self.N + 1)
            # Curvature kappa : Cartesian (gl) and Curvilinear (cl)
            kappa_gl = np.zeros(self.N + 1)
            kappa_cl = np.zeros(self.N + 1)

            # Initialize Feasibility boolean
            if not self._draw_traj_set:
                # pre-filter with quick underapproximative check for feasibility
                if np.any(np.abs(s_acceleration) > self.vehicle_params.a_max):
                    self.msg_logger.debug(f"Acceleration {np.max(np.abs(s_acceleration))}")
                    trajectory.feasible = False
                    infeasible_invalid_count_kinematics[1] += 1
                    trajectory_list.append(trajectory)
                    continue
                if np.any(s_velocity < -_EPS):
                    self.msg_logger.debug(f"Velocity {min(s_velocity)} at step")
                    trajectory.feasible = False
                    infeasible_invalid_count_kinematics[2] += 1
                    trajectory_list.append(trajectory)
                    continue

            infeasible_count_kinematics_traj = np.zeros(11)
            for i in range(0, len(s)):
                # compute orientations
                # see Appendix A.1 of Moritz Werling's PhD Thesis for equations
                if not self._LOW_VEL_MODE:
                    if s_velocity[i] > 0.001:
                        dp = d_velocity[i] / s_velocity[i]
                    else:
                        # if abs(d_velocity[i]) > 0.001:
                        #     dp = None
                        # else:
                        dp = 0.
                    # see Eq. (A.8) from Moritz Werling's Diss
                    ddot = d_acceleration[i] - dp * s_acceleration[i]

                    if s_velocity[i] > 0.001:
                        dpp = ddot / (s_velocity[i] ** 2)
                    else:
                        # if np.abs(ddot) > 0.00003:
                        #     dpp = None
                        # else:
                        dpp = 0.
                else:
                    dp = d_velocity[i]
                    dpp = d_acceleration[i]

                # factor for interpolation
                s_idx = np.argmax(self.coordinate_system.ref_pos > s[i]) - 1
                if s_idx + 1 >= len(self.coordinate_system.ref_pos):
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[3] = 1
                    break
                s_lambda = (s[i] - self.coordinate_system.ref_pos[s_idx]) / (self.coordinate_system.ref_pos[s_idx + 1] - self.coordinate_system.ref_pos[s_idx])

                # compute curvilinear (theta_cl) and global Cartesian (theta_gl) orientation
                if s_velocity[i] > 0.001:
                    # LOW VELOCITY MODE: dp = d_velocity[i]
                    # HIGH VELOCITY MODE: dp = d_velocity[i]/s_velocity[i]
                    theta_cl[i] = np.arctan2(dp, 1.0)

                    theta_gl[i] = theta_cl[i] + interpolate_angle(
                        s[i],
                        self.coordinate_system.ref_pos[s_idx],
                        self.coordinate_system.ref_pos[s_idx + 1],
                        self.coordinate_system.ref_theta[s_idx],
                        self.coordinate_system.ref_theta[s_idx + 1])
                else:
                    if self._LOW_VEL_MODE:
                        # dp = velocity w.r.t. to travelled arclength (s)
                        theta_cl[i] = np.arctan2(dp, 1.0)

                        theta_gl[i] = theta_cl[i] + interpolate_angle(
                            s[i],
                            self.coordinate_system.ref_pos[s_idx],
                            self.coordinate_system.ref_pos[s_idx + 1],
                            self.coordinate_system.ref_theta[s_idx],
                            self.coordinate_system.ref_theta[s_idx + 1])
                    else:
                        # in stillstand (s_velocity~0) and High velocity mode: assume vehicle keeps global orientation
                        theta_gl[i] = self.x_0.orientation if i == 0 else theta_gl[i - 1]

                        theta_cl[i] = theta_gl[i] - interpolate_angle(
                            s[i],
                            self.coordinate_system.ref_pos[s_idx],
                            self.coordinate_system.ref_pos[s_idx + 1],
                            self.coordinate_system.ref_theta[s_idx],
                            self.coordinate_system.ref_theta[s_idx + 1])

                # Interpolate curvature of reference path k_r at current position
                k_r = (self.coordinate_system.ref_curv[s_idx + 1] - self.coordinate_system.ref_curv[s_idx]) * s_lambda + self.coordinate_system.ref_curv[s_idx]
                # Interpolate curvature rate of reference path k_r_d at current position
                k_r_d = (self.coordinate_system.ref_curv_d[s_idx + 1] - self.coordinate_system.ref_curv_d[s_idx]) * s_lambda + \
                        self.coordinate_system.ref_curv_d[s_idx]

                # compute global curvature (see appendix A of Moritz Werling's PhD thesis)
                oneKrD = (1 - k_r * d[i])
                cosTheta = math.cos(theta_cl[i])
                tanTheta = np.tan(theta_cl[i])

                kappa_gl[i] = (dpp + (k_r * dp + k_r_d * d[i]) * tanTheta) * cosTheta * ((cosTheta / oneKrD) ** 2) + \
                              (cosTheta / oneKrD) * k_r

                kappa_cl[i] = kappa_gl[i] - k_r

                # compute (global) Cartesian velocity
                v[i] = s_velocity[i] * (oneKrD / (math.cos(theta_cl[i])))

                # compute (global) Cartesian acceleration
                a[i] = s_acceleration[i] * (oneKrD / cosTheta) + ((s_velocity[i] ** 2) / cosTheta) * (
                        oneKrD * tanTheta * (kappa_gl[i] * (oneKrD / cosTheta) - k_r) - (
                        k_r_d * d[i] + k_r * dp))

                # **************************
                # Velocity constraint
                # **************************
                if v[i] < -_EPS:
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[4] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Curvature constraint
                # **************************
                kappa_max = np.tan(self.vehicle_params.delta_max) / self.vehicle_params.wheelbase
                if abs(kappa_gl[i]) > kappa_max:
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[5] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Yaw rate constraint
                # **************************
                yaw_rate = (theta_gl[i] - theta_gl[i - 1]) / self.dT if i > 0 else 0.
                theta_dot_max = kappa_max * v[i]
                if abs(round(yaw_rate, 5)) > theta_dot_max:
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[6] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Curvature rate constraint
                # **************************
                # steering_angle = np.arctan2(self.vehicle_params.wheelbase * kappa_gl[i], 1.0)
                # kappa_dot_max = self.vehicle_params.v_delta_max / (self.vehicle_params.wheelbase *
                #                                                    math.cos(steering_angle) ** 2)
                kappa_dot = (kappa_gl[i] - kappa_gl[i - 1]) / self.dT if i > 0 else 0.
                if abs(kappa_dot) > 0.4:
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[7] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Acceleration rate constraint
                # **************************
                v_switch = self.vehicle_params.v_switch
                a_max = self.vehicle_params.a_max * v_switch / v[i] if v[i] > v_switch else self.vehicle_params.a_max
                a_min = -self.vehicle_params.a_max
                if not a_min <= a[i] <= a_max:
                    trajectory.feasible = False
                    infeasible_count_kinematics_traj[8] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

            # if selected polynomial trajectory is feasible, store it's Cartesian and Curvilinear trajectory
            if trajectory.feasible or self._draw_traj_set:
                for i in range(0, len(s)):
                    # compute (global) Cartesian position
                    pos: np.ndarray = self.coordinate_system.convert_to_cartesian_coords(s[i], d[i])
                    if pos is not None:
                        x[i] = pos[0]
                        y[i] = pos[1]
                    else:
                        trajectory.valid = False
                        infeasible_count_kinematics_traj[9] = 1
                        self.msg_logger.debug("Out of projection domain")
                        break

                if trajectory.feasible or self._draw_traj_set:
                    # store Cartesian trajectory
                    trajectory.cartesian = CartesianSample(x, y, theta_gl, v, a, kappa_gl,
                                                           kappa_dot=np.append([0], np.diff(kappa_gl)),
                                                           current_time_step=traj_len)

                    # store Curvilinear trajectory
                    trajectory.curvilinear = CurviLinearSample(s, d, theta_cl,
                                                               ss=s_velocity, sss=s_acceleration,
                                                               dd=d_velocity, ddd=d_acceleration,
                                                               current_time_step=traj_len)

                    trajectory.actual_traj_length = traj_len

                    # check if trajectories planning horizon is shorter than expected and extend if necessary
                    # if self.N + 1 > trajectory.cartesian.current_time_step:
                    #    trajectory.enlarge(self.dT)

                trajectory_list.append(trajectory)

            infeasible_invalid_count_kinematics += infeasible_count_kinematics_traj

        if self._multiproc:
            # store feasible trajectories in Queue 1
            queue_1.put(trajectory_list)
            if self._kinematic_debug:
                queue_2.put(infeasible_invalid_count_kinematics)
        else:
            return trajectory_list

    def _compute_standstill_trajectory(self) -> TrajectorySample:
        """
        Computes a standstill trajectory if the vehicle is already at velocity 0
        :return: The TrajectorySample for a standstill trajectory
        """
        # current planner initial state
        x_0 = self.x_0
        x_0_lon, x_0_lat = self.x_cl

        # create artificial standstill trajectory
        self.msg_logger.debug('Adding standstill trajectory')
        self.msg_logger.debug("x_0 is {}".format(x_0))
        self.msg_logger.debug("x_0_lon is {}".format(x_0_lon))
        self.msg_logger.debug("x_0_lon is {}".format(type(x_0_lon)))

        # create lon and lat polynomial
        traj_lon = QuarticTrajectory(tau_0=0, delta_tau=self.horizon, x_0=np.asarray(x_0_lon),
                                     x_d=np.array([0, 0]))
        traj_lat = QuinticTrajectory(tau_0=0, delta_tau=self.horizon, x_0=np.asarray(x_0_lat),
                                     x_d=np.array([x_0_lat[0], 0, 0]))

        # compute initial ego curvature (global coordinates) from initial steering angle
        kappa_0 = np.tan(x_0.steering_angle) / self.vehicle_params.wheelbase

        # create Trajectory sample
        p = TrajectorySample(self.horizon, self.dT, traj_lon, traj_lat, uniqueId=0,
                             costMap=self.cost_function.cost_weights)

        # create Cartesian trajectory sample
        a = np.repeat(0.0, self.N)
        a[1] = - self.x_0.velocity / self.dT
        p.cartesian = CartesianSample(np.repeat(x_0.position[0], self.N), np.repeat(x_0.position[1], self.N),
                                      np.repeat(x_0.orientation, self.N), np.repeat(0.0, self.N),
                                      a, np.repeat(kappa_0, self.N), np.repeat(0.0, self.N),
                                      current_time_step=self.N)

        # create Curvilinear trajectory sample
        # compute orientation in curvilinear coordinate frame
        s_idx = np.argmax(self.coordinate_system.ref_pos > x_0_lon[0]) - 1
        ref_theta = np.unwrap(self.coordinate_system.ref_theta)
        theta_cl = x_0.orientation - interpolate_angle(x_0_lon[0], self.coordinate_system.ref_pos[s_idx], self.coordinate_system.ref_pos[s_idx + 1],
                                                       ref_theta[s_idx], ref_theta[s_idx + 1])

        p.curvilinear = CurviLinearSample(np.repeat(x_0_lon[0], self.N), np.repeat(x_0_lat[0], self.N),
                                          np.repeat(theta_cl, self.N), dd=np.repeat(x_0_lat[1], self.N),
                                          ddd=np.repeat(x_0_lat[2], self.N), ss=np.repeat(x_0_lon[1], self.N),
                                          sss=np.repeat(x_0_lon[2], self.N), current_time_step=self.N)
        return p

    def _create_end_point_trajectory_bundle(self, x_0_lon, x_0_lat, stop_point_s, cost_function, samp_level):
        self.msg_logger.debug('sampling stopping trajectory at stop line')
        # reset cost statistic
        self._min_cost = 10 ** 9
        self._max_cost = 0

        trajectories = list()

        self.sampling_handler.set_s_sampling((x_0_lon[0]+stop_point_s)/2, stop_point_s)

        for t in self.sampling_handler.t_sampling.to_range(samp_level):
            # Longitudinal sampling for all possible velocities
            for s in self.sampling_handler.s_sampling.to_range(samp_level):
                end_state_lon = np.array([s, 0.0, 0.0])
                trajectory_long = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=end_state_lon)

                # Sample lateral end states (add x_0_lat to sampled states)
                if trajectory_long.coeffs is not None:
                    for d in self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]}):
                        end_state_lat = np.array([d, 0.0, 0.0])
                        # SWITCHING TO POSITION DOMAIN FOR LATERAL TRAJECTORY PLANNING
                        if self._LOW_VEL_MODE:
                            s_lon_goal = trajectory_long.evaluate_state_at_tau(t)[0] - x_0_lon[0]
                            if s_lon_goal <= 0:
                                s_lon_goal = t
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=s_lon_goal, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)

                        # Switch to sampling over t for high velocities
                        else:
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)
                        if trajectory_lat.coeffs is not None:
                            trajectory_sample = TrajectorySample(self.horizon, self.dT, trajectory_long, trajectory_lat,
                                                                 len(trajectories), costMap=self.cost_function.cost_weights)
                            trajectories.append(trajectory_sample)

        # perform pre-check and order trajectories according their cost
        trajectory_bundle = TrajectoryBundle(trajectories, cost_function=cost_function,
                                             multiproc=self._multiproc, num_workers=self._num_workers)
        self._total_count = len(trajectory_bundle._trajectory_bundle)
        self.msg_logger.debug('%s trajectories sampled' % len(trajectory_bundle._trajectory_bundle))

        return trajectory_bundle


