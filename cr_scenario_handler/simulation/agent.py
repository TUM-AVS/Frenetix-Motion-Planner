__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# standard imports
import time
import warnings
from copy import deepcopy
from typing import List

# third party
import numpy as np

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InputState, CustomState
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

# reactive planner
from cr_scenario_handler.utils.configuration import Configuration
import frenetix_motion_planner.prediction_helpers as ph
from cr_scenario_handler.utils.goalcheck import GoalReachedChecker

# scenario handler
from cr_scenario_handler.utils.multiagent_helpers import trajectory_to_obstacle
from cr_scenario_handler.utils.visualization import visualize_multiagent_at_timestep, make_gif
from cr_scenario_handler.utils.evaluation import evaluate
from cr_scenario_handler.planner_interfaces.frenet_interface import FrenetPlannerInterface


class Agent:

    def __init__(self, agent_id: int, planning_problem: PlanningProblem,
                 scenario: Scenario, config: Configuration, log_path: str, mod_path: str):
        """Represents one agent of a multiagent or single-agent simulation.

        Manages the agent's local view on the scenario, the planning problem,
        planner interface, collision detection, and per-agent plotting and logging.
        Contains the step function of the agent.

        :param agent_id: The agent ID, equal to the obstacle_id of the
            DynamicObstacle it is represented by.
        :param planning_problem: The planning problem of this agent.
        :param config: The configuration.
        :param log_path: Path for logging and visualization.
        :param mod_path: working directory of the planner, containing planner configuration.
        """

        # List of dummy obstacles for past time steps
        self.ego_obstacle_list = list()
        # List of past states
        self.record_state_list = list()
        # List of input states for the planner
        self.record_input_list = list()
        # log of times required for planning steps
        self.planning_times = list()

        # Agent id, equals the id of the dummy obstacle
        self.id = agent_id

        self.planning_problem = planning_problem
        self.goal_checker = GoalReachedChecker(planning_problem)

        # Configuration
        self.config = config
        self.log_path = log_path
        self.mod_path = mod_path

        # Local view on the scenario, with dummy obstacles
        # for all agents except the ego agent
        self.scenario = None
        # initialize scenario: Remove dynamic obstacle for ego vehicle
        self.scenario = deepcopy(scenario)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not contained in the scenario")
            if self.scenario.obstacle_by_id(self.id) is not None:
                self.scenario.remove_obstacle(self.scenario.obstacle_by_id(self.id))

        # Initialize Planner
        self.planner = FrenetPlannerInterface(config, scenario, planning_problem, log_path, mod_path)

        self.current_timestep = self.planning_problem.initial_state.time_step
        self.max_timestep = int(self.config.general.max_steps *
                                planning_problem.goal.state_list[0].time_step.end)

        # Create states of the agent before the start of the simulation
        self.initialize_state_list()

        # Dummy obstacle for the agent
        state_list = deepcopy(self.record_state_list)
        self.ego_obstacle_list.append(trajectory_to_obstacle(state_list, config.vehicle, self.id))

        # add initial inputs to recorded input list
        self.record_input_list.append(InputState(
            acceleration=self.record_state_list[-1].acceleration,
            time_step=self.record_state_list[-1].time_step,
            steering_angle_speed=0.))

    def initialize_state_list(self):
        """ Initialize the recorded trajectory of the agent.

        Fills the state list before the agent's initial time step with empty states,
        and creates and inserts the initial state of the agent.
        """

        # In case of late startup, fill history with empty states
        for i in range(self.current_timestep):
            self.record_state_list.append(
                CustomState(time_step=i,
                            position=np.array([float("NaN"), float("NaN")]),
                            steering_angle=0, velocity=0, orientation=0,
                            acceleration=0, yaw_rate=0)
            )
        # Convert initial state to required format, append it to the state list
        self.record_state_list.append(
            CustomState(time_step=self.planning_problem.initial_state.time_step,
                        position=self.planning_problem.initial_state.position,
                        steering_angle=0,
                        velocity=self.planning_problem.initial_state.velocity,
                        orientation=self.planning_problem.initial_state.orientation,
                        acceleration=self.planning_problem.initial_state.acceleration,
                        yaw_rate=self.planning_problem.initial_state.yaw_rate)
        )

    def update_scenario(self, outdated_agents: List[int], dummy_obstacles: List[DynamicObstacle]):
        """ Update the scenario to synchronize the agents.

        :param outdated_agents: Obstacle IDs of all dummy obstacles that need to be updated
        :param dummy_obstacles: New dummy obstacles
        """

        # Remove outdated obstacles
        for i in outdated_agents:
            # ego does not have a dummy of itself,
            # joining agents may not yet be in the scenario
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not contained in the scenario")
                if self.scenario.obstacle_by_id(i) is not None:
                    self.scenario.remove_obstacle(self.scenario.obstacle_by_id(i))

        # Add all dummies except of the ego one
        self.scenario.add_objects(list(filter(lambda obs: not obs.obstacle_id == self.id, dummy_obstacles)))

    def is_completed(self):
        """Check for completion of the planner.

        :return: True iff the goal area has been reached.
        """
        self.goal_checker.register_current_state(self.record_state_list[-1])
        goal_status, _, _ = self.goal_checker.goal_reached_status()
        return goal_status

    def step_agent(self, global_predictions):
        """ Execute one planning step.

        Checks for collisions, filters the predictions by visibility,
        calls the step function of the planner, extends the recorded trajectory,
        creates the updated dummy obstacle for synchronization,
        records planning times and handles per-agent plotting.

        :param global_predictions: Dictionary of predictions for all obstacles and agents

        :returns: status, ego_obstacle
            where status is:
                0: if successful.
                1: if completed.
                2: on error.
                3: on collision.
            and ego_obstacle is:
                The dummy obstacle at the new position of the agent,
                    including both history and planned trajectory,
                or None if status > 0
        """

        print(f"[Agent {self.id}] current time step: {self.current_timestep}")

        # check for completion of this agent
        if self.is_completed() or self.current_timestep >= self.max_timestep:
            self.finalize()
            return 1, None

        # Check for collisions in previous timestep
        if self.current_timestep > 1 and self.ego_obstacle_list[-1] is not None:
            crash = self.planner.check_collision(self.ego_obstacle_list, self.current_timestep-1)
            if crash:
                print(f"[Agent {self.id}] Collision Detected!")
                self.finalize()
                return 3, None

        # Process new predictions
        predictions = dict()
        visible_obstacles, visible_area = ph.prediction_preprocessing(
            self.scenario, self.ego_obstacle_list[-1], self.current_timestep, self.config, self.id
        )
        for obstacle_id in visible_obstacles:
            # Handle obstacles with higher initial timestep
            if obstacle_id in global_predictions.keys():
                predictions[obstacle_id] = global_predictions[obstacle_id]
        if self.config.prediction.cone_angle > 0 \
                and self.config.prediction.mode == "walenet":
            predictions = ph.ignore_vehicles_in_cone_angle(predictions, self.ego_obstacle_list[-1],
                                                           self.config.vehicle.length,
                                                           self.config.prediction.cone_angle,
                                                           self.config.prediction.cone_safety_dist)

        # START TIMER
        comp_time_start = time.time()

        # Execute planner step
        self.planner.update_planner(self.scenario, predictions)
        error, new_trajectory = self.planner.plan()

        comp_time_end = time.time()
        # END TIMER

        # if the planner fails to find an optimal trajectory -> terminate
        if error:
            print(f"[Agent {self.id}] Could not plan trajectory: Planner returned {error}")
            self.finalize()
            return 2, None

        # store planning times
        self.planning_times.append(comp_time_end - comp_time_start)
        print(f"[Agent {self.id}] ***Total Planning Time: {self.planning_times[-1]}")

        # get next state from state list of planned trajectory
        new_state = new_trajectory.state_list[1]
        new_state.time_step = self.current_timestep + 1

        # add input to recorded input list
        self.record_input_list.append(InputState(
            acceleration=new_state.acceleration,
            steering_angle_speed=(new_state.steering_angle - self.record_state_list[-1].steering_angle)
                                     / self.config.planning.dt,
            time_step=new_state.time_step
        ))
        # add new state to recorded state list
        self.record_state_list.append(new_state)

        # commonroad obstacle for synchronization between agents
        # List of all past and future states.
        state_list = deepcopy(self.record_state_list)
        state_list.extend(new_trajectory.state_list[2:])
        self.ego_obstacle_list.append(trajectory_to_obstacle(state_list, self.config.vehicle, self.id))

        # plot own view on scenario
        if self.id in self.config.multiagent.show_specific_individual_plots or \
                self.config.multiagent.show_all_individual_plots or \
                self.id in self.config.multiagent.save_specific_individual_plots or \
                self.config.multiagent.save_all_individual_plots:
            visualize_multiagent_at_timestep(scenario=self.scenario,
                                             planning_problem_set=PlanningProblemSet([self.planning_problem]),
                                             agent_list=[self.ego_obstacle_list[-1]],
                                             timestep=self.current_timestep,
                                             config=self.config, log_path=self.log_path,
                                             traj_set_list=[self.planner.get_all_traj()],
                                             ref_path_list=[self.planner.get_ref_path()],
                                             predictions=predictions, visible_area=visible_area,
                                             plot_window=self.config.debug.plot_window_dyn)

        self.current_timestep += 1
        return 0, self.ego_obstacle_list[-1]

    def finalize(self):
        """ Execute post-simulation tasks.

        Create a gif from plotted images, and run the evaluation function.
        """

        # make gif
        if self.id in self.config.multiagent.save_specific_individual_gifs or \
                self.config.multiagent.save_all_individual_gifs:
            make_gif(self.scenario,
                     range(self.planning_problem.initial_state.time_step,
                           self.current_timestep),
                     self.log_path, duration=0.1)

        # evaluate driven trajectory
        if self.config.debug.evaluation:
            evaluate(self.scenario, self.planning_problem, self.id,
                     self.record_state_list, self.record_input_list,
                     self.config, self.log_path)
