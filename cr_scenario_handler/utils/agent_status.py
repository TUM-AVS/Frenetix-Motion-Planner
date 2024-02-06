__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from enum import IntEnum
import cr_scenario_handler.utils.goalcheck as gc

"""Timeout value used when waiting for messages during parallel execution"""
TIMEOUT = 20


class AgentStatus(IntEnum):
    IDLE = -1
    RUNNING = 0
    COMPLETED_SUCCESS = 1
    COMPLETED_OUT_OF_TIME = 2
    COMPLETED_FASTER = 3
    MAX_S_POSITION = 4
    TIMELIMIT = 5
    ERROR = 6
    COLLISION = 7


class AgentState:
    def __init__(self, planning_problem, reference_path, coordinate_system):
        self.status = AgentStatus.IDLE
        self.last_timestep = planning_problem.initial_state.time_step
        self.message = None
        self.goal_status = False
        self.goal_message = None

        self.running = False
        self.goal_checker_status = None
        self.crashed = False
        self.goal_checker = gc.GoalReachedChecker(planning_problem, reference_path, coordinate_system)

    def update_goal_checker(self, planning_problem, reference_path, coordinate_system):
        self.goal_checker = gc.GoalReachedChecker(planning_problem, reference_path, coordinate_system)

    def check_goal_reached(self, record_state_list, x_cl):
        """Check for completion of the planner.

        :return: True iff the goal area has been reached.
        """

        self.goal_checker.register_current_state(record_state_list[-1], x_cl)
        self.running, self.status, self.goal_checker_status, self.message = self.goal_checker.goal_reached_status(AgentStatus)

    def log_running(self, timestep):
        self.status = AgentStatus.RUNNING
        self.last_timestep = timestep
        self.message = "running"

    def log_collision(self, timestep):
        self.status = AgentStatus.COLLISION
        self.last_timestep = timestep
        self.message = "collision" if timestep > 0 else "initial state already crashed"

    def log_finished(self, timestep):
        self.last_timestep = timestep
        self.goal_status = True
        if self.status == AgentStatus.COMPLETED_SUCCESS:
            self.message = "goal reached successful"
        elif self.status == AgentStatus.COMPLETED_FASTER:
            self.message = "goal reached faster"
        elif self.status == AgentStatus.COMPLETED_OUT_OF_TIME:
            self.message = "goal reached out of time"
        else:
            self.message = "ERROR: not valid finishing state"

    def log_max_s_position(self, timestep):
        self.status = AgentStatus.MAX_S_POSITION
        self.last_timestep = timestep
        self.message = "Maximum S-Position reached"

    def log_timelimit(self, timestep):
        self.status = AgentStatus.TIMELIMIT
        self.last_timestep = timestep
        self.message = "timelimit reached"

    def log_error(self, timestep):
        self.status = AgentStatus.ERROR
        self.last_timestep = timestep
        self.message = "no valid or feasible trajectory found"
