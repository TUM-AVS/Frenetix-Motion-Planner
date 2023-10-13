__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from abc import abstractmethod, ABC
from typing import List

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario


class PlannerInterface(ABC):

    @abstractmethod
    def __init__(self, config, scenario: Scenario,
                 planning_problem: PlanningProblem,
                 log_path: str, mod_path: str):
        """Wrappers providing a consistent interface for different planners.
        To be implemented for every specific planner.

        :param config: Object containing the configuration of the planner.
        :param scenario: The commonroad scenario to be simulated.
        :param planning_problem: The PlanningProblem for this planner.
        :param log_path: Path the planner's log files will be written to.
        :param mod_path: Working directory for the planner.
        """
        raise NotImplementedError()

    def get_all_traj(self):
        """Return the sampled trajectories of the last step for plotting.

        If plotting of trajectory bundles is not required, leave as is.
        """
        return None

    def get_ref_path(self):
        """Return the reference path of the planner for plotting.

        If plotting of reference paths is not required, leave as is.
        """
        return None

    @abstractmethod
    def check_collision(self, ego_obstacle: List[DynamicObstacle], timestep: int):
        """Check for a collision at the given timestep.

        To be implemented by every specific planner.

        :param ego_obstacle: Dummy obstacles of the ego vehicle at every timestep.
        :param timestep: Time step at which to check for collisions.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_planner(self, scenario: Scenario, predictions: dict):
        """Update scenario for synchronization between agents.

        To be implemented for every specific planner.

        :param scenario: Updated scenario showing new agent positions.
        :param predictions: Updated predictions for all obstacles in the scenario.
        """
        raise NotImplementedError()

    @abstractmethod
    def plan(self):
        """Planner step function.

        To be implemented for every specific planner.

        :returns: Exit code of the planner step,
                  The planned trajectory.
        """
        raise NotImplementedError()