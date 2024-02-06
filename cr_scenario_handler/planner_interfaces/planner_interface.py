__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from abc import abstractmethod, ABC
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario


class PlannerInterface(ABC):

    @abstractmethod
    def __init__(self, agent_id: int, config_planner, config_sim, scenario: Scenario,
                 planning_problem: PlanningProblem,
                 log_path: str, mod_path: str):
        """Wrappers providing a consistent interface for different planners.
        To be implemented for every specific planner.

        :param config_planner: The configuration object for the trajectory planner.
        :param config_sim: The configuration object for the simulation framework
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param planning_problem: The planning problem of the ego vehicle.
        :param log_path: Path for writing planner-specific log files to.
        :param mod_path: Working directory of the planner.
        """
        self.planner = None
        self.reference_path = None
        raise NotImplementedError()

    @property
    def all_trajectories(self):
        """Return the sampled trajectories of the last step for plotting."""
        return None

    @property
    def record_input_list(self):
        """Return the recorded input list of all states to the controller."""
        return None

    @property
    def record_state_list(self):
        """Return the recorded planner state lists of the vehicle."""
        return None

    @property
    def vehicle_history(self):
        """Return the recorded object state lists of the vehicle."""
        return None

    @property
    def coordinate_system(self):
        """Return the coordinate system of the planner."""
        return None

    @property
    def optimal_trajectory(self):
        """Return the optimal trajectory in a timestep of the planner."""
        return None

    @property
    def trajectory_pair(self):
        """Return the optimal trajectory pair in a timestep of the planner."""
        return None

    @abstractmethod
    def update_planner(self, scenario: Scenario, predictions: dict):
        """Update scenario for synchronization between agents.

        To be implemented for every specific planner.

        :param scenario: Updated scenario showing new agent positions.
        :param predictions: Updated predictions for all obstacles in the scenario.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_interface(self, current_timestep=None):
        """Planner step function.

        To be implemented for every specific planner.

        :returns: Exit code of the planner step,
                  The planned trajectory.
        """
        raise NotImplementedError()

