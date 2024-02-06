__author__ = "Alexander Hobmeier, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from omegaconf import OmegaConf
from frenetix_motion_planner.trajectories import TrajectorySample
import frenetix_motion_planner.cost_functions.partial_cost_functions as cost_functions


class CostFunction(ABC):
    """
    Abstract base class for new cost functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, trajectories: List[TrajectorySample]):
        """
        Computes the costs of a given trajectory sample
        :param trajectory: The trajectory sample for the cost computation
        :return: The cost of the given trajectory sample
        """
        pass


class AdaptableCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(self, rp, configuration):
        super(AdaptableCostFunction, self).__init__()

        self.scenario = None
        self.rp = rp
        self.reachset = None
        self.desired_speed = None
        self.predictions = None
        self.configuration = configuration

        self.vehicle_params = rp.vehicle_params
        self.cost_weights = OmegaConf.to_object(configuration.cost.cost_weights)
        self.save_unweighted_costs = configuration.debug.save_unweighted_costs
        self.cost_weights_names = None

        for category in list(self.cost_weights.keys()):
            if self.cost_weights[category] == 0:
                del self.cost_weights[category]
        names = list(self.cost_weights.keys())
        names.sort()
        self.cost_weights_names = names

        self.functions = dict()
        for num, function in enumerate(self.cost_weights_names):
            self.functions[function] = getattr(cost_functions, str(function) + "_costs")

    def update_state(self, scenario, rp, predictions, reachset):
        self.scenario = scenario
        self.rp = rp
        self.reachset = reachset
        self.desired_speed = rp.desired_velocity
        self.predictions = predictions

    # calculate all costs for all trajcetories
    def evaluate(self, trajectories: List[TrajectorySample]):
        self.calc_cost(trajectories)

    # calculate all costs and weigh them
    def calc_cost(self, trajectories: List[TrajectorySample]):

        for i, trajectory in enumerate(trajectories):
            costlist = np.zeros(len(self.cost_weights))
            costlist_weighted = np.zeros(len(self.cost_weights))

            for num, function in enumerate(self.cost_weights_names):
                costlist[num] = self.functions[function](trajectory=trajectory, planner=self.rp,
                                                         scenario=self.scenario, desired_speed=self.desired_speed)
                costlist_weighted[num] = self.cost_weights[function] * costlist[num]

            total_cost = np.sum(costlist_weighted)

            trajectory.set_costs(total_cost, costlist, costlist_weighted, self.cost_weights_names)

