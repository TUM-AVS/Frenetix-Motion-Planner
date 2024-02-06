__author__ = "Marc Kaufeld,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from abc import ABC
import pandas as pd

from cr_scenario_handler.evaluation.metrics import Measures
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from commonroad.common.solution import VehicleType


class Evaluator(ABC):
    def __init__(self, config_sim, scenario, msg_logger, sim_logger, agent_ids=None, agents=None):
        self.sim_logger = sim_logger
        self.msg_logger = msg_logger
        self.id = scenario.scenario_id
        self.config = config_sim.evaluation
        self.scenario = scenario

        self.start_min = None
        self.end_max = None

        self.measures = []
        self.set_measures(self.config)

        self._agents = None
        self._agent_ids = None
        self._df_criticality = None
        self.reference_paths = dict()
        self.set_agents(agent_ids, agents)

    @property
    def agent_ids(self):
        return self._agent_ids

    @property
    def agents(self):
        return self._agents

    @property
    def criticality_dataframe(self):
        return self._df_criticality

    def set_measures(self, config):
        """set the measures specified in the configuration"""
        self.measures = [metric for metric, used
                         in config.criticality_metrics.items()
                         if hasattr(Measures, metric) and used]

    def set_agents(self, agent_id=None, agents=None):
        """set evaluated agents"""
        if agents and not agent_id:
            agent_id = [i.id for i in agents]
        elif not agent_id and not agents:
            agent_id = [obs.id for obs in self.scenario.dynamic_obstacles]
        # get the earliest start and latest end time step
        self.start_min = min(self.scenario.obstacle_by_id(agent).initial_state.time_step for agent in agent_id)
        self.end_max = max(self.scenario.obstacle_by_id(agent).prediction.final_time_step for agent in agent_id)

        self._agent_ids = agent_id

    def evaluate(self):
        """run evaluation of simulation for each agent considered"""
        for idx, agent_id in enumerate(self.agent_ids):
            self.msg_logger.critical(f"Evaluate agent {agent_id}")
            self.msg_logger.info(f"metrics: {self.measures}")
            agent_results = self.evaluate_agent(agent_id)
            # store results in dataframe
            if self._df_criticality is None:
                self._df_criticality = agent_results.set_index([[agent_id] * len(agent_results), agent_results.index])
            else:
                df_to_append = agent_results.set_index([[agent_id] * len(agent_results), agent_results.index])
                self._df_criticality = pd.concat([self._df_criticality, df_to_append])

    def evaluate_agent(self, agent_id):
        """
        evaluate agent with given id
        :param agent_id:
        :return: dataframe with evaluation results
        """
        t_start = self.scenario.obstacle_by_id(agent_id).initial_state.time_step
        t_end = self.scenario.obstacle_by_id(agent_id).prediction.final_time_step
        a_max_long = VehicleParameterMapping.from_vehicle_type(VehicleType(2)).longitudinal.a_max
        a_max_lat = self.config.a_max_lat
        radius = self.config.radius
        tau = self.config.tau
        results = pd.DataFrame(None, index=list(range(t_start, t_end + 1)), columns=self.measures)
        measures = Measures(agent_id, self.scenario, a_max_long, a_max_lat, radius, tau, t_start, t_end,
                            self.msg_logger)
        for measure in self.measures:
            value = getattr(measures, measure)()
            results[measure] = value


        self.msg_logger.debug("Evaluation completed")
        return results

    def log_results(self):
        """write results into sql database"""
        self.sim_logger.log_evaluation(self.criticality_dataframe)


def evaluate_simulation(simulation, vehicle_ids=None):
    config = simulation.config
    if not config.evaluation.evaluate_simulation:
        return

    agents = simulation.agents
    agent_ids = vehicle_ids if vehicle_ids else simulation.agent_id_list
    scenario = simulation.scenario
    sim_logger = simulation.sim_logger
    msg_logger = simulation.msg_logger

    evaluation = Evaluator(config, scenario, msg_logger, sim_logger, agent_ids, agents)
    evaluation.evaluate()
    evaluation.log_results()
    return evaluation
