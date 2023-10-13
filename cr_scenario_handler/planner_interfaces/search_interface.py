__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from commonroad.scenario.scenario import Scenario

from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig

from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface


class SearchInterface(PlannerInterface):

    def __init__(self, config, scenario, planning_problem, log_path, mod_path):
        self.config = config
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.log_path = log_path

        name_file_motion_primitives = '../../../commonroad-search/SMP/maneuver_automaton/primitives/V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
        automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)
        self.planner = AStarSearch(scenario, planning_problem, automaton, plot_config=DefaultPlotConfig)

        self.path = []

    def check_collision(self, ego_vehicle_list, timestep):
        return not self.planner.is_collision_free(self.path)

    def update_planner(self, scenario: Scenario, _: dict):
        self.scenario = scenario

        self.planner.scenario = scenario
        self.planner.collision_checker = create_collision_checker(scenario)

    def plan(self):

        result = self.planner.execute_search()
        state_list = result[0]
        if state_list is None:
            return 1, None
        else:
            self.path = state_list
            return 0, state_list
