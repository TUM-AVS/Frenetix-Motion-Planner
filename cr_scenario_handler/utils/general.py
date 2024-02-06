__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import csv
from commonroad.common.file_reader import CommonRoadFileReader
import random
from os import listdir
from os.path import isfile, join


def load_scenario_and_planning_problem(config, idx_planning_problem: int = 0):
    scenario, planning_problem_set = CommonRoadFileReader(config.simulation.scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[
        idx_planning_problem
    ]

    # if config.planning.dt != scenario.dt:
    #     scenario, planning_problem_set = interpolate_scenario(scenario=scenario, dt_new=config.planning.dt,
    #                                                           planning_problem_set=planning_problem_set)
    #
    # scenario = check_if_obstacle_on_same_position(scenario, planning_problem.initial_state.position)

    return scenario, planning_problem, planning_problem_set


def check_if_obstacle_on_same_position(scenario, init_state):
    init_pos = np.array((init_state[0], init_state[1]))
    for i in scenario.dynamic_obstacles:
        if i.prediction.initial_time_step < 2:
            obs_pos = np.array((i.initial_state.position[0], i.initial_state.position[1]))
            # Euclidean distance
            if np.linalg.norm(init_pos - obs_pos) < 2:
                scenario.remove_obstacle(i)
    return scenario


def read_scenario_list(scenario_list_path) -> list:

    # Create an empty list to store the lines
    scenario_names = []

    # Open the CSV and read each line
    with open(scenario_list_path, 'r') as infile:
        reader = csv.reader(infile)

        # Append each line to the list
        for row in reader:
            scenario_without_xml = row[0]
            if scenario_without_xml not in scenario_names:
                scenario_names.append(scenario_without_xml)

    return scenario_names


def get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, example_scenarios_list,
                      use_specific_scenario_list):
    """Saves all scenarios that sould be processed in one list. Its also the logic to find the right settings.

    Args:
        scenario_name (str): considered scenario
        scenario_folder(str): folder of the scenarios
        evaluation_pipeline (bool): use evaluation pipeline
        example_scenarios_list: path to scenario list
        use_specific_scenario_list (bool) use scenario list boolean
    """
    scenario_files = []
    if not evaluation_pipeline and not use_specific_scenario_list:
        scenario_files = [scenario_name]
    elif not use_specific_scenario_list and evaluation_pipeline:
        scenario_files = [f.split(".")[-2] for f in listdir(scenario_folder) if isfile(join(scenario_folder, f))]
        random.shuffle(scenario_files)
    elif use_specific_scenario_list and evaluation_pipeline and example_scenarios_list:
        scenario_files = read_scenario_list(example_scenarios_list)
        random.shuffle(scenario_files)
    elif use_specific_scenario_list and not evaluation_pipeline:
        raise IOError("Scenario list can only be used if evaluation pipeline is activated!")
    return scenario_files
