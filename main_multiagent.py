import csv
import os
import shutil
import sys
import traceback
from datetime import datetime

from cr_scenario_handler.evaluation.simulation_evaluation import evaluate_simulation
from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.utils.general import get_scenario_list


def start_simulation(scenario_name, scenario_folder, mod_path, logs_path, count=0, use_cpp=True, start_multiagent=True):
    log_path = os.path.join(logs_path, scenario_name)
    config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
    config_sim.simulation.use_multiagent = start_multiagent

    config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name)
    config_planner.debug.use_cpp = use_cpp

    simulation = None
    evaluation = None

    try:
        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()
        if config_sim.evaluation.evaluate_simulation:
            evaluation = evaluate_simulation(simulation)
        # close sim_logger
        simulation.sim_logger.con.close()

    except Exception as e:
        try:
            # close sim_logger
            simulation.sim_logger.con.close()
            # close child processes
            simulation.close_processes()
        except:
            pass
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open('logs/log_failures.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            writer.writerow([str(count) +" ; " + "Scenario Name: " + log_path.split("/")[-1] + " ; " +
                             "Error time: " + str(current_time) + " ; " +
                             "In Scenario Timestep: " + current_timestep + " ; " +
                             "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n"])
        raise Exception
    return simulation, evaluation


def main():
    if sys.platform == "darwin":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    mod_path = os.path.dirname(os.path.abspath(__file__))
    logs_path = os.path.join(mod_path, "logs")

    # *********************************************************
    # Link a Scenario Folder & Start many Scenarios to evaluate
    # *********************************************************
    evaluation_pipeline = False

    # **********************************************************************
    # If the previous are set to "False", please specify a specific scenario
    # **********************************************************************
    scenario_name = "ZAM_Tjunction-1_42_T-1"  # do not add .xml format to the name
    scenario_folder = os.path.join(mod_path, "example_scenarios")
    scenario_files = get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, None, False)

    # ***************************************************
    # Delete former logs & Create new score overview file
    # ***************************************************
    delete_former_logs = True
    if delete_former_logs:
        shutil.rmtree(logs_path, ignore_errors=True)
    os.makedirs(logs_path, exist_ok=True)
    if not os.path.exists(os.path.join(logs_path, "score_overview.csv")):
        os.makedirs(logs_path, exist_ok=True)
        with open(os.path.join(logs_path, "score_overview.csv"), 'a') as file:
            line = "scenario;agent;timestep;status;message\n"
            file.write(line)

    if evaluation_pipeline:
        count = 0
        for scenario_file in scenario_files:
            start_simulation(scenario_file, scenario_folder, mod_path, logs_path, count)
            count += 1

    else:
        # If not in evaluation_pipeline mode, just run one scenario
        simulation_result, evaluation_result = start_simulation(scenario_files[0], scenario_folder, mod_path, logs_path)
        return simulation_result, evaluation_result


if __name__ == '__main__':
    simulation, evaluation = main()
