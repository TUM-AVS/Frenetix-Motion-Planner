__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os

from typing import List


def init_log(log_path: str):
    """Create log file for simulation-level logging and write the header.

    The log file will contain the following fields:
    time_step: The time step of this log entry.
    domain_time: The time step expressed in simulated time.
    total_planning_time: The wall clock time required for executing the planner step of all agents.
    total_synchronization_time: The wall clock time required for synchronizing the agents.
    agent_ids: A list of the IDs of all agents in the simulation.
    agent_states: A list of the return values of the agents' step function,
        in the same order as the agent_ids

    :param log_path: Base path the log file is written to
    """

    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, "execution_logs.csv"), "w+") as log_file:
        log_file.write("time_step;domain_time;total_planning_time;total_synchronization_time;agent_ids;agent_states;")


def append_log(log_path: str, time_step: int, domain_time: float, total_planning_time: float,
               total_synchronization_time: float, agent_ids: List[int], agent_states: List[int]):
    """Write the log entry for one simulation step.

    :param log_path: Path to the directory containing the log file
    :param time_step: Number of the current time step
    :param domain_time: Current time inside the simulation
    :param total_planning_time: Wall clock time for stepping all agents
    :param total_synchronization_time: Wall clock time for exchanging dummy obstacles
    :param agent_ids: List of all agent ids in the scenario
    :param agent_states: Return codes from all agents
    """

    entry = "\n"
    entry += str(time_step) + ";"
    entry += str(domain_time) + ";"
    entry += str(total_planning_time) + ";"
    entry += str(total_synchronization_time) + ";"
    entry += str(agent_ids) + ";"
    entry += str(agent_states) + ";"

    with open(os.path.join(log_path, "execution_logs.csv"), "a") as log_file:
        log_file.write(entry)
