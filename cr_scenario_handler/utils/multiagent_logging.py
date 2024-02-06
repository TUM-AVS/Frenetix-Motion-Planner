__author__ = "Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import csv
import json
import logging
import os
import sqlite3
import sys

from omegaconf import DictConfig, ListConfig

from cr_scenario_handler.utils.agent_status import AgentStatus, TIMEOUT
from cr_scenario_handler.utils.configuration import Configuration


class SimulationLogger:
    """
        Simulation logging object, which writes the data collected during the simulation to a sql database
    """
    @staticmethod
    def _convert_config(config: Configuration) -> dict:
        """converts config to dict for writing configinto sql database"""
        data = dict()
        for item in config.__dict__:
            # print(item)
            ii = getattr(config, item)
            data[item] = dict()
            for it in ii.__dict__:
                val = ii.__dict__[it]
                if isinstance(val, DictConfig):
                    valdic = dict()
                    for key, value in val.items():
                        if isinstance(value, ListConfig):
                            valdic[key] = list(value)
                        elif isinstance(value, DictConfig):
                            valdic[key] = dict(value)
                        else:
                            valdic[key] = value
                    val = valdic
                if isinstance(val, ListConfig):
                    val = list(val)

                data[item][it] = val

        return data

    def __init__(self, config):
        """"""

        self.config = config
        self.eval_conf = config.evaluation
        self.scenario = self.config.simulation.name_scenario
        self.original_planning_problem_id = None
        log_path = self.config.simulation.log_path
        self.log_path = log_path if self.scenario not in log_path else log_path.replace(self.scenario, "")
        self.log_time = self.eval_conf.evaluate_runtime
        self.scenario = self.config.simulation.name_scenario

        os.makedirs(self.log_path, exist_ok=True)

        self.con = sqlite3.connect(os.path.join(self.log_path, "simulation.db"), timeout=TIMEOUT,
                                                isolation_level="EXCLUSIVE"
                                   )

        self.con.executescript("""
            PRAGMA journal_mode = OFF;
            PRAGMA temp_store = MEMORY;
        """)
        self.con.commit()

        self.create_tables()

    def create_tables(self):
        if self.log_time:
            # Table for main simulation time measurement
            self.con.execute("""
                    CREATE TABLE  IF NOT EXISTS global_performance_measure(
                        scenario TEXT NOT NULL,
                        time_step INT NOT NULL,
                        total_sim_time REAL NOT NULL,
                        global_sim_preprocessing REAL,
                        global_batch_synchronization REAL,
                        global_visualization REAL,
                        --PRIMARY KEY(scenario, time_step)
                        PRIMARY KEY(scenario, time_step)
                       ) STRICT
                   """)


            # Table for batch simulation time measurement
            self.con.execute("""
                    CREATE TABLE  IF NOT EXISTS batch_performance_measure(
                        scenario TEXT NOT NULL,
                        batch TEXT NOT NULL,
                        time_step INT NOT NULL,
                        process_iteration_time REAL,
                        sim_step_time REAL NOT NULL,
                        agent_planning_time REAL NOT NULL,
                        sync_time_in REAL,
                        sync_time_out REAL,
                        -- PRIMARY KEY(scenario, batch, time_step)
                        PRIMARY KEY(scenario, batch, time_step)
                       ) STRICT
                   """)


        # Table for general information (Scenarios
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                scenario TEXT NOT NULL ,
                num_agents INT Not NULL,
                agent_ids ANY,
                original_planning_problem_id ANY,
                agent_batches ANY,
                duration_init REAL NOT NULL, 
                sim_duration REAL,
                post_duration REAL,
                simulation_config ANY NOT NULL,
                planner_config TEXT NOT NULL,
                PRIMARY KEY(scenario)
            ) STRICT
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS results(
                scenario TEXT NOT NULL ,
                agent_id INT NOT NULL,
                original_planning_problem INTEGER NOT NULL ,
                final_status INTEGER NOT NULL,
                last_timestep INTEGER NOT NULL ,
                message TEXT,
                agent_success TEXT NOT NULL,
                PRIMARY KEY(scenario, agent_id)
            ) STRICT
        """)

        if self.eval_conf.evaluate_simulation:

            columns = ' ANY, '.join(key for key, value in self.eval_conf.criticality_metrics.items() if value==True) + ' ANY,'
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS scenario_evaluation(
                    scenario TEXT NOT NULL,
                    agent_id INT NOT NULL,
                    original_planning_problem INTEGER NOT NULL,
                    timestep INT NOT NULL,""" 
                    f"{columns}"
                   """ PRIMARY KEY(scenario, agent_id, timestep)
                ) STRICT 
            """)


        self.con.commit()


    def log_evaluation(self, results):
        """
        Log the criticality evaluation results
         """
        data = []
        for (agent_id, t) in results.index:
            orig_pp = True if agent_id in self.original_planning_problem_id else False
            data.append([self.scenario, agent_id+10, orig_pp, t] + list(results.loc[(agent_id, t)]))
        text = "INSERT INTO scenario_evaluation VALUES(" + "?," *len(data[0])
        text = text[:-1] + ")"
        self.con.executemany(text, data)
        self.con.commit()

    def log_results(self,agents):
        """
        Log the planning results of the simulation
         """
        data = []
        for agent in agents:
            orig_pp = True if agent.id in self.original_planning_problem_id else False
            success = "success" if agent.status == AgentStatus.COMPLETED_SUCCESS else "failed"
            data.append([self.scenario, agent.id, orig_pp, agent.status, agent.agent_state.last_timestep, agent.agent_state.message, success])
        self.con.executemany("INSERT INTO results VALUES(?,?,?,?,?,?,?)", data)
        self.con.commit()

    def log_meta(self, agent_ids, original_planning_problem_id, batch_names, duration_init, config_sim, config_planner):
        """
        Log the meta information of the simulation
        """
        self.original_planning_problem_id = original_planning_problem_id
        conf_sim = json.dumps(SimulationLogger._convert_config(config_sim))
        conf_plan = json.dumps(SimulationLogger._convert_config(config_planner))
        data = [self.scenario, len(agent_ids), json.dumps(agent_ids), json.dumps(original_planning_problem_id), json.dumps(batch_names), duration_init, None, None,conf_sim, conf_plan]
        self.con.execute("INSERT INTO meta VALUES(?,?,?,?,?,?,?,?,?,?)", data)
        self.con.commit()

    def update_meta(self, **kwargs):
        """
        Update the meta table with additional entries in the kwargs-dict
         """
        tmp = self.con.execute("select * from meta")
        cols = [tmp.description[i][0] for i in range(len(tmp.description))]
        cols2update = ""
        data = []
        for key, value in kwargs.items():
            if key in cols:
                cols2update += f"{key}= ?, "
                data.append(value)
        cols2update = cols2update[:-2]
        self.con.execute(f"UPDATE meta SET {cols2update} WHERE scenario = ?", data +[self.scenario])
        self.con.commit()


    def log_global_time(self, timestep, time_dict):
        """
        Log global computation performance
        :param timestep: current timestep to log
        :param time_dict: dict with data to be logged
        :return:
        """

        data = [self.scenario, timestep,
                time_dict.pop("total_sim_step"),
                time_dict.pop("preprocessing"),
                time_dict.pop("time_sync") if "time_sync" in time_dict.keys() else None,
                time_dict.pop("time_visu") if "time_visu" in time_dict.keys() else None]
        self.con.execute("INSERT INTO global_performance_measure VALUES(?,?,?,?, ?,?)",data)
        self.con.commit()
        if len(time_dict) > 0:
            self.log_batch_time(timestep, time_dict)



    def log_batch_time(self,time_step, time_dict):
        """
        Log batch computation performance
        :param timestep: current timestep to log
        :param time_dict: dict with data to be logged
        :return:
        """
        data = []
        for batch_name, process_times in time_dict.items():
            data.append([self.scenario,batch_name,
                    time_step,
                    process_times["process_iteration_time"]  if "process_iteration_time" in process_times.keys() else None,
                    process_times["sim_step_time"],
                    process_times["agent_planning_time"],
                    process_times["sync_time_in"] if "sync_time_in" in process_times.keys() else None,
                    process_times["sync_time_out"] if "sync_time_out" in process_times.keys() else None,
                    ])
        self.con.executemany("INSERT INTO batch_performance_measure VALUES(?,?,?,?,?,?,?,?)", data)
        self.con.commit()


    def write_csv(self, table:str, file_name: str, value: str = "*"):
        """
        Writes database table to csv file
        :param table: table to export
        :param file_name: filename
        :param value: column to export ("*" := all columns)
        :return:
        """
        res = self.con.execute(f"SELECT {value} from {table}")
        rows = res.fetchall()

        # Get column names
        column_names = [description[0] for description in res.description]

        # Specify the CSV file path
        csv_file_path = os.path.join(self.config.simulation.mod_path, file_name + '.csv')

        # Write data to CSV file
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header
            csv_writer.writerow(column_names)

            # Write data
            csv_writer.writerows(rows)

    def close(self):
        """close sim_logger"""
        self.con.close()



#
#
def logger_initialization(config: Configuration, log_path, logger = "Simulation_logger") -> logging.Logger:
    """
    Message Logger Initialization
    """

    # msg logger
    msg_logger = logging.getLogger(logger) # logging.getLogger("Simulation_logger")

    if msg_logger.handlers:
        return msg_logger

    # Create directories
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # create file handler (outputs to file)
    path_log = os.path.join(log_path, "messages.log")
    file_handler = logging.FileHandler(path_log)

    # set logging levels
    loglevel = config.debug.msg_log_mode if hasattr(config, "debug") else config.simulation.msg_log_mode

    msg_logger.setLevel(loglevel)
    file_handler.setLevel(loglevel)

    # create log formatter
    # formatter = logging.Formatter('%(asctime)s\t%(filename)s\t\t%(funcName)s@%(lineno)d\t%(levelname)s\t%(message)s')
    log_formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)",
                                  "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(log_formatter)

    # create stream handler (prints to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(loglevel)

    # create stream formatter
    stream_formatter = logging.Formatter("%(levelname)-8s [%(filename)s]: %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # add handlers
    msg_logger.addHandler(file_handler)
    msg_logger.addHandler(stream_handler)
    msg_logger.propagate = False

    return msg_logger
