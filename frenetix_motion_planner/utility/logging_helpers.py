import os
import sys
import numpy as np
import json
from pathlib import Path
import logging
import tempfile
import sqlite3
import math

from omegaconf import DictConfig, ListConfig
from cr_scenario_handler.utils.configuration import Configuration

from commonroad.common.util import FileFormat
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile

from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle


class SqlLogger:
    con: sqlite3.Connection
    __db_path: Path
    cost_names: list[str]
    inf_names: list[str]

    @staticmethod
    def _convert_config(config: Configuration) -> dict:
        data = dict()
        for item in config.__dict__:
            # print(item)
            ii = getattr(config, item)
            data[item] = dict()
            for it in ii.__dict__:
                val = ii.__dict__[it]
                if isinstance(val, DictConfig):
                    val = dict(val)
                if isinstance(val, ListConfig):
                    val = list(val)

                # print(f"name={it} type={type(val)}")

                data[item][it] = val

        return data

    def __init__(self, path_logs: Path, config: Configuration, scenario: Scenario, planning_problem: PlanningProblem) -> None:
        self.path_logs = path_logs

        path_logs.mkdir(exist_ok=True)

        self.__db_path = path_logs / "trajectories.db"
        self.__db_path.unlink(missing_ok=True)

        self.con = sqlite3.connect(self.__db_path, isolation_level="EXCLUSIVE")

        self.con.executescript("""
            PRAGMA journal_mode = OFF;
            PRAGMA locking_mode = EXCLUSIVE;
            PRAGMA temp_store = MEMORY;
        """)

        self.con.execute("""
            CREATE TABLE trajectories(
                time_step INT NOT NULL,
                id INT NOT NULL,
                x TEXT NOT NULL,
                y TEXT NOT NULL,
                theta TEXT NOT NULL,
                kappa TEXT NOT NULL,
                curvilinear_theta TEXT NOT NULL,
                v TEXT NOT NULL,
                a TEXT NOT NULL,
                PRIMARY KEY(time_step, id)
            ) STRICT
        """)

        self.con.execute("""
            CREATE TABLE trajectories_meta(
                time_step INT NOT NULL,
                id INT NOT NULL,
                dt REAL NOT NULL,
                s_position REAL NOT NULL,
                d_position REAL NOT NULL,
                ego_risk REAL,
                obst_risk REAL,
                PRIMARY KEY(time_step, id)
            ) STRICT
        """)

        self.con.execute("""
            CREATE TABLE sampling_params(
                time_step INT NOT NULL,
                id INT NOT NULL,
                t0 REAL NOT NULL,
                t1 REAL NOT NULL,
                s0 REAL NOT NULL,
                ss0 REAL NOT NULL,
                sss0 REAL NOT NULL,
                ss1 REAL NOT NULL,
                sss1 REAL NOT NULL,
                d0 REAL NOT NULL,
                dd0 REAL NOT NULL,
                ddd0 REAL NOT NULL,
                d1 REAL NOT NULL,
                dd1 REAL NOT NULL,
                ddd1 REAL NOT NULL,
                PRIMARY KEY(time_step, id)
            ) STRICT
        """)

        self.con.execute("""
            CREATE TABLE meta(
                key TEXT PRIMARY KEY,
                value ANY
            ) STRICT
        """)

        with tempfile.NamedTemporaryFile(prefix="reactive-planner-scenario-pb_") as tmp:
            planning_problem_set = PlanningProblemSet([planning_problem])
            writer = CommonRoadFileWriter(scenario, planning_problem_set, file_format=FileFormat.PROTOBUF)
            writer.write_to_file(tmp.name, overwrite_existing_file=OverwriteExistingFile.ALWAYS)

            scenario_data = tmp.read()
            self.con.execute("INSERT INTO meta VALUES(?, ?)", ("scenario", scenario_data))

            self.con.commit()

        converted_config = SqlLogger._convert_config(config)

        json_config = json.dumps(converted_config, skipkeys=True)
        self.con.execute("INSERT INTO meta VALUES(?, json(?))", ("config", json_config))

        self.con.commit()

        self.set_inf_names([
            "Yaw_rate",
            "Acceleration",
            "Curvature"
        ])

    def write_reference_path(self, reference_path) -> None:
        rp = dict()
        rp["x"] = reference_path[:,0].tolist()
        rp["y"] = reference_path[:,1].tolist()
        json_reference_path = json.dumps(rp)
        self.con.execute("INSERT INTO meta VALUES(?, json(?))", ("reference_path", json_reference_path))

    def set_inf_names(self, inf_names_list: list[str]) -> None:
        self.inf_names = inf_names_list

        inf_columns = ""
        for inf_name in self.inf_names:
            inf_columns += f"{inf_name} INT NOT NULL, \n"

        self.con.execute(f"""
            CREATE TABLE infeasability(
                time_step INT NOT NULL,
                id INT NOT NULL,
                feasible INT NOT NULL,
                {inf_columns}
                PRIMARY KEY(time_step, id)
            ) STRICT
        """)

    def set_cost_names(self, cost_names_list: list[str]) -> None:
        self.cost_names = cost_names_list

        cost_columns = ""
        for cost_name in self.cost_names:
            cost_columns += f"{cost_name} REAL NOT NULL, \n"

        self.con.execute(f"""
            CREATE TABLE costs(
                time_step INT NOT NULL,
                id INT NOT NULL,
                costs_cumulative_weighted REAL NOT NULL,
                {cost_columns}
                PRIMARY KEY(time_step, id)
            ) STRICT
        """)

    @staticmethod
    def _trajectories_row(time_step: int, trajectory) -> tuple[int, str, str, str, str, str, str, str, str]:
        def float_values(values):
            value_list = ','.join(map(lambda x: "{:.5g}".format(x), values))
            return "[" + value_list + "]"

        return (
            time_step,
            str(trajectory.uniqueId),
            float_values(trajectory.cartesian.x),
            float_values(trajectory.cartesian.y),
            float_values(trajectory.cartesian.theta),
            float_values(trajectory.cartesian.kappa),
            float_values(trajectory.curvilinear.theta),
            float_values(trajectory.cartesian.v),
            float_values(trajectory.cartesian.a)
            )

    @staticmethod
    def _trajectories_meta_row(time_step: int, trajectory):
        return (
            time_step,
            trajectory.uniqueId,
            trajectory.dt,
            trajectory.curvilinear.s[0],
            trajectory.curvilinear.d[0],
            trajectory._ego_risk,
            trajectory._obst_risk,
            )

    @staticmethod
    def _sampling_params_row(time_step: int, trajectory):
        return [time_step, trajectory.uniqueId] + list(trajectory.sampling_parameters)

    def _costs_row(self, time_step: int, trajectory):
        cost_row = [time_step, trajectory.uniqueId, trajectory.cost]

        for cost_name in self.cost_names:
            try:
                cost = trajectory.costMap[cost_name][1]
            except KeyError:
                cost = 0.0

            cost_row.append(cost)

        return cost_row

    def _infeasability_row(self, time_step: int, trajectory):
        inf_row = [time_step, trajectory.uniqueId, trajectory.feasible]

        for inf_name in self.inf_names:
            key_name = inf_name.replace("_", " ") + " Constraint"
            inf = trajectory.feasabilityMap[key_name]
            inf_row.append(inf)

        return inf_row

    def log_all_trajectories(self, all_trajectories, time_step: int):
        trajectory_data = []
        meta_data = []
        sampling_data = []
        cost_data = []
        inf_data = []
        for trajectory in all_trajectories:
            trajectory_data.append(SqlLogger._trajectories_row(time_step, trajectory))
            meta_data.append(SqlLogger._trajectories_meta_row(time_step, trajectory))
            sampling_data.append(SqlLogger._sampling_params_row(time_step, trajectory))
            cost_data.append(self._costs_row(time_step, trajectory))
            inf_data.append(self._infeasability_row(time_step, trajectory))

        self.con.executemany("INSERT INTO trajectories VALUES(?, ?, json(?), json(?), json(?), json(?), json(?), json(?), json(?))", trajectory_data)
        self.con.executemany(f"INSERT INTO trajectories_meta VALUES(?, ?, {','.join(5 * '?')})", meta_data)
        self.con.executemany(f"INSERT INTO sampling_params VALUES(?, ?, {','.join(13 * '?')})", sampling_data)
        self.con.executemany(f"INSERT INTO costs VALUES(?, ?, ?, {','.join(len(self.cost_names) * '?')})", cost_data)
        self.con.executemany(f"INSERT INTO infeasability VALUES(?, ?, ?, {','.join(len(self.inf_names) * '?')})", inf_data)

        self.con.commit()


class DataLoggingCosts:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, path_logs: str, config, scenario: Scenario, planning_problem: PlanningProblem,
                 header_only: bool = False, save_all_traj: bool = False, cost_params: dict = None) -> None:
        """"""

        self.save_all_traj = save_all_traj

        self.header = None
        self.trajectories_header = None
        self.prediction_header = None
        self.collision_header = None
        self.save_unweighted_costs = config.debug.save_unweighted_costs

        self.path_logs = path_logs
        self._cost_list_length = None
        self.cost_names_list = None

        log_file_name = "logs.csv"
        prediction_file_name = "predictions.csv"
        collision_file_name = "collision.csv"
        self.trajectories_file_name = "trajectories.csv"

        if header_only:
            return
        self.trajectory_number = 0

        self.__trajectories_log_path = None

        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)
        self.__log_path = os.path.join(path_logs, log_file_name)
        self.__prediction_log_path = os.path.join(
            path_logs, prediction_file_name)
        self.__collision_log_path = os.path.join(
            path_logs, collision_file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(
            parents=True, exist_ok=True)

        self.sql_logger = SqlLogger(Path(path_logs), config, scenario, planning_problem)

        self.set_logging_header(cost_params, config)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def set_logging_header(self, cost_function_names=None, config=None):

        cost_names = str()
        if cost_function_names:
            self.cost_names_list = list(set(list(cost_function_names.keys())) | set(list(config.cost.external_cost_weights.keys())))
            self.cost_names_list.sort()
            self._cost_list_length = len(self.cost_names_list)
            for names in self.cost_names_list:
                cost_names += names.replace(" ", "_") + '_cost;'

        self.sql_logger.set_cost_names(self.cost_names_list)

        self.header = (
            "trajectory_number;"
            "calculation_time_s;"
            "x_position_vehicle_m;"
            "y_position_vehicle_m;"
            "optimal_trajectory;"
            "percentage_feasible_traj;"
            "infeasible_sum;"
            "inf_kin_acceleration;"
            "inf_kin_negative_s_velocity;"
            "inf_kin_max_s_idx;"
            "inf_kin_negative_v_velocity;"
            "inf_kin_max_curvature;"
            "inf_kin_yaw_rate;"
            "inf_kin_max_curvature_rate;"
            "inf_kin_vehicle_acc;"
            "inf_cartesian_transform;"
            "inf_precision_error;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "kappa_rad;"
            "curvilinear_orientations_rad;"
            "velocities_mps;"
            "desired_velocity_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "ego_risk;"
            "obst_risk;"
            "accepted_occ_harm;"
            "costs_cumulative_weighted;"
            +
            cost_names.strip(";")
        )
        self.trajectories_header = (
            "time_step;"
            "trajectory_number;"
            "unique_id;"
            "feasible;"
            "horizon;"
            "dt;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "kappa_rad;"
            "curvilinear_orientations_rad;"
            "velocities_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "ego_risk;"
            "obst_risk;"
            "costs_cumulative_weighted;"
            +
            cost_names
            +
            "inf_kin_yaw_rate;"
            "inf_kin_acceleration;"
            "inf_kin_max_curvature;"
            "inf_kin_max_curvature_rate;"
        ).strip(";")



        self.prediction_header = (
            "trajectory_number;"
            "prediction"
        )

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

        with open(self.__prediction_log_path, "w+") as fh:
            fh.write(self.prediction_header)

        if self.save_all_traj:
            self.__trajectories_log_path = os.path.join(
                self.path_logs, self.trajectories_file_name)
            with open(self.__trajectories_log_path, "w+") as fh:
                fh.write(self.trajectories_header)

    def get_headers(self):
        return self.header

    def log(self, trajectory, time_step: int, infeasible_kinematics, percentage_kinematics,
            planning_time: float, ego_vehicle: DynamicObstacle, collision: bool = False, desired_velocity: float = None):

        new_line = "\n" + str(time_step)

        if trajectory is not None:

            cartesian = trajectory.cartesian
            cost_list_names = list(trajectory.costMap.keys())

            # log time
            new_line += ";" + json.dumps(str(planning_time), default=default)

            # Vehicle Occupancy Position
            new_line += ";" + json.dumps(str(ego_vehicle.initial_state.position[0]), default=default)
            new_line += ";" + json.dumps(str(ego_vehicle.initial_state.position[1]), default=default)

            # optimal trajectory available
            new_line += ";True"
            # log infeasible
            if percentage_kinematics is not None:
                new_line += ";" + json.dumps(str(percentage_kinematics), default=default)
            else:
                new_line += ";"

            for kin in infeasible_kinematics:
                new_line += ";" + json.dumps(str(kin), default=default)

            # log position
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.x))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.y))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.theta))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.kappa))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, trajectory.curvilinear.theta))), default=default)
            # log velocity & acceleration
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.v))), default=default)
            new_line += ";" + json.dumps(str(desired_velocity), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.a))), default=default)

            # # log frenet coordinates (distance to reference path)
            new_line += ";" + \
                json.dumps(str(trajectory.curvilinear.s[0]), default=default)
            new_line += ";" + \
                json.dumps(str(trajectory.curvilinear.d[0]), default=default)

            # log risk values number
            if trajectory._ego_risk is not None and trajectory._obst_risk is not None:
                new_line += ";" + json.dumps(str(trajectory._ego_risk), default=default)
                new_line += ";" + json.dumps(str(trajectory._obst_risk), default=default)
            else:
                new_line += ";;"

            # log occ module harm
            if hasattr(trajectory, "harm_occ_module") and trajectory.harm_occ_module is not None:
                new_line += ";" + json.dumps(str(trajectory.harm_occ_module), default=default)
            else:
                new_line += ";"

            new_line = self.log_costs_of_single_trajectory(trajectory, new_line, cost_list_names)

        else:
            # log time
            new_line += ";" + json.dumps(str(planning_time), default=default)
            new_line += ";False"
            # log infeasible
            for kin in infeasible_kinematics:
                new_line += ";" + json.dumps(str(kin), default=default)

            # log position
            new_line += ";None"
            new_line += ";None"
            new_line += ";None"
            # log velocity & acceleration
            new_line += ";None"
            new_line += ";None"

            # # log frenet coordinates (distance to reference path)
            new_line += ";None"
            new_line += ";None"

            # log costs
            new_line += ";None"
            # log costs
            for i in range(0, self._cost_list_length):
                new_line += ";None"

        with open(self.__log_path, "a") as fh:
            fh.write(new_line)

    def log_predicition(self, prediction):
        new_line = "\n" + str(self.trajectory_number)

        new_line += ";" + json.dumps(prediction, default=default)

        with open(self.__prediction_log_path, "a") as fh:
            fh.write(new_line)

    def log_collision(self, collision_with_obj, ego_length, ego_width, progress, center=None, last_center=None, r_x=None, r_y=None, orientation=None):
        self.collision_header = (
            "ego_length;"
            "ego_width;"
            "progress;"
            "center_x;"
            "center_y;"
            "last_center_x;"
            "last_center_y;"
            "r_x;"
            "r_y;"
            "orientation"
        )

        with open(self.__collision_log_path, "w+") as fh:
            fh.write(self.collision_header)

        new_line = "\n" + str(ego_length)
        new_line += ";" + str(ego_width)
        new_line += ";" + str(progress)
        if collision_with_obj:
            new_line += ";" + str(center[0])
            new_line += ";" + str(center[1])
            new_line += ";" + str(last_center[0])
            new_line += ";" + str(last_center[1])
            new_line += ";" + str(r_x)
            new_line += ";" + str(r_y)
            new_line += ";" + str(orientation)
        else:
            new_line += ";None;None;None;None;None;None;None"

        with open(self.__collision_log_path, "a") as fh:
            fh.write(new_line)

    def log_all_trajectories(self, all_trajectories, time_step: int):
        i = 0

        for trajectory in all_trajectories:
            self.log_trajectory(trajectory, i, time_step, trajectory.feasible)
            i += 1

        self.sql_logger.log_all_trajectories(all_trajectories, time_step)

    def log_trajectory(self, trajectory, trajectory_number: int, time_step, feasible: bool):
        new_line = "\n" + str(time_step)
        new_line += ";" + str(trajectory_number)
        new_line += ";" + str(trajectory.uniqueId)
        new_line += ";" + str(feasible)

        if hasattr(trajectory, 'horizon'):
            new_line += ";" + str(round(trajectory.horizon, 3))
        else:
            new_line += ";" + str(round(trajectory.sampling_parameters[1], 3))

        new_line += ";" + str(trajectory.dt)
        cartesian = trajectory.cartesian
        cost_list_names = list(trajectory.costMap.keys())

        def format_float(x) -> str:
            assert math.isfinite(x)
            return "{:.5e}".format(x)

        def float_values(values):
            value_list = ','.join(map(format_float, values))
            return json.dumps(str(value_list), default=default)

        # log position
        new_line += ";" + float_values(cartesian.x)
        new_line += ";" + float_values(cartesian.y)
        new_line += ";" + float_values(cartesian.theta)
        new_line += ";" + float_values(cartesian.kappa)
        new_line += ";" + float_values(trajectory.curvilinear.theta)
        # log velocity & acceleration
        new_line += ";" + float_values(cartesian.v)
        new_line += ";" + float_values(cartesian.a)

        # log frenet coordinates (distance to reference path)
        new_line += ";" + \
            json.dumps(str(trajectory.curvilinear.s[0]), default=default)
        new_line += ";" + \
            json.dumps(str(trajectory.curvilinear.d[0]), default=default)

        # log risk values number
        if trajectory._ego_risk is not None and trajectory._obst_risk is not None:
            new_line += ";" + json.dumps(str(trajectory._ego_risk), default=default)
            new_line += ";" + json.dumps(str(trajectory._obst_risk), default=default)
        else:
            new_line += ";;"

        new_line = self.log_costs_of_single_trajectory(trajectory, new_line, cost_list_names)
            
        new_line += ";" + str(trajectory.feasabilityMap["Yaw rate Constraint"])
        new_line += ";" + str(trajectory.feasabilityMap["Acceleration Constraint"])
        new_line += ";" + str(trajectory.feasabilityMap["Curvature Constraint"])
        new_line += ";" + str(trajectory.feasabilityMap["Curvature Rate Constraint"])

        #for k, v in trajectory.feasabilityMap.items():
        #    new_line += ";" + \
        #        json.dumps(str(v), default=default)

        with open(self.__trajectories_log_path, "a") as fh:
            fh.write(new_line)

    def log_costs_of_single_trajectory(self, trajectory, new_line, cost_list_names):

        # log costs
        new_line += ";" + json.dumps(str(trajectory.cost), default=default)

        # log costs
        for cost_template in self.cost_names_list:
            if cost_template in cost_list_names:
                if not self.save_unweighted_costs:
                    new_line += ";" + json.dumps(str(trajectory.costMap[cost_template][1]), default=default)
                else:
                    new_line += ";" + json.dumps(str(trajectory.costMap[cost_template][0]), default=default)
            else:
                new_line += ";" + json.dumps(str(0), default=default)

        return new_line

def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")


# def messages_logger_initialization(config: Configuration, log_path, logger=logging.getLogger("Message_logger")) -> logging.Logger:
#     """
#     Message Logger Initialization
#     """
#
#     # msg logger
#     msg_logger = logger
#
#     if msg_logger.handlers:
#         return msg_logger
#
#     # Create directories
#     if not os.path.exists(log_path):
#         os.makedirs(log_path)
#
#     # create file handler (outputs to file)
#     path_log = os.path.join(log_path, "messages.log")
#     file_handler = logging.FileHandler(path_log)
#
#     # set logging levels
#     loglevel = config.debug.msg_log_mode
#     msg_logger.setLevel(loglevel)
#     file_handler.setLevel(loglevel)
#
#     # create log formatter
#     # formatter = logging.Formatter('%(asctime)s\t%(filename)s\t\t%(funcName)s@%(lineno)d\t%(levelname)s\t%(message)s')
#     log_formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)",
#                                   "%Y-%m-%d %H:%M:%S")
#     file_handler.setFormatter(log_formatter)
#
#     # create stream handler (prints to stdout)
#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setLevel(loglevel)
#
#     # create stream formatter
#     stream_formatter = logging.Formatter("%(levelname)-8s [%(filename)s]: %(message)s")
#     stream_handler.setFormatter(stream_formatter)
#
#     # add handlers
#     msg_logger.addHandler(file_handler)
#     msg_logger.addHandler(stream_handler)
#     msg_logger.propagate = False
#
#     return msg_logger
