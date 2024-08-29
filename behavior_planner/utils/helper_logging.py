__author__ = "Moritz Ellermann, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import logging
import os
import sys

from rich.logging import RichHandler

class BehaviorLogger(object):

    def __init__(self, behavior_config, logger="Behavior_logger", column_headers: [] = None):
        self._behavior_config = behavior_config

        if column_headers is None:
            if behavior_config.minimal_logging:
                self._column_headers = [
                    "time_step",
                    "ref_position_s",
                    "street_setting",
                    "behavior_state_static",
                    "situation_state_static",
                    "behavior_state_dynamic",
                    "situation_state_dynamic",
                    "velocity",
                    "goal_velocity",
                    "desired_velocity",
                    "situation_time_step_counter",
                    "lane_change_target_lanelet_id",
                    "nav_lane_changes_left",
                    "nav_lane_changes_right",
                    "overtaking",
                    "velocity_mode",
                    "TTC",
                    "MAX",
                    "change_velocity_for_lane_change",
                    "slowing_car_for_traffic_light",
                    "waiting_for_green_light"
                ]
            else:
                self._column_headers = [
                    "time_step",
                    "ref_position_s",
                    "street_setting",
                    "behavior_state_static",
                    "situation_state_static",
                    "behavior_state_dynamic",
                    "situation_state_dynamic",
                    "situation_time_step_counter",
                    "stop_point_s",
                    "stop_point_dist",
                    "desired_velocity_stop_point",
                    "stop_point_mode",
                    "velocity",
                    "goal_velocity",
                    "desired_velocity",
                    "velocity_mode",
                    "TTC",
                    "MAX",
                    "speed_limit_default",
                    "speed_limit",
                    "change_velocity_for_lane_change",
                    "traffic_light_state",
                    "slowing_car_for_traffic_light",
                    "waiting_for_green_light",
                    "closest_preceding_vehicle",
                    "dist_preceding_veh",
                    "vel_preceding_veh",
                    "ttc_conditioned",
                    "ttc_relative",
                    "min_safety_dist",
                    "safety_dist",
                    "stop_distance",
                    "comfortable_stopping_distance",
                    "dist_to_tl",
                    "condition_factor",
                    "lon_dyn_cond_factor",
                    "lat_dyn_cond_factor",
                    "visual_cond_factor",
                    "lane_change_target_lanelet_id",
                    "nav_lane_changes_left",
                    "nav_lane_changes_right",
                    "overtake_lange_changes_offset",
                    "overtaking",
                    "do_lane_change",
                    "undo_lane_change",
                    "initiated_lane_change",
                    "undid_lane_change",
                    "detected_lanelets",
                    "obstacles_on_target_lanelet",
                    "free_space_offset",
                    "free_space_on_target_lanelet",
                    "lane_change_left_ok",
                    "lane_change_right_ok",
                    "lane_change_left_done",
                    "lane_change_right_done",
                    "lane_change_prep_right_abort",
                    "lane_change_prep_left_abort",
                    "lane_change_right_abort",
                    "lane_change_left_abort",
                    "no_auto_lane_change",
                    "turn_clear",
                    "crosswalk_clear",
                    "stop_yield_sign_clear",
                    "reference_path"
                ]
        else:
            self._column_headers = column_headers

        # Define the path for the log file
        self._message_log_file_path = os.path.join(self._behavior_config.behavior_log_path_scenario, "messages.log")
        self._data_log_file_path = os.path.join(self._behavior_config.behavior_log_path_scenario, "data.csv")

        if behavior_config.archive_previous_logs:
            with (open(self._data_log_file_path, "r") as file):
                if behavior_config.minimal_logging:
                    # header length of minimal logging >= header length of previous log
                    if len(self._column_headers) >= len(file.read().split("\n", 1)[0].split(";")):
                        self._data_archive_log_file_path = os.path.join(
                            self._behavior_config.behavior_log_path_scenario, "data_previous_minimal.csv")
                    else:
                        self._data_archive_log_file_path = os.path.join(
                            self._behavior_config.behavior_log_path_scenario, "data_previous.csv")
                else:
                    # header length of normal logging > header length of previous log
                    if len(self._column_headers) > len(file.read().split("\n", 1)[0].split(";")):
                        self._data_archive_log_file_path = os.path.join(
                            self._behavior_config.behavior_log_path_scenario, "data_previous_minimal.csv")
                    else:
                        self._data_archive_log_file_path = os.path.join(
                            self._behavior_config.behavior_log_path_scenario, "data_previous.csv")
        else:
            self._data_archive_log_file_path = ""
            self._data_archive_log_file_path = ""

        self._logger = logger
        self.message_logger = None

        self._init_message_logger()
        self._init_data_logger()

    def _init_message_logger(self):
        """
        Message Logger for Behavior Planner Initialization

        Format for FileHandler:
            "%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S"

        Format for StreamHandler:
            "%(levelname)-8s [%(filename)s]: %(message)s"
        """

        # bhv logger
        self.message_logger = logging.getLogger(self._logger)

        # create file handler (outputs to file)
        file_handler = logging.FileHandler(self._message_log_file_path)
        file_formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)",
                                           "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

        # create stream handler (prints to stdout)
        # (replaced by RichHandler)
        # stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler = RichHandler()
        # stream_formatter = logging.Formatter("%(levelname)-8s [%(filename)s]: %(message)s")
        # stream_handler.setFormatter(stream_formatter)

        # set logging levels
        self.message_logger.setLevel(
            min(self._behavior_config.behavior_log_mode_file, self._behavior_config.behavior_log_mode_stdout))
        file_handler.setLevel(self._behavior_config.behavior_log_mode_file)  # file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(self._behavior_config.behavior_log_mode_stdout)  # file_handler.setLevel(logging.INFO)

        # add handlers
        self.message_logger.addHandler(file_handler)
        self.message_logger.addHandler(stream_handler)
        self.message_logger.propagate = False

    def _init_data_logger(self):
        """
        Initialize Data Logger for Behavior Planner.

        This function archives the existing data.log if specified in the behavior config
        and then prepares the new CSV file.
        """
        if not isinstance(self._column_headers, list):
            raise TypeError(f"expected an argument of type list, but got {type(self._column_headers).__name__}")

        # Define headers of columns by joining keys of the column_headers dictionary
        header = ";".join(str(col_head) for col_head in self._column_headers) + "\n"

        try:
            # create log archive file and set the headers
            if ((not os.path.exists(self._data_archive_log_file_path)) or
                    (os.path.getsize(self._data_archive_log_file_path) == 0)):
                with (open(self._data_log_file_path, "r") as file,
                      open(self._data_archive_log_file_path, 'w') as archive):
                    archive.write(file.read().split("\n", 1)[0] + "\n")

            # archiving old logs
            if self._behavior_config.archive_previous_logs:
                with (open(self._data_log_file_path, "r") as file,
                      open(self._data_archive_log_file_path, 'a') as archive):
                    archive.write(file.read().split("\n", 1)[1])  # remove header
        except FileNotFoundError:
            if self._behavior_config.archive_previous_logs:
                self.message_logger.warning("FileNotFoundError: no data log found")
            else:
                self.message_logger.warning("saving previous log disabled")

        # Clear file and write the header to initialize the log file
        with open(self._data_log_file_path, 'w') as file:
            file.write(header)

    def log_data(self, data):
        """
        Log Data to a CSV file.

        This function appends a row of data to a CSV file specified in the behavior configuration.

        Args:
            data (dict): Dictionary containing the data (values) to be logged.
        """
        # create a dictionary, where
        log_data_dict = {}
        for key in data:
            if str(key).__contains__("state"):
                try:
                    for inner_date in data[key].__dict__:
                        if inner_date in self._column_headers:
                            log_data_dict.update({inner_date: data[key].__dict__[inner_date]})
                except AttributeError:
                    if key in self._column_headers:
                        log_data_dict.update({key: data[key]})
            else:
                if key in self._column_headers:
                    log_data_dict.update({key: data[key]})

        # sort the dictionary according to the order of the column headers
        log_data_dict = {key: log_data_dict[key] for key in self._column_headers}

        # Create a row by joining the values from the 'data' dictionary
        row = ";".join(str(col_value) for col_value in log_data_dict.values()).replace("\n", ",") + "\n"

        # Open the file in write mode and write the row to the log file
        with open(self._data_log_file_path, 'a') as file:
            file.write(row)
