import os
import glob
from typing import Union
from omegaconf import OmegaConf, ListConfig, DictConfig
from cr_scenario_handler.utils.configuration import SimConfiguration, FrenetConfiguration


class ConfigurationBuilder:
    path_root: str = None
    path_config: str = None
    path_config_default: str = None

    @classmethod
    def build_configuration(cls, scenario_name: str, path_root: str = None,
                            dir_config="configurations", module="simulation"):
        """Builds configuration from default, scenario-specific, and commandline config files.

        Args:
            scenario_name (str): considered scenario
            path_root (str): root path of the package
            dir_config (str): folder of configs
            module (str): module name of config

        """
        if path_root is None:
            path_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))

        cls.set_paths(path_root=path_root, dir_config=dir_config, dir_config_default=module)

        config_default = cls.construct_configuration(scenario_name)
        config_cli = OmegaConf.from_cli()

        # configurations coming after overrides the ones coming before
        config_merged = OmegaConf.merge(config_default, config_cli)

        return config_merged

    @classmethod
    def build_sim_configuration(cls, scenario_name: str, scenario_folder: str, root_path: str,
                                module: str = "simulation") -> SimConfiguration:
        config_merged = cls.build_configuration(scenario_name, path_root=root_path, module=module)
        config_merged.simulation["scenario_path"] = os.path.join(scenario_folder, scenario_name + ".xml")
        config_merged.simulation["log_path"] = os.path.join(root_path, config_merged.simulation.path_output, scenario_name)
        config_merged.simulation["mod_path"] = root_path
        return SimConfiguration(config_merged)

    @classmethod
    def build_frenetplanner_configuration(cls, scenario_name: str = "Default", root_path: str = None,
                                          module: str="frenetix_motion_planner") -> FrenetConfiguration:
        config_merged = cls.build_configuration(scenario_name, path_root=root_path, module=module)
        return FrenetConfiguration(config_merged)

    @classmethod
    def set_paths(cls, path_root: str, dir_config: str, dir_config_default: str):
        """Sets necessary paths of the configuration builder.

        Args:
            path_root (str): root directory
            dir_config (str): directory storing configurations
            dir_config_default (str): directory storing default configurations
        """
        cls.path_root = path_root
        cls.path_config = os.path.join(path_root, dir_config)
        cls.path_config_default = os.path.join(cls.path_config, dir_config_default)

    @classmethod
    def construct_configuration(cls, name_scenario: str) -> Union[ListConfig, DictConfig]:
        """Constructs default configuration by accumulating yaml files.

        Collects all configuration files ending with '.yaml'.
        """

        config_default = OmegaConf.create()

        for path_file in glob.glob(cls.path_config_default + "/*.yaml"):
            with open(path_file, "r") as file_config:
                try:
                    config_partial = OmegaConf.load(file_config)
                    name_file = path_file.split("/")[-1].split(".")[0]

                except Exception as e:
                    print(e)

                else:
                    config_default[name_file] = config_partial

        if cls.path_config_default.split("/")[-1] == "simulation":
            config_default["simulation"]["name_scenario"] = name_scenario

        return config_default


