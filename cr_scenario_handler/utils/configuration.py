from typing import Union
from omegaconf import ListConfig, DictConfig
from commonroad.common.solution import VehicleType
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping


class Configuration:
    def __init__(self, config: Union[ListConfig, DictConfig]):
        pass


class SimConfiguration(Configuration):
    """
    Main Configuration class holding all planner-relevant configurations
    """
    def __init__(self, config: Union[ListConfig, DictConfig]):
        super().__init__(config)
        self.visualization = None
        self.simulation = None
        self.prediction = None
        self.occlusion = None
        self.behavior = None
        self.evaluation = None
        # initialize subclasses automatically
        for subclasses in config.keys():
            setattr(self, subclasses,  SubConfiguration(config[subclasses]))

        # initialize vehicle parameters
        self.vehicle: VehicleConfiguration = VehicleConfiguration(config.vehicle)


class FrenetConfiguration(Configuration):
    """
    Main Configuration class holding all planner-relevant configurations
    """
    def __init__(self, config: Union[ListConfig, DictConfig]):
        super().__init__(config)
        # self.multiagent = None
        self.planning = None
        self.debug = None
        self.cost = None

        # initialize subclasses automatically
        for subclasses in config.keys():
            setattr(self, subclasses,  SubConfiguration(config[subclasses]))


class SubConfiguration:
    """Class to store the sub-configs into class configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        for attributes in config.keys():
            setattr(self, attributes, config[attributes])


class VehicleConfiguration:
    """Class to store vehicle configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.cr_vehicle_id = config.cr_vehicle_id

        # get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
        vehicle_parameters = VehicleParameterMapping.from_vehicle_type(VehicleType(config.cr_vehicle_id))

        # get dimensions from given vehicle ID
        self.length = vehicle_parameters.l
        self.width = vehicle_parameters.w
        self.wb_front_axle = vehicle_parameters.a
        self.wb_rear_axle = vehicle_parameters.b
        self.wheelbase = vehicle_parameters.a + vehicle_parameters.b
        self.mass = vehicle_parameters.m

        # get constraints from given vehicle ID
        self.a_max = vehicle_parameters.longitudinal.a_max
        self.v_max = vehicle_parameters.longitudinal.v_max
        self.v_switch = vehicle_parameters.longitudinal.v_switch
        self.delta_min = vehicle_parameters.steering.min
        self.delta_max = vehicle_parameters.steering.max
        self.v_delta_min = vehicle_parameters.steering.v_min
        self.v_delta_max = vehicle_parameters.steering.v_max

        # overwrite parameters given by vehicle ID if they are explicitly provided in the *.yaml file
        for key, value in config.items():
            if value is not None:
                setattr(self, key, value)

# EOF
