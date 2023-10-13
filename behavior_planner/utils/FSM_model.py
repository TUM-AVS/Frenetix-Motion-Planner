__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import behavior_planner.utils.FSM_logic_modules as Logic
from commonroad.geometry.shape import Rectangle
import behavior_planner.utils.helper_functions as hf
import time
import logging

# get logger
msg_logger = logging.getLogger("Message_logger")

'''base type 'State' used to create states'''
State = type("State", (object,), {})


class EgoFSM(object):
    """
    Finite State Machine class: Used by the Behavior Planner mirror the state of the real ego vehicle. To reduce
    complexity the FSM is built in recursive fashion with 3 layers. First layer is the Street Setting States, Second
    Layer is the Behavior States and Third Layer is the Situation States.

    Possible States are:

    Street Setting: 'Highway', 'Country', 'Urban'

    Behavior: 'PrepareLaneChangeLeft', 'LaneChangeLeft', 'PrepareLaneChangeRight', 'LaneChangeRight', 'LaneMerge',
                'PrepareLaneMerge', 'PrepareRoadExit', 'RoadExit', 'PrepareOvertake', 'Overtake', 'PrepareTurnLeft'
                'TurnLeft', 'PrepareTurnRight', 'TurnRight', 'PrepareTrafficLight', 'TrafficLight', 'PrepareYieldSign',
                'YieldSign', 'PrepareStopSign', 'StopSign'

    Situation: 'IdentifyTargetLaneAndVehiclesOnTargetLane', 'IdentifyFreeSpaceOnTargetLaneForLaneChange', 'PreparationsDon',
                'InitiateLaneChange', 'EgoVehicleBetweenTwoLanes', 'BehaviorStateComplete',
                'EstimateMergingLaneLengthAndEmergencyStopPoint', 'InitiateLaneMerge, 'InitiateRoadExit'

    TODO: Complete set of possible states
    TODO: Create logic modules for each Simple FSM

    """
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state  # FSM output for behavior module
        self.street_setting = BM_state.street_setting  # street setting of CR scenario
        self.FSM_street_setting = SimpleFSM(
            possible_states={
                'Highway': Highway(BM_state=BM_state),
                'Country': Country(BM_state=BM_state),
                'Urban': Urban(BM_state=BM_state)},
            possible_transitions={
                'toHighway': Transition('Highway'),
                'toCountry': Transition('Country'),
                'toUrban': Transition('Urban')},
            default_state=BM_state.street_setting)
        self.logic = Logic.LogicStreetSetting(BM_state=BM_state)

    def execute(self):
        # FSM
        transition, self.street_setting = self.logic.execute(cur_state=self.street_setting)
        if transition is not None:
            self.FSM_street_setting.transition(transition)
            self.FSM_street_setting.reset_current_state()
        self.FSM_state.street_setting = self.street_setting

        # actions
        self.FSM_street_setting.execute()


'''FSM Base Class'''


class SimpleFSM(object):
    """Finite State Machine (FSM) base class"""

    def __init__(self, possible_states, possible_transitions, default_state):
        self.states = possible_states
        self.transitions = possible_transitions
        self.cur_state = self.states[default_state]
        self.trans = None

    def set_state(self, state_name):
        self.cur_state = self.states[state_name]

    def transition(self, trans_name):
        self.trans = self.transitions[trans_name]

    def execute(self):
        if self.trans:
            self.trans.execute()
            self.set_state(self.trans.to_state)
            self.trans = None
        self.cur_state.execute()

    def reset_current_state(self):
        self.cur_state.reset_state()


'''Street Setting States: '''


class Country(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_country_static = SimpleFSM(
            possible_states={
                'StaticDefault': StaticDefault(),
                'PrepareIntersection': PrepareIntersection(),
                'PrepareTurnLeft': PrepareTurnLeft(),
                'PrepareTurnRight': PrepareTurnRight(),
                'PrepareTrafficLight': PrepareTrafficLight(BM_state=BM_state),
                'Intersection': Intersection(),
                'TurnLeft': TurnLeft(),
                'TurnRight': TurnRight(),
                'TrafficLight': TrafficLight(BM_state=BM_state)},
            possible_transitions={
                'toStaticDefault': Transition('StaticDefault'),
                'toPrepareIntersection': Transition('PrepareIntersection'),
                'toPrepareTurnLeft': Transition('PrepareTurnLeft'),
                'toPrepareTurnRight': Transition('PrepareTurnRight'),
                'toPrepareTrafficLight': Transition('PrepareTrafficLight'),
                'toIntersection': Transition('Intersection'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toTrafficLight': Transition('TrafficLight')},
            default_state='StaticDefault')
        self.FSM_country_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareOvertake': PrepareOvertake(),
                'Overtake': Overtake()},
            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareOvertake': Transition('PrepareOvertake'),
                'toOvertake': Transition('Overtake')},
            default_state='DynamicDefault')
        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicCountryDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        msg_logger.debug("FSM Street Setting: Country")
        # FSM static
        transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
        if transition_static is not None:
            self.FSM_country_static.transition(transition_static)
            self.FSM_country_static.reset_current_state()
        self.FSM_state.behavior_state_static = self.cur_state_static
        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_country_dynamic.transition(transition_dynamic)
            self.FSM_country_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = True
        else:
            self.FSM_state.no_auto_lane_change = False
        self.FSM_country_static.execute()
        self.FSM_country_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_country_static.cur_state = self.FSM_country_static.states['StaticDefault']
        self.FSM_state.behavior_state_static = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_country_dynamic.cur_state = self.FSM_country_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class Urban(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_urban_static = SimpleFSM(
            possible_states={
                'StaticDefault': StaticDefault(),
                'PrepareIntersection': PrepareIntersection(),
                'PrepareTurnLeft': PrepareTurnLeft(),
                'PrepareTurnRight': PrepareTurnRight(),
                'PrepareTrafficLight': PrepareTrafficLight(BM_state=BM_state),
                'PrepareCrosswalk': PrepareCrosswalk(),
                'PrepareStopSign': PrepareStopSign(),
                'PrepareYieldSign': PrepareYieldSign(),
                'Intersection': Intersection(),
                'TurnLeft': TurnLeft(),
                'TurnRight': TurnRight(),
                'TrafficLight': TrafficLight(BM_state=BM_state),
                'Crosswalk': Crosswalk(),
                'StopSign': StopSign(),
                'YieldSign': YieldSign()},

            possible_transitions={
                'toStaticDefault': Transition('StaticDefault'),
                'toPrepareIntersection': Transition('PrepareIntersection'),
                'toPrepareTurnLeft': Transition('PrepareTurnLeft'),
                'toPrepareTurnRight': Transition('PrepareTurnRight'),
                'toPrepareTrafficLight': Transition('PrepareTrafficLight'),
                'toPrepareCrosswalk': Transition('PrepareCrosswalk'),
                'toPrepareStopSign': Transition('PrepareStopSign'),
                'toPrepareYieldSign': Transition('PrepareYieldSign'),
                'toIntersection': Transition('Intersection'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toTrafficLight': Transition('TrafficLight'),
                'toCrosswalk': Transition('Crosswalk'),
                'toStopSign': Transition('StopSign'),
                'toYieldSign': Transition('YieldSign')},
            default_state='StaticDefault')
        self.FSM_urban_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareLaneChangeLeft': PrepareLaneChangeLeft(BM_state=BM_state),
                'PrepareLaneChangeRight': PrepareLaneChangeRight(BM_state=BM_state),
                'PrepareOvertake': PrepareOvertake(),
                'LaneChangeLeft': LaneChangeLeft(BM_state=BM_state),
                'LaneChangeRight': LaneChangeRight(BM_state=BM_state),
                'Overtake': Overtake()},

            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareLaneChangeLeft': Transition('PrepareLaneChangeLeft'),
                'toPrepareLaneChangeRight': Transition('PrepareLaneChangeRight'),
                'toPrepareOvertake': Transition('PrepareOvertake'),
                'toLaneChangeLeft': Transition('LaneChangeLeft'),
                'toLaneChangeRight': Transition('LaneChangeRight'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toOvertake': Transition('Overtake')},
            default_state='DynamicDefault')
        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicUrbanDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        msg_logger.debug("FSM Street Setting: Urban")
        # FSM static
        transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
        if transition_static is not None:
            self.FSM_urban_static.transition(transition_static)
            self.FSM_urban_static.reset_current_state()
        self.FSM_state.behavior_state_static = self.cur_state_static
        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_urban_dynamic.transition(transition_dynamic)
            self.FSM_urban_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = False #True
        else:
            self.FSM_state.no_auto_lane_change = False

        self.FSM_urban_static.execute()
        self.FSM_urban_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_urban_static.cur_state = self.FSM_urban_static.states['StaticDefault']
        self.FSM_state.behavior_state_static = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_urban_dynamic.cur_state = self.FSM_urban_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class Highway(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_highway_static = SimpleFSM(
            possible_states={
                'StaticDefault': StaticDefault(),
                'PrepareLaneMerge': PrepareLaneMerge(BM_state=BM_state),
                'PrepareRoadExit': PrepareRoadExit(BM_state=BM_state),
                'LaneMerge': LaneMerge(BM_state=BM_state),
                'RoadExit': RoadExit(BM_state=BM_state)},
            possible_transitions={
                'toStaticDefault': Transition('StaticDefault'),
                'toPrepareLaneMerge': Transition('PrepareLaneMerge'),
                'toPrepareRoadExit': Transition('PrepareRoadExit'),
                'toLaneMerge': Transition('LaneMerge'),
                'toRoadExit': Transition('RoadExit')},
            default_state='StaticDefault')
        self.FSM_highway_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareLaneChangeLeft': PrepareLaneChangeLeft(BM_state=BM_state),
                'PrepareLaneChangeRight': PrepareLaneChangeRight(BM_state=BM_state),
                'LaneChangeLeft': LaneChangeLeft(BM_state=BM_state),
                'LaneChangeRight': LaneChangeRight(BM_state=BM_state)},
            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareLaneChangeLeft': Transition('PrepareLaneChangeLeft'),
                'toPrepareLaneChangeRight': Transition('PrepareLaneChangeRight'),
                'toLaneChangeLeft': Transition('LaneChangeLeft'),
                'toLaneChangeRight': Transition('LaneChangeRight')},
            default_state='DynamicDefault')

        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicHighwayDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        msg_logger.debug("FSM Street Setting: Highway")
        # FSM static
        transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
        if transition_static is not None:
            self.FSM_highway_static.transition(transition_static)
            self.FSM_highway_static.reset_current_state()
        self.FSM_state.behavior_state_static = self.cur_state_static
        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_highway_dynamic.transition(transition_dynamic)
            self.FSM_highway_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = True
        else:
            self.FSM_state.no_auto_lane_change = False
        self.FSM_highway_static.execute()
        self.FSM_highway_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_highway_static.cur_state = self.FSM_highway_static.states['StaticDefault']
        self.FSM_state.behavior_state = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_highway_dynamic.cur_state = self.FSM_highway_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class StaticDefault(State):

    def execute(self):
        return 0

    def reset_state(self):
        return 0


class NoLaneChanges(State):

    def execute(self):
        msg_logger.debug("FSM Dynamic Behavior State: NoLaneChanges")

    def reset_state(self):
        return 0


class DynamicDefault(State):

    def execute(self):
        return 0

    def reset_state(self):
        return 0


class PrepareLaneChangeLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_prepare_lane_change_left = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneChange': IdentifyFreeSpaceOnTargetLaneForLaneChange(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifyFreeSpaceOnTargetLaneForLaneChange': Transition('IdentifyFreeSpaceOnTargetLaneForLaneChange'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareLaneChangeLeft(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane',
                                                      BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.FSM_state.situation_time_step_counter = 0

    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Lane Change Left")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_prepare_lane_change_left.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        # velocity planner actions
        self.FSM_prepare_lane_change_left.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_prepare_lane_change_left.cur_state = self.FSM_prepare_lane_change_left.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_dynamic = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.FSM_state.situation_time_step_counter = 0


class LaneChangeLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_lane_change_left = SimpleFSM(
            possible_states={
                'InitiateLaneChange': InitiateLaneChange(),
                'EgoVehicleBetweenTwoLanes': EgoVehicleBetweenTwoLanes(),
                'LaneChangeComplete': LaneChangeComplete()},
            possible_transitions={
                'toInitiateLaneChange': Transition('InitiateLaneChange'),
                'toEgoVehicleBetweenTwoLanes': Transition('EgoVehicleBetweenTwoLanes'),
                'toLaneChangeComplete': Transition('LaneChangeComplete')},
            default_state='InitiateLaneChange')
        self.logic = Logic.LogicLaneChangeLeft(start_state='InitiateLaneChange', BM_state=BM_state)
        self.cur_state = 'InitiateLaneChange'
        self.FSM_state.situation_time_step_counter = 0

    def execute(self):
        msg_logger.debug("FSM Dynamic Behavior State: Changing Lane to the Left")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_lane_change_left.transition(transition)
        self.FSM_state._dynamic = self.cur_state

        # actions
        vehicle_shape = Rectangle(length=self.BM_state.vehicle_params.length / 2,
                                  width=self.BM_state.vehicle_params.width / 2,
                                  center=self.BM_state.ego_state.position,
                                  orientation=self.BM_state.ego_state.orientation)
        self.FSM_state.detected_lanelets = self.BM_state.scenario.lanelet_network.find_lanelet_by_shape(vehicle_shape)
        self.FSM_lane_change_left.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'InitiateLaneChange'
        self.logic.reset_state(state='InitiateLaneChange')
        self.FSM_lane_change_left.cur_state = self.FSM_lane_change_left.states['InitiateLaneChange']
        self.FSM_state.situation_state_dynamic = 'InitiateLaneChange'
        self.FSM_state.detected_lanelets = None
        self.FSM_state.situation_time_step_counter = 0


class PrepareLaneChangeRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_prepare_lane_change_right = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneChange': IdentifyFreeSpaceOnTargetLaneForLaneChange(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifyFreeSpaceOnTargetLaneForLaneChange': Transition('IdentifyFreeSpaceOnTargetLaneForLaneChange'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareLaneChangeRight(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane',
                                                       BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.FSM_state.situation_time_step_counter = 0

    def execute(self):
        msg_logger.debug("FSM Dynamic Behavior State: Preparing for Lane Change Right")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_prepare_lane_change_right.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        self.FSM_prepare_lane_change_right.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_prepare_lane_change_right.cur_state = self.FSM_prepare_lane_change_right.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_dynamic = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.FSM_state.situation_time_step_counter = 0


class LaneChangeRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_lane_change_right = SimpleFSM(
            possible_states={
                'InitiateLaneChange': InitiateLaneChange(),
                'EgoVehicleBetweenTwoLanes': EgoVehicleBetweenTwoLanes(),
                'LaneChangeComplete': LaneChangeComplete()},
            possible_transitions={
                'toInitiateLaneChange': Transition('InitiateLaneChange'),
                'toEgoVehicleBetweenTwoLanes': Transition('EgoVehicleBetweenTwoLanes'),
                'toLaneChangeComplete': Transition('LaneChangeComplete')},
            default_state='InitiateLaneChange')
        self.logic = Logic.LogicLaneChangeRight(start_state='InitiateLaneChange', BM_state=BM_state)
        self.cur_state = 'InitiateLaneChange'
        self.FSM_state.situation_time_step_counter = 0

    def execute(self):
        msg_logger.debug("FSM Dynamic Behavior State: Changing Lane to the Right")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_lane_change_right.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        vehicle_shape = Rectangle(length=self.BM_state.vehicle_params.length / 2,
                                  width=self.BM_state.vehicle_params.width / 2,
                                  center=self.BM_state.ego_state.position,
                                  orientation=self.BM_state.ego_state.orientation)
        self.FSM_state.detected_lanelets = self.BM_state.scenario.lanelet_network.find_lanelet_by_shape(vehicle_shape)
        self.FSM_lane_change_right.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'InitiateLaneChange'
        self.logic.reset_state(state='InitiateLaneChange')
        self.FSM_lane_change_right.cur_state = self.FSM_lane_change_right.states['InitiateLaneChange']
        self.FSM_state.situation_state_dynamic = 'InitiateLaneChange'
        self.FSM_state.detected_lanelets = None
        self.FSM_state.situation_time_step_counter = 0


class PrepareLaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_prepare_lane_merge = SimpleFSM(
            possible_states={
                'EstimateMergingLaneLengthAndEmergencyStopPoint': EstimateMergingLaneLengthAndEmergencyStopPoint(),
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneMerge': IdentifyFreeSpaceOnTargetLaneForLaneMerge(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toEstimateMergingLaneLengthAndEmergencyStopPoint': Transition(
                    'EstimateMergingLaneLengthAndEmergencyStopPoint'),
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifyFreeSpaceOnTargetLaneForLaneMerge': Transition('IdentifyFreeSpaceOnTargetLaneForLaneMerge'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='EstimateMergingLaneLengthAndEmergencyStopPoint')
        self.logic = Logic.LogicPrepareLaneMerge(start_state='EstimateMergingLaneLengthAndEmergencyStopPoint')
        self.cur_state = 'EstimateMergingLaneLengthAndEmergencyStopPoint'

    def execute(self):
        msg_logger.debug("FSM Dynamic Behavior State: Preparing for Lane Merge")
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_prepare_lane_merge.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_prepare_lane_merge.execute()

    def reset_state(self):
        self.cur_state = 'EstimateMergingLaneLengthAndEmergencyStopPoint'
        self.logic.reset_state(state='EstimateMergingLaneLengthAndEmergencyStopPoint')
        self.FSM_prepare_lane_merge.cur_state = self.FSM_prepare_lane_merge.states[
            'EstimateMergingLaneLengthAndEmergencyStopPoint']
        self.FSM_state.situation_state_static = 'EstimateMergingLaneLengthAndEmergencyStopPoint'


class LaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_lane_merge = SimpleFSM(
            possible_states={
                'InitiateLaneMerge': InitiateLaneMerge(),
                'EgoVehicleBetweenTwoLanes': EgoVehicleBetweenTwoLanes(),
                'BehaviorStateComplete': BehaviorStateComplete()},
            possible_transitions={
                'toInitiateLaneMerge': Transition('InitiateLaneMerge'),
                'toEgoVehicleBetweenTwoLanes': Transition('EgoVehicleBetweenTwoLanes'),
                'toBehaviorStateComplete': Transition('BehaviorStateComplete')},
            default_state='InitiateLaneMerge')
        self.logic = Logic.LogicLaneMerge(start_state='InitiateLaneMerge')
        self.cur_state = 'InitiateLaneMerge'

    def execute(self):
        msg_logger.debug("FSM Behavior State: Merging in Lane")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_lane_merge.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_lane_merge.execute()

    def reset_state(self):
        self.cur_state = 'InitiateLaneMerge'
        self.logic.reset_state(state='InitiateLaneMerge')
        self.FSM_lane_merge.cur_state = self.FSM_lane_merge.states['InitiateLaneMerge']
        self.FSM_state.situation_state_static = 'InitiateLaneMerge'


class PrepareRoadExit(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_prepare_road_exit = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneMerge': IdentifyFreeSpaceOnTargetLaneForLaneMerge(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifyFreeSpaceOnTargetLaneForLaneMerge': Transition('IdentifyFreeSpaceOnTargetLaneForLaneMerge'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareRoadExit(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Road Exit")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_prepare_road_exit.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_prepare_road_exit.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_prepare_road_exit.cur_state = self.FSM_prepare_road_exit.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = 'IdentifyTargetLaneAndVehiclesOnTargetLane'


class RoadExit(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_road_exit = SimpleFSM(
            possible_states={
                'InitiateRoadExit': InitiateRoadExit(),
                'EgoVehicleBetweenTwoLanes': EgoVehicleBetweenTwoLanes(),
                'BehaviorStateComplete': BehaviorStateComplete()},
            possible_transitions={
                'toInitiateRoadExit': Transition('InitiateRoadExit'),
                'toEgoVehicleBetweenTwoLanes': Transition('EgoVehicleBetweenTwoLanes'),
                'toBehaviorStateComplete': Transition('BehaviorStateComplete')},
            default_state='InitiateRoadExit')
        self.logic = Logic.LogicRoadExit(start_state='InitiateRoadExit')
        self.cur_state = 'InitiateRoadExit'

    def execute(self):
        msg_logger.debug("FSM Behavior State: Exiting Road")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_road_exit.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_road_exit.execute()

    def reset_state(self):
        self.cur_state = 'InitiateRoadExit'
        self.logic.reset_state(state='InitiateRoadExit')
        self.FSM_road_exit.cur_state = self.FSM_road_exit.states['InitiateRoadExit']
        self.FSM_state.situation_state_static = 'InitiateRoadExit'


class PrepareIntersection(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Intersection")
        pass

    def reset_state(self):
        return 0


class Intersection(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Intersection")

    def reset_state(self):
        return 0


class PrepareTurnLeft(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Left Turn")

    def reset_state(self):
        return 0


class TurnLeft(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Turning Left")

    def reset_state(self):
        return 0


class PrepareTurnRight(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Right Turn")

    def reset_state(self):
        return 0


class TurnRight(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Turning Right")

    def reset_state(self):
        return 0


class PrepareOvertake(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Overtake")

    def reset_state(self):
        return 0


class Overtake(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Overtaking")

    def reset_state(self):
        return 0


class PrepareTrafficLight(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_prepare_traffic_light = SimpleFSM(
            possible_states={
                'ObservingTrafficLight': ObservingTrafficLight(BM_state),
                'SlowingDown': SlowingDown(BM_state)},
            possible_transitions={
                'toObservingTrafficLight': Transition('ObservingTrafficLight'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='ObservingTrafficLight')
        self.logic = Logic.LogicPrepareTrafficLight(start_state='ObservingTrafficLight', BM_state=BM_state)
        self.cur_state = 'ObservingTrafficLight'

    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Traffic Light")

        self.FSM_state.traffic_light_state = self.BM_state.current_static_goal.goal_object.get_state_at_time_step(
            time_step=self.BM_state.time_step).value

        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_prepare_traffic_light.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_prepare_traffic_light.execute()
        msg_logger.debug("FSM Behavior State: Traffic Light is: ", self.FSM_state.traffic_light_state)

    def reset_state(self):
        self.cur_state = 'ObservingTrafficLight'
        self.logic.reset_state(state='ObservingTrafficLight')
        self.FSM_prepare_traffic_light.cur_state = self.FSM_prepare_traffic_light.states[
            'ObservingTrafficLight']
        self.FSM_state.situation_state_static = 'ObservingTrafficLight'


class TrafficLight(State):
    """'GreenLight, Stopping, WaitingForGreenLight, ContinueDriving'"""

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.traffic_light_state == 'green':
            self.init_state = 'GreenLight'
        else:
            self.init_state = 'Stopping'

        self.FSM_traffic_light = SimpleFSM(
            possible_states={
                'GreenLight': GreenLight(BM_state=BM_state),
                'Stopping': Stopping(BM_state=BM_state),
                'WaitingForGreenLight': WaitingForGreenLight(BM_state=BM_state),
                'ContinueDriving': ContinueDriving(BM_state=BM_state)},
            possible_transitions={
                'toGreenLight': Transition('GreenLight'),
                'toStopping': Transition('Stopping'),
                'toWaitingForGreenLight': Transition('WaitingForGreenLight'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicTrafficLight(start_state=self.init_state, BM_state=BM_state)
        self.cur_state = self.init_state

    def execute(self):
        msg_logger.debug("FSM Behavior State: Traffic Light")

        self.FSM_state.traffic_light_state = self.BM_state.current_static_goal.goal_object.get_state_at_time_step(
            time_step=self.BM_state.time_step).value

        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_traffic_light.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_traffic_light.execute()
        msg_logger.debug("FSM Behavior State: Traffic Light is: ", self.FSM_state.traffic_light_state)

    def reset_state(self):
        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_traffic_light.cur_state = self.FSM_traffic_light.states[self.init_state]
        self.FSM_state.situation_state_static = self.init_state


class PrepareCrosswalk(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Crosswalk")

    def reset_state(self):
        return 0


class Crosswalk(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Crosswalk")

    def reset_state(self):
        return 0


class PrepareStopSign(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Stop Sign")

    def reset_state(self):
        return 0


class StopSign(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Stop Sign")

    def reset_state(self):
        return 0


class PrepareYieldSign(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Preparing for Yield Sign")

    def reset_state(self):
        return 0


class YieldSign(State):
    def execute(self):
        msg_logger.debug("FSM Behavior State: Yield Sign")

    def reset_state(self):
        return 0


'''Situation States'''


class IdentifyTargetLaneAndVehiclesOnTargetLane(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Identifying target lane and vehicles on target lane")
        # identify target lanelet
        if self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeRight':
            self.FSM_state.lane_change_target_lanelet_id = self.BM_state.current_lanelet.adj_right
            self.FSM_state.lane_change_target_lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                self.FSM_state.lane_change_target_lanelet_id)
        elif self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeLeft':
            self.FSM_state.lane_change_target_lanelet_id = self.BM_state.current_lanelet.adj_left
            self.FSM_state.lane_change_target_lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                self.FSM_state.lane_change_target_lanelet_id)
        elif self.FSM_state.behavior_state_dynamic == 'PrepareMergingLanes':
            return 0  # TODO
        # identify relevant objects on target lane
        self.FSM_state.obstacles_on_target_lanelet = hf.get_predicted_obstacles_on_lanelet(
            predictions=self.BM_state.predictions,
            lanelet_network=self.BM_state.scenario.lanelet_network,
            lanelet_id=self.FSM_state.lane_change_target_lanelet_id,
            search_point=self.BM_state.ego_state.position,
            search_distance=self.BM_state.VP_state.speed_limit_default * 2)


class IdentifyFreeSpaceOnTargetLaneForLaneChange(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Identifying free space on target lane")
        time1 = time.time()
        # if target lane is empty withing search radius
        if not self.FSM_state.obstacles_on_target_lanelet:
            self.FSM_state.free_space_on_target_lanelet = True
            msg_logger.debug('\n*Free Space: Target Lane Empty\n')
        else:
            self.FSM_state.free_space_offset = 0
            ego_position_offsets = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15]
            # obstacle behind or next to ego vehicle
            risk_factor = 1.1
            for ego_position_offset in ego_position_offsets:
                free_space_on_target_lanelet = True
                if self.FSM_state.free_space_offset == 0 and not self.FSM_state.free_space_on_target_lanelet:
                    for obstacle_id in self.FSM_state.obstacles_on_target_lanelet:
                        obstacle = self.FSM_state.obstacles_on_target_lanelet.get(obstacle_id)
                        obstacle_position = obstacle.get('pos_list')[0]
                        obstacle_velocity = obstacle.get('v_list')[0]
                        try:
                            obstacle_pos_s = self.BM_state.PP_state.cl_ref_coordinate_system.\
                                convert_to_curvilinear_coords(x=obstacle_position[0], y=obstacle_position[1])[0]
                        except:
                            msg_logger.error('\n*Free Space: obstacle out of projection domain. obstacle position: \n',
                                  obstacle_position)
                            continue
                        '''
                        calc example for velocity dependent safety distance
                        minimum distance of velocity * 2 in meters => e.g. 100 km/h = 27.8 m/s => 55.6 m distance
                        '''
                        if obstacle_pos_s + ego_position_offset <= self.BM_state.ref_position_s + ego_position_offset:
                            # if not obstacle is further behind than half the velocity distance.
                            if not (obstacle_pos_s < (self.BM_state.ref_position_s + ego_position_offset -
                                                      self.BM_state.vehicle_params.length / 2 -
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False

                                msg_logger.debug('\n*Free Space: obstacle behind ego vehicle')
                                msg_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                msg_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                msg_logger.debug('*Free Space: obstacle is not further behind than half the velocity distance\n')
                        # obstacle in front of ego vehicle
                        else:
                            # if not obstacle is further away than half the velocity distance.
                            if not (obstacle_pos_s > (self.BM_state.ref_position_s + ego_position_offset +
                                                      self.BM_state.vehicle_params.length +
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False
                                msg_logger.debug('\n*Free Space: obstacle in front of ego vehicle')
                                msg_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                msg_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                msg_logger.debug('*Free Space: obstacle is not further away than half the velocity distance\n')
                if ego_position_offset == 0:
                    if free_space_on_target_lanelet:
                        self.FSM_state.free_space_on_target_lanelet = True
                        break
                else:
                    if free_space_on_target_lanelet:
                        msg_logger.debug('\n*Free Space: free space detected with offset: \n', ego_position_offset)
                        self.FSM_state.free_space_offset = ego_position_offset
                        self.FSM_state.change_velocity_for_lane_change = True
                        break
        if self.FSM_state.free_space_on_target_lanelet:
            msg_logger.debug('\n*Free Space: free space detected\n')
        time2 = time.time()
        msg_logger.debug('Free Space calc time: ', time2-time1)


class IdentifyFreeSpaceOnTargetLaneForLaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Identifying free space on target lane")
        time1 = time.time()
        # if target lane is empty withing search radius
        if not self.FSM_state.obstacles_on_target_lanelet:
            self.FSM_state.free_space_on_target_lanelet = True
            msg_logger.debug('\n*Free Space: Target Lane Empty\n')
        else:
            self.FSM_state.free_space_offset = 0
            ego_position_offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9, -10, 10, -11, 11,
                                    -12, 12, -13, 13, -14, 14, -15, 15]
            # obstacle behind or next to ego vehicle
            risk_factor = 1.0
            for ego_position_offset in ego_position_offsets:
                free_space_on_target_lanelet = True
                if self.FSM_state.free_space_offset == 0 and not self.FSM_state.free_space_on_target_lanelet:
                    if ego_position_offset != 0:
                        msg_logger.debug('\n*Free Space: Ego position offset: \n', ego_position_offset)
                    for obstacle_id in self.FSM_state.obstacles_on_target_lanelet:
                        obstacle = self.FSM_state.obstacles_on_target_lanelet.get(obstacle_id)
                        obstacle_position = obstacle.get('pos_list')[0]
                        obstacle_velocity = obstacle.get('v_list')[0]
                        try:
                            obstacle_pos_s = self.BM_state.PP_state.cl_ref_coordinate_system.\
                                convert_to_curvilinear_coords(x=obstacle_position[0], y=obstacle_position[1])[0]
                        except:
                            msg_logger.error('\n*Free Space: obstacle out of projection domain. obstacle position: \n',
                                  obstacle_position)
                            continue
                        '''
                        calc example for velocity dependent safety distance
                        minimum distance of velocity * 2 in meters => e.g. 100 km/h = 27.8 m/s => 55.6 m distance
                        '''
                        if obstacle_pos_s + ego_position_offset <= self.BM_state.ref_position_s + ego_position_offset:
                            # if not obstacle is further behind than half the velocity distance.
                            if not (obstacle_pos_s < (self.BM_state.ref_position_s + ego_position_offset -
                                                      self.BM_state.vehicle_params.length / 2 -
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False
                                msg_logger.debug('\n*Free Space: obstacle behind ego vehicle')
                                msg_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                msg_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                msg_logger.debug('*Free Space: obstacle is not further behind than half the velocity distance\n')
                        # obstacle in front of ego vehicle
                        else:
                            # if not obstacle is further away than half the velocity distance.
                            if not (obstacle_pos_s > (self.BM_state.ref_position_s + ego_position_offset +
                                                      self.BM_state.vehicle_params.length +
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False
                                msg_logger.debug('\n*Free Space: obstacle in front of ego vehicle')
                                msg_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                msg_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                msg_logger.debug('*Free Space: obstacle is not further away than half the velocity distance\n')
                if ego_position_offset == 0:
                    if free_space_on_target_lanelet:
                        self.FSM_state.free_space_on_target_lanelet = True
                        break
                else:
                    if free_space_on_target_lanelet:
                        msg_logger.debug('\n*Free Space: free space detected with offset: \n', ego_position_offset)
                        self.FSM_state.free_space_offset = ego_position_offset
                        self.FSM_state.change_velocity_for_lane_change = True
                        break
        if self.FSM_state.free_space_on_target_lanelet:
            msg_logger.debug('\n*Free Space: free space detected\n')
        time2 = time.time()
        msg_logger.debug('Free Space calc time: ', time2-time1)


class PreparationsDone(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Preparations Done")


class InitiateLaneChange(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Initiating lane change")


class EgoVehicleBetweenTwoLanes(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Vehicle crossing lanes")


class LaneChangeComplete(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Lane change complete")


class BehaviorStateComplete(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Behavior State complete")


class EstimateMergingLaneLengthAndEmergencyStopPoint(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Estimating merging lane length and setting emergency stop point")


class InitiateLaneMerge(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Initiating lane merge")


class InitiateRoadExit(State):
    def execute(self):
        msg_logger.debug("FSM Situation State: Preparations Done")


class ObservingTrafficLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Observing Traffic Light")


class SlowingDown(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.VP_state = BM_state.VP_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Slowing Down Car")
        self.FSM_state.slowing_car_for_traffic_light = True
        self.VP_state.dist_to_tlna = self.BM_state.current_static_goal.stop_point_s - self.BM_state.nav_position_s - \
                                   self.BM_state.vehicle_params.length
        self.VP_state.stop_distance = self.VP_state.dist_to_tl

        # check for other stopping cars in front of vehicle
        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.dist_preceding_veh - self.BM_state.vehicle_params.length - \
                    self.VP_state.closest_preceding_vehicle.get('shape').get('length') \
                    <= self.VP_state.dist_to_tl:
                self.VP_state.stop_distance = self.VP_state.dist_preceding_veh - \
                                              self.BM_state.vehicle_params.length - \
                                              self.VP_state.closest_preceding_vehicle.get('shape').get('length')
        msg_logger.debug("FSM distance to stopping line is: ", self.VP_state.dist_to_tl)


class GreenLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Green Light")
        self.FSM_state.slowing_car_for_traffic_light = False


class Stopping(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.VP_state = BM_state.VP_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Stopping the Car")
        self.FSM_state.slowing_car_for_traffic_light = True
        self.VP_state.dist_to_tl = self.BM_state.current_static_goal.stop_point_s - self.BM_state.nav_position_s - \
                                   self.BM_state.vehicle_params.length
        self.VP_state.stop_distance = self.VP_state.dist_to_tl

        # check for other stopping cars in front of vehicle
        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.dist_preceding_veh - self.BM_state.vehicle_params.length - \
                    self.VP_state.closest_preceding_vehicle.get('shape').get('length') \
                    <= self.VP_state.dist_to_tl:
                self.VP_state.stop_distance = self.VP_state.dist_preceding_veh - \
                                              self.BM_state.vehicle_params.length - \
                                              self.VP_state.closest_preceding_vehicle.get('shape').get('length')

        msg_logger.debug("FSM distance to stopping line is: ", self.VP_state.dist_to_tl)


class WaitingForGreenLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Waiting for Green Light")


class ContinueDriving(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        msg_logger.debug("FSM Situation State: Continue driving")
        self.FSM_state.slowing_car_for_traffic_light = False
        self.FSM_state.waiting_for_green_light = False


'''Transition State'''


class Transition(object):
    """transition base class"""

    def __init__(self, to_state):
        self.to_state = to_state

    def execute(self):
        msg_logger.debug("FSM Transitioning to: " + str(self.to_state))
