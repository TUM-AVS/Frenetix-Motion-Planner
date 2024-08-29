__author__ = "Moritz Ellermann, Rainer Trauth"
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
bhv_logger = logging.getLogger("Behavior_logger")

'''base type 'State' used to create states'''
State = type("State", (object,), {})


class EgoFSM(object):
    """
    Finite State Machine class: Used by the Behavior Planner mirror the state of the real ego vehicle. To reduce
    complexity the FSM is built in recursive fashion with 3 layers. First layer is the Street Setting States, Second
    Layer is the Behavior States and Third Layer is the Situation States.

    Possible States are:

    Street Setting: 'Highway', 'Country', 'Urban'

    Behavior:   'PrepareLaneChangeLeft', 'LaneChangeLeft',
                'PrepareLaneChangeRight', 'LaneChangeRight',
                'PrepareLaneMerge', 'LaneMerge',
                'PrepareRoadExit', 'RoadExit',
                'PrepareOvertake', 'Overtake',
                'PrepareTurnLeft', 'TurnLeft',
                'PrepareTurnRight', 'TurnRight',
                'PrepareTrafficLight', 'TrafficLight',
                'PrepareYieldSign', 'YieldSign',
                'PrepareStopSign', 'StopSign'

    Situation: 'IdentifyTargetLaneAndVehiclesOnTargetLane', 'IdentifyFreeSpaceOnTargetLaneForLaneChange', 'PreparationsDone',
                'InitiateLaneChange', 'EgoVehicleBetweenTwoLanes', 'BehaviorStateComplete',
                'EstimateMergingLaneLengthAndEmergencyStopPoint', 'InitiateLaneMerge, 'InitiateRoadExit'
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


####################################################################################################
#                                          FSM Base Class                                          #
####################################################################################################


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


####################################################################################################
#                                      Street Setting States                                       #
####################################################################################################


class Country(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_static = SimpleFSM(
            possible_states={
                'StaticDefault': StaticDefault(),
                'PrepareIntersection': PrepareIntersection(BM_state=BM_state),
                'PrepareTurnLeft': PrepareTurnLeft(BM_state=BM_state),
                'PrepareTurnRight': PrepareTurnRight(BM_state=BM_state),
                'PrepareLaneMerge': PrepareLaneMerge(BM_state=BM_state),
                'PrepareTrafficLight': PrepareTrafficLight(BM_state=BM_state),
                'Intersection': Intersection(BM_state=BM_state),
                'TurnLeft': TurnLeft(BM_state=BM_state),
                'TurnRight': TurnRight(BM_state=BM_state),
                'LaneMerge': LaneMerge(BM_state=BM_state),
                'TrafficLight': TrafficLight(BM_state=BM_state)},
            possible_transitions={
                'toStaticDefault': Transition('StaticDefault'),
                'toPrepareIntersection': Transition('PrepareIntersection'),
                'toPrepareTurnLeft': Transition('PrepareTurnLeft'),
                'toPrepareTurnRight': Transition('PrepareTurnRight'),
                'toPrepareLaneMerge': Transition('PrepareLaneMerge'),
                'toPrepareTrafficLight': Transition('PrepareTrafficLight'),
                'toIntersection': Transition('Intersection'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toLaneMerge': Transition('LaneMerge'),
                'toTrafficLight': Transition('TrafficLight')},
            default_state='StaticDefault')
        self.FSM_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareOvertake': PrepareOvertake(BM_state=BM_state),
                'Overtake': Overtake(BM_state=BM_state),
                'FinishOvertake': FinishOvertake(BM_state=BM_state)},
            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareOvertake': Transition('PrepareOvertake'),
                'toOvertake': Transition('Overtake'),
                'toFinishOvertake': Transition('FinishOvertake')},
            default_state='DynamicDefault')
        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicCountryDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        bhv_logger.debug("FSM Street Setting: Country")
        # FSM static
        if not self.BM_state.plan_dynamics_only:
            transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
            if transition_static is not None:
                self.FSM_static.transition(transition_static)
                self.FSM_static.reset_current_state()
            self.FSM_state.behavior_state_static = self.cur_state_static
        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_dynamic.transition(transition_dynamic)
            self.FSM_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = True
        else:
            self.FSM_state.no_auto_lane_change = False

        if not self.BM_state.plan_dynamics_only:
            self.FSM_static.execute()
        self.FSM_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_static.cur_state = self.FSM_static.states['StaticDefault']
        self.FSM_state.behavior_state_static = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_dynamic.cur_state = self.FSM_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class Urban(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_static = SimpleFSM(
            possible_states={
                'StaticDefault': StaticDefault(),
                'PrepareIntersection': PrepareIntersection(BM_state=BM_state),
                'PrepareTurnLeft': PrepareTurnLeft(BM_state=BM_state),
                'PrepareTurnRight': PrepareTurnRight(BM_state=BM_state),
                'PrepareLaneMerge': PrepareLaneMerge(BM_state=BM_state),
                'PrepareTrafficLight': PrepareTrafficLight(BM_state=BM_state),
                'PrepareCrosswalk': PrepareCrosswalk(BM_state=BM_state),
                'PrepareStopSign': PrepareStopSign(BM_state=BM_state),
                'PrepareYieldSign': PrepareYieldSign(BM_state=BM_state),
                'Intersection': Intersection(BM_state=BM_state),
                'TurnLeft': TurnLeft(BM_state=BM_state),
                'TurnRight': TurnRight(BM_state=BM_state),
                'LaneMerge': LaneMerge(BM_state=BM_state),
                'TrafficLight': TrafficLight(BM_state=BM_state),
                'Crosswalk': Crosswalk(BM_state=BM_state),
                'StopSign': StopSign(BM_state=BM_state),
                'YieldSign': YieldSign(BM_state=BM_state)},

            possible_transitions={
                'toStaticDefault': Transition('StaticDefault'),
                'toPrepareIntersection': Transition('PrepareIntersection'),
                'toPrepareTurnLeft': Transition('PrepareTurnLeft'),
                'toPrepareTurnRight': Transition('PrepareTurnRight'),
                'toPrepareLaneMerge': Transition('PrepareLaneMerge'),
                'toPrepareTrafficLight': Transition('PrepareTrafficLight'),
                'toPrepareCrosswalk': Transition('PrepareCrosswalk'),
                'toPrepareStopSign': Transition('PrepareStopSign'),
                'toPrepareYieldSign': Transition('PrepareYieldSign'),
                'toIntersection': Transition('Intersection'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toLaneMerge': Transition('LaneMerge'),
                'toTrafficLight': Transition('TrafficLight'),
                'toCrosswalk': Transition('Crosswalk'),
                'toStopSign': Transition('StopSign'),
                'toYieldSign': Transition('YieldSign')},
            default_state='StaticDefault')
        self.FSM_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareLaneChangeLeft': PrepareLaneChangeLeft(BM_state=BM_state),
                'PrepareLaneChangeRight': PrepareLaneChangeRight(BM_state=BM_state),
                'LaneChangeLeft': LaneChangeLeft(BM_state=BM_state),
                'LaneChangeRight': LaneChangeRight(BM_state=BM_state),
                'PrepareOvertake': PrepareOvertake(BM_state=BM_state),
                'Overtake': Overtake(BM_state=BM_state),
                'FinishOvertake': FinishOvertake(BM_state=BM_state)},

            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareLaneChangeLeft': Transition('PrepareLaneChangeLeft'),
                'toPrepareLaneChangeRight': Transition('PrepareLaneChangeRight'),
                'toLaneChangeLeft': Transition('LaneChangeLeft'),
                'toLaneChangeRight': Transition('LaneChangeRight'),
                'toTurnLeft': Transition('TurnLeft'),
                'toTurnRight': Transition('TurnRight'),
                'toPrepareOvertake': Transition('PrepareOvertake'),
                'toOvertake': Transition('Overtake'),
                'toFinishOvertake': Transition('FinishOvertake')},
            default_state='DynamicDefault')
        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicUrbanDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        bhv_logger.debug("FSM Street Setting: Urban")
        # FSM static
        if not self.BM_state.plan_dynamics_only:
            transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
            if transition_static is not None:
                self.FSM_static.transition(transition_static)
                self.FSM_static.reset_current_state()
            self.FSM_state.behavior_state_static = self.cur_state_static

        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_dynamic.transition(transition_dynamic)
            self.FSM_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = False  # True
        else:
            self.FSM_state.no_auto_lane_change = False

        if not self.BM_state.plan_dynamics_only:
            self.FSM_static.execute()
        self.FSM_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_static.cur_state = self.FSM_static.states['StaticDefault']
        self.FSM_state.behavior_state_static = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_dynamic.cur_state = self.FSM_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class Highway(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_static = SimpleFSM(
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
        self.FSM_dynamic = SimpleFSM(
            possible_states={
                'DynamicDefault': DynamicDefault(),
                'NoLaneChanges': NoLaneChanges(),
                'PrepareLaneChangeLeft': PrepareLaneChangeLeft(BM_state=BM_state),
                'PrepareLaneChangeRight': PrepareLaneChangeRight(BM_state=BM_state),
                'LaneChangeLeft': LaneChangeLeft(BM_state=BM_state),
                'LaneChangeRight': LaneChangeRight(BM_state=BM_state),
                'PrepareOvertake': PrepareOvertake(BM_state=BM_state),
                'Overtake': Overtake(BM_state=BM_state),
                'FinishOvertake': FinishOvertake(BM_state=BM_state)},
            possible_transitions={
                'toDynamicDefault': Transition('DynamicDefault'),
                'toNoLaneChanges': Transition('NoLaneChanges'),
                'toPrepareLaneChangeLeft': Transition('PrepareLaneChangeLeft'),
                'toPrepareLaneChangeRight': Transition('PrepareLaneChangeRight'),
                'toLaneChangeLeft': Transition('LaneChangeLeft'),
                'toLaneChangeRight': Transition('LaneChangeRight'),
                'toPrepareOvertake': Transition('PrepareOvertake'),
                'toOvertake': Transition('Overtake'),
                'toFinishOvertake': Transition('FinishOvertake')},
            default_state='DynamicDefault')

        self.logic_static = Logic.LogicBehaviorStatic(start_state='StaticDefault', BM_state=BM_state)
        self.logic_dynamic = Logic.LogicHighwayDynamic(start_state='DynamicDefault', BM_state=BM_state)
        self.cur_state_static = 'StaticDefault'
        self.cur_state_dynamic = 'DynamicDefault'

    def execute(self):
        bhv_logger.debug("FSM Street Setting: Highway")
        # FSM static
        if not self.BM_state.plan_dynamics_only:
            transition_static, self.cur_state_static = self.logic_static.execute(self.cur_state_static)
            if transition_static is not None:
                self.FSM_static.transition(transition_static)
                self.FSM_static.reset_current_state()
            self.FSM_state.behavior_state_static = self.cur_state_static
        # FSM dynamic
        transition_dynamic, self.cur_state_dynamic = self.logic_dynamic.execute(self.cur_state_dynamic)
        if transition_dynamic is not None:
            self.FSM_dynamic.transition(transition_dynamic)
            self.FSM_dynamic.reset_current_state()
        self.FSM_state.behavior_state_dynamic = self.cur_state_dynamic

        # actions
        if self.cur_state_static != 'StaticDefault':
            self.FSM_state.no_auto_lane_change = True
        else:
            self.FSM_state.no_auto_lane_change = False

        if not self.BM_state.plan_dynamics_only:
            self.FSM_static.execute()
        self.FSM_dynamic.execute()

    def reset_state(self):
        self.cur_state_static = 'StaticDefault'
        self.logic_static.reset_state(state='StaticDefault')
        self.FSM_static.cur_state = self.FSM_static.states['StaticDefault']
        self.FSM_state.behavior_state_static = 'StaticDefault'

        self.cur_state_dynamic = 'DynamicDefault'
        self.logic_dynamic.reset_state(state='DynamicDefault')
        self.FSM_dynamic.cur_state = self.FSM_dynamic.states['DynamicDefault']
        self.FSM_state.behavior_state_dynamic = 'DynamicDefault'


class StaticDefault(State):

    def execute(self):
        return 0

    def reset_state(self):
        return 0


class NoLaneChanges(State):

    def execute(self):
        bhv_logger.debug("FSM Dynamic Behavior State: NoLaneChanges")

    def reset_state(self):
        return 0


class DynamicDefault(State):

    def execute(self):
        return 0

    def reset_state(self):
        return 0


####################################################################################################
#                                         Behavior States                                          #
####################################################################################################


class PrepareLaneChangeLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Behavior State: Preparing for Lane Change Left")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        # velocity planner actions
        self.FSM_situation.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_dynamic = None
        self.FSM_state.situation_time_step_counter = 0


class LaneChangeLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Dynamic Behavior State: Changing Lane to the Left")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        vehicle_shape = Rectangle(length=self.BM_state.vehicle_params.length / 2,
                                  width=self.BM_state.vehicle_params.width / 2,
                                  center=self.BM_state.ego_state.position,
                                  orientation=self.BM_state.ego_state.orientation)
        self.FSM_state.detected_lanelets = self.BM_state.scenario.lanelet_network.find_lanelet_by_shape(vehicle_shape)
        self.FSM_situation.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'InitiateLaneChange'
        self.logic.reset_state(state='InitiateLaneChange')
        self.FSM_situation.cur_state = self.FSM_situation.states['InitiateLaneChange']
        self.FSM_state.situation_state_dynamic = None
        self.FSM_state.detected_lanelets = None
        self.FSM_state.situation_time_step_counter = 0


class PrepareLaneChangeRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Dynamic Behavior State: Preparing for Lane Change Right")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        self.FSM_situation.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_dynamic = None
        self.FSM_state.situation_time_step_counter = 0


class LaneChangeRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Dynamic Behavior State: Changing Lane to the Right")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_dynamic = self.cur_state

        # actions
        vehicle_shape = Rectangle(length=self.BM_state.vehicle_params.length / 2,
                                  width=self.BM_state.vehicle_params.width / 2,
                                  center=self.BM_state.ego_state.position,
                                  orientation=self.BM_state.ego_state.orientation)
        self.FSM_state.detected_lanelets = self.BM_state.scenario.lanelet_network.find_lanelet_by_shape(vehicle_shape)
        self.FSM_situation.execute()
        self.FSM_state.situation_time_step_counter += 1

    def reset_state(self):
        self.cur_state = 'InitiateLaneChange'
        self.logic.reset_state(state='InitiateLaneChange')
        self.FSM_situation.cur_state = self.FSM_situation.states['InitiateLaneChange']
        self.FSM_state.situation_state_dynamic = None
        self.FSM_state.detected_lanelets = None
        self.FSM_state.situation_time_step_counter = 0


class PrepareLaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Dynamic Behavior State: Preparing for Lane Merge")
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'EstimateMergingLaneLengthAndEmergencyStopPoint'
        self.logic.reset_state(state='EstimateMergingLaneLengthAndEmergencyStopPoint')
        self.FSM_situation.cur_state = self.FSM_situation.states[
            'EstimateMergingLaneLengthAndEmergencyStopPoint']
        self.FSM_state.situation_state_static = None


class LaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'InitiateLaneMerge': InitiateLaneMerge(),
                'EgoVehicleBetweenTwoLanes': EgoVehicleBetweenTwoLanes(),
                'BehaviorStateComplete': BehaviorStateComplete()},
            possible_transitions={
                'toInitiateLaneMerge': Transition('InitiateLaneMerge'),
                'toEgoVehicleBetweenTwoLanes': Transition('EgoVehicleBetweenTwoLanes'),
                'toBehaviorStateComplete': Transition('BehaviorStateComplete')},
            default_state='InitiateLaneMerge')
        self.logic = Logic.LogicLaneMerge(start_state='InitiateLaneMerge', BM_state=self.BM_state)
        self.cur_state = 'InitiateLaneMerge'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Merging in Lane")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'InitiateLaneMerge'
        self.logic.reset_state(state='InitiateLaneMerge')
        self.FSM_situation.cur_state = self.FSM_situation.states['InitiateLaneMerge']
        self.FSM_state.situation_state_static = None


class PrepareRoadExit(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Behavior State: Preparing for Road Exit")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states[
            'IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = None


class RoadExit(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Behavior State: Exiting Road")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'InitiateRoadExit'
        self.logic.reset_state(state='InitiateRoadExit')
        self.FSM_situation.cur_state = self.FSM_situation.states['InitiateRoadExit']
        self.FSM_state.situation_state_static = None


class PrepareIntersection(State):
    # TODO implement class PrepareIntersection
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'ObservingIntersection': ObservingIntersection(BM_state=self.BM_state),
                'SlowingDown': SlowingDown(BM_state=self.BM_state)},
            possible_transitions={
                'toObservingIntersection': Transition('ObservingIntersection'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='ObservingIntersection')
        self.logic = Logic.LogicPrepareIntersection(start_state='ObservingIntersection', BM_state=BM_state)
        self.cur_state = 'ObservingIntersection'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing for Intersection")
        pass

    def reset_state(self):
        return 0


class Intersection(State):
    # TODO implement class Intersection
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.intersection_clear == True:
            self.init_state = 'IntersectionClear'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
            possible_states={
                'IntersectionClear': IntersectionClear(BM_state=self.BM_state),
                'Stopping': Stopping(BM_state=self.BM_state),
                'WaitingForIntersectionClearance': WaitingForIntersectionClearance(BM_state=self.BM_state),
                'ContinueDriving': ContinueDriving(BM_state=self.BM_state)},
            possible_transitions={
                'toIntersectionClear': Transition('IntersectionClear'),
                'toStopping': Transition('Stopping'),
                'toWaitingForIntersectionClearance': Transition('WaitingForIntersectionClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicIntersection(start_state=self.init_state, BM_state=self.BM_state)
        self.cur_state = self.init_state

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Intersection")

    def reset_state(self):
        return 0


class PrepareTurnLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane':
                    IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state=self.BM_state),
                'SlowingDown': SlowingDown(BM_state=self.BM_state)},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareTurnLeft(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane', BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing for Left Turn")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states['IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = None


class TurnLeft(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.turn_clear == True:
            self.init_state = 'TurnClear'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
            possible_states={
                'TurnClear': TurnClear(BM_state=self.BM_state),
                'Stopping': Stopping(BM_state=self.BM_state),
                'WaitingForTurnClearance': WaitingForTurnClearance(BM_state=self.BM_state),
                'ContinueDriving': ContinueDriving(BM_state=self.BM_state)},
            possible_transitions={
                'toTurnClear': Transition('TurnClear'),
                'toStopping': Transition('Stopping'),
                'toWaitingForTurnClearance': Transition('WaitingForTurnClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicTurnLeft(start_state=self.init_state, BM_state=BM_state)
        self.cur_state = self.init_state

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Turning Left")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        if self.FSM_state.turn_clear == True:
            self.init_state = 'TurnClear'
        else:
            self.init_state = 'Stopping'

        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_situation.cur_state = self.FSM_situation.states[self.init_state]
        self.FSM_state.situation_state_static = None


class PrepareTurnRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane':
                    IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state=self.BM_state),
                'SlowingDown': SlowingDown(BM_state=self.BM_state)},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareTurnRight(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane', BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing for Right Turn")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states['IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = None


class TurnRight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.turn_clear == True:
            self.init_state = 'TurnClear'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
            possible_states={
                'TurnClear': TurnClear(BM_state=self.BM_state),
                'Stopping': Stopping(BM_state=self.BM_state),
                'WaitingForTurnClearance': WaitingForTurnClearance(BM_state=self.BM_state),
                'ContinueDriving': ContinueDriving(BM_state=self.BM_state)},
            possible_transitions={
                'toTurnClear': Transition('TurnClear'),
                'toStopping': Transition('Stopping'),
                'toWaitingForTurnClearance': Transition('WaitingForTurnClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicTurnRight(start_state=self.init_state, BM_state=self.BM_state)
        self.cur_state = self.init_state

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Turning Right")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        if self.FSM_state.turn_clear == True:
            self.init_state = 'TurnClear'
        else:
            self.init_state = 'Stopping'

        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_situation.cur_state = self.FSM_situation.states[self.init_state]
        self.FSM_state.situation_state_static = None


class PrepareOvertake(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifySpeedOfObstaclesOnTargetLane': IdentifySpeedOfObstaclesOnTargetLane(BM_state),
                'AbortOvertake': AbortOvertake(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneMerge': IdentifyFreeSpaceOnTargetLaneForLaneMerge(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifySpeedOfObstaclesOnTargetLane': Transition('IdentifySpeedOfObstaclesOnTargetLane'),
                'AbortOvertake': AbortOvertake(BM_state),  # maybot not needen, insted just setting a flag
                'toIdentifyFreeSpaceOnTargetLaneForLaneMerge': Transition('IdentifyFreeSpaceOnTargetLaneForLaneMerge'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicPrepareOvertake(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane', BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing Overtake")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states['IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = None


class Overtake(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'Overtaking': Overtaking(BM_state),
                'OvertakeComplete': OvertakeComplete(BM_state)},
            possible_transitions={
                'toOvertaking': Transition('Overtaking'),
                'toOvertakeComplete': Transition('OvertakeComplete')},
            default_state='Overtaking')
        self.logic = Logic.LogicOvertake(start_state='Overtaking', BM_state=BM_state)
        self.cur_state = 'Overtaking'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Overtaking")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'Overtaking'
        self.logic.reset_state(state='Overtaking')
        self.FSM_situation.cur_state = self.FSM_situation.states['Overtaking']
        self.FSM_state.situation_state_static = None


class FinishOvertake(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'IdentifyTargetLaneAndVehiclesOnTargetLane': IdentifyTargetLaneAndVehiclesOnTargetLane(BM_state),
                'IdentifySpeedOfObstaclesOnTargetLane': IdentifySpeedOfObstaclesOnTargetLane(BM_state),
                'IdentifyFreeSpaceOnTargetLaneForLaneMerge': IdentifyFreeSpaceOnTargetLaneForLaneMerge(BM_state),
                'PreparationsDone': PreparationsDone()},
            possible_transitions={
                'toIdentifyTargetLaneAndVehiclesOnTargetLane': Transition('IdentifyTargetLaneAndVehiclesOnTargetLane'),
                'toIdentifySpeedOfObstaclesOnTargetLane': Transition('IdentifySpeedOfObstaclesOnTargetLane'),
                'toIdentifyFreeSpaceOnTargetLaneForLaneMerge': Transition('IdentifyFreeSpaceOnTargetLaneForLaneMerge'),
                'toPreparationsDone': Transition('PreparationsDone')},
            default_state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.logic = Logic.LogicFinishOvertake(start_state='IdentifyTargetLaneAndVehiclesOnTargetLane', BM_state=BM_state)
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Finishing Overtake")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
        self.logic.reset_state(state='IdentifyTargetLaneAndVehiclesOnTargetLane')
        self.FSM_situation.cur_state = self.FSM_situation.states['IdentifyTargetLaneAndVehiclesOnTargetLane']
        self.FSM_state.situation_state_static = None


class PrepareTrafficLight(State):

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Behavior State: Preparing for Traffic Light")

        self.FSM_state.traffic_light_state = self.BM_state.current_static_goal.goal_object.get_state_at_time_step(
            time_step=self.BM_state.time_step).value

        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()
        bhv_logger.debug("FSM Behavior State: Traffic Light is: ", self.FSM_state.traffic_light_state)

    def reset_state(self):
        self.cur_state = 'ObservingTrafficLight'
        self.logic.reset_state(state='ObservingTrafficLight')
        self.FSM_situation.cur_state = self.FSM_situation.states[
            'ObservingTrafficLight']
        self.FSM_state.situation_state_static = None


class TrafficLight(State):
    """'GreenLight, Stopping, WaitingForGreenLight, ContinueDriving'"""

    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.traffic_light_state == 'green':
            self.init_state = 'GreenLight'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
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
        bhv_logger.debug("FSM Behavior State: Traffic Light")

        self.FSM_state.traffic_light_state = self.BM_state.current_static_goal.goal_object.get_state_at_time_step(
            time_step=self.BM_state.time_step).value

        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()
        bhv_logger.debug("FSM Behavior State: Traffic Light is: ", self.FSM_state.traffic_light_state)

    def reset_state(self):
        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_situation.cur_state = self.FSM_situation.states[self.init_state]
        self.FSM_state.situation_state_static = None


class PrepareCrosswalk(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'ObservingCrosswalk': ObservingCrosswalk(BM_state),
                'SlowingDown': SlowingDown(BM_state)},
            possible_transitions={
                'toObservingCrosswalk': Transition('ObservingCrosswalk'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='ObservingCrosswalk')
        self.logic = Logic.LogicPrepareCrosswalk(start_state='ObservingCrosswalk', BM_state=BM_state)
        self.cur_state = 'ObservingCrosswalk'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing Crosswalk")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'ObservingCrosswalk'
        self.logic.reset_state(state='ObservingCrosswalk')
        self.FSM_situation.cur_state = self.FSM_situation.states['ObservingCrosswalk']
        self.FSM_state.situation_state_static = None


class Crosswalk(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.crosswalk_clear == True:
            self.init_state = 'CrosswalkClear'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
            possible_states={
                'CrosswalkClear': CrosswalkClear(BM_state),
                'Stopping': Stopping(BM_state),
                'WaitingForCrosswalkClearance': WaitingForCrosswalkClearance(BM_state),
                'ContinueDriving': ContinueDriving(BM_state)},
            possible_transitions={
                'toCrosswalkClear': Transition('CrosswalkClear'),
                'toStopping': Transition('Stopping'),
                'toWaitingForCrosswalkClearance': Transition('WaitingForCrosswalkClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicCrosswalk(start_state=self.init_state, BM_state=BM_state)
        self.cur_state = self.init_state

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Crosswalk")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        if self.FSM_state.crosswalk_clear == True:
            self.init_state = 'CrosswalkClear'
        else:
            self.init_state = 'Stopping'

        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_situation.cur_state = self.FSM_situation.states[self.init_state]
        self.FSM_state.situation_state_static = None


class PrepareStopSign(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'ObservingStopYieldSign': ObservingStopYieldSign(BM_state),
                'SlowingDown': SlowingDown(BM_state)},
            possible_transitions={
                'toObservingStopYieldSign': Transition('ObservingStopYieldSign'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='ObservingStopYieldSign')
        self.logic = Logic.LogicPrepareStopSign(start_state='ObservingStopYieldSign', BM_state=BM_state)
        self.cur_state = 'ObservingStopYieldSign'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing Stop Sign")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'ObservingStopYieldSign'
        self.logic.reset_state(state='ObservingStopYieldSign')
        self.FSM_situation.cur_state = self.FSM_situation.states['ObservingStopYieldSign']
        self.FSM_state.situation_state_static = None


class StopSign(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'Stopping': Stopping(BM_state),
                'WaitingForStopYieldSignClearance': WaitingForStopYieldSignClearance(BM_state),
                'ContinueDriving': ContinueDriving(BM_state)},
            possible_transitions={
                'toStopping': Transition('Stopping'),
                'toWaitingForStopYieldSignClearance': Transition('WaitingForStopYieldSignClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state='Stopping')
        self.logic = Logic.LogicStopSign(start_state='Stopping', BM_state=BM_state)
        self.cur_state = 'Stopping'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Stop Sign")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'Stopping'
        self.logic.reset_state(state='Stopping')
        self.FSM_situation.cur_state = self.FSM_situation.states['Stopping']
        self.FSM_state.situation_state_static = None


class PrepareYieldSign(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.FSM_situation = SimpleFSM(
            possible_states={
                'ObservingStopYieldSign': ObservingStopYieldSign(BM_state),
                'SlowingDown': SlowingDown(BM_state)},
            possible_transitions={
                'toObservingStopYieldSign': Transition('ObservingStopYieldSign'),
                'toSlowingDown': Transition('SlowingDown')},
            default_state='ObservingStopYieldSign')
        self.logic = Logic.LogicPrepareYieldSign(start_state='ObservingStopYieldSign', BM_state=BM_state)
        self.cur_state = 'ObservingStopYieldSign'

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Preparing Yield Sign")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        self.cur_state = 'ObservingStopYieldSign'
        self.logic.reset_state(state='ObservingStopYieldSign')
        self.FSM_situation.cur_state = self.FSM_situation.states['ObservingStopYieldSign']
        self.FSM_state.situation_state_static = None


class YieldSign(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

        if self.FSM_state.stop_yield_sign_clear == True:
            self.init_state = 'StopYieldSignClear'
        else:
            self.init_state = 'Stopping'

        self.FSM_situation = SimpleFSM(
            possible_states={
                'StopYieldSignClear': StopYieldSignClear(BM_state),
                'Stopping': Stopping(BM_state),
                'WaitingForStopYieldSignClearance': WaitingForStopYieldSignClearance(BM_state),
                'ContinueDriving': ContinueDriving(BM_state)},
            possible_transitions={
                'toStopYieldSignClear': Transition('StopYieldSignClear'),
                'toStopping': Transition('Stopping'),
                'toWaitingForStopYieldSignClearance': Transition('WaitingForStopYieldSignClearance'),
                'toContinueDriving': Transition('ContinueDriving')},
            default_state=self.init_state)
        self.logic = Logic.LogicYieldSign(start_state=self.init_state, BM_state=BM_state)
        self.cur_state = self.init_state

    def execute(self):
        bhv_logger.debug("FSM Behavior State: Yield Sign")
        # FSM
        transition, self.cur_state = self.logic.execute(self.cur_state)
        if transition is not None:
            self.FSM_situation.transition(transition)
        self.FSM_state.situation_state_static = self.cur_state

        # actions
        self.FSM_situation.execute()

    def reset_state(self):
        if self.FSM_state.stop_yield_sign_clear == True:
            self.init_state = 'StopYieldSignClear'
        else:
            self.init_state = 'Stopping'

        self.cur_state = self.init_state
        self.logic.reset_state(state=self.init_state)
        self.FSM_situation.cur_state = self.FSM_situation.states[self.init_state]
        self.FSM_state.situation_state_static = None


####################################################################################################
#                                         Situation States                                         #
####################################################################################################


class IdentifyTargetLaneAndVehiclesOnTargetLane(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Identifying target lane and vehicles on target lane")
        # identify target lanelet
        if self.FSM_state.behavior_state_static == 'PrepareLaneMerge':
            lane_merge_lanelet_id = None
            # search for next lane merge and get the id of the goal lanelet
            for static_goal in self.BM_state.PP_state.static_route_plan:
                if static_goal.goal_type == 'LaneMerge' and static_goal.end_s >= self.BM_state.ref_position_s:
                    lane_merge_lanelet_id = static_goal.goal_lanelet_id
            if lane_merge_lanelet_id is None:
                bhv_logger.warning("No lane merge found ahead of current position.")

            self.FSM_state.lane_change_target_lanelet_id = lane_merge_lanelet_id
        elif self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeRight':
            self.FSM_state.lane_change_target_lanelet_id = self.BM_state.current_lanelet.adj_right
            self.FSM_state.lane_change_target_lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                self.FSM_state.lane_change_target_lanelet_id)
        elif self.FSM_state.behavior_state_dynamic == 'PrepareLaneChangeLeft':
            self.FSM_state.lane_change_target_lanelet_id = self.BM_state.current_lanelet.adj_left
            self.FSM_state.lane_change_target_lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                self.FSM_state.lane_change_target_lanelet_id)

        # identify relevant objects on target lane
        self.FSM_state.obstacles_on_target_lanelet = hf.get_predicted_obstacles_on_lanelet(
            predictions=self.BM_state.predictions,
            lanelet_network=self.BM_state.scenario.lanelet_network,
            lanelet_id=self.FSM_state.lane_change_target_lanelet_id,
            search_point=self.BM_state.ego_state.position,
            search_distance=self.BM_state.VP_state.speed_limit_default * 2)


class IdentifySpeedOfObstaclesOnTargetLane(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Identifying speed of obstacles on target lane for overtake
        bhv_logger.debug("FMS Situation State: Identifying speed of obstacles on target lane for overtake")


class IdentifyFreeSpaceOnTargetLaneForLaneChange(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Identifying free space on target lane")
        time1 = time.time()
        # if target lane is empty withing search radius
        if not self.FSM_state.obstacles_on_target_lanelet:
            self.FSM_state.free_space_on_target_lanelet = True
            bhv_logger.debug('\n*Free Space: Target Lane Empty\n')
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
                            bhv_logger.error('\n*Free Space: obstacle out of projection domain. obstacle position: \n',
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
                                bhv_logger.debug('\n*Free Space: obstacle behind ego vehicle')
                                bhv_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                bhv_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                bhv_logger.debug('*Free Space: obstacle is not further behind than half the velocity distance\n')
                        # obstacle in front of ego vehicle
                        else:
                            # if not obstacle is further away than half the velocity distance.
                            if not (obstacle_pos_s > (self.BM_state.ref_position_s + ego_position_offset +
                                                      self.BM_state.vehicle_params.length +
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False
                                bhv_logger.debug('\n*Free Space: obstacle in front of ego vehicle')
                                bhv_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                bhv_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                bhv_logger.debug('*Free Space: obstacle is not further away than half the velocity distance\n')
                if ego_position_offset == 0:
                    if free_space_on_target_lanelet:
                        self.FSM_state.free_space_on_target_lanelet = True
                        break
                else:
                    if free_space_on_target_lanelet:
                        bhv_logger.debug('\n*Free Space: free space detected with offset: \n', ego_position_offset)
                        self.FSM_state.free_space_offset = ego_position_offset
                        self.FSM_state.change_velocity_for_lane_change = True
                        break
        if self.FSM_state.free_space_on_target_lanelet:
            bhv_logger.debug('\n*Free Space: free space detected\n')
        time2 = time.time()
        bhv_logger.debug('Free Space calc time: ', time2 - time1)


class IdentifyFreeSpaceOnTargetLaneForLaneMerge(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Identifying free space on target lane")
        time1 = time.time()
        # if target lane is empty withing search radius
        if not self.FSM_state.obstacles_on_target_lanelet:
            self.FSM_state.free_space_on_target_lanelet = True
            bhv_logger.debug('\n*Free Space: Target Lane Empty\n')
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
                        bhv_logger.debug('\n*Free Space: Ego position offset: \n', ego_position_offset)
                    for obstacle_id in self.FSM_state.obstacles_on_target_lanelet:
                        obstacle = self.FSM_state.obstacles_on_target_lanelet.get(obstacle_id)
                        obstacle_position = obstacle.get('pos_list')[0]
                        obstacle_velocity = obstacle.get('v_list')[0]
                        try:
                            obstacle_pos_s = self.BM_state.PP_state.cl_ref_coordinate_system.\
                                convert_to_curvilinear_coords(x=obstacle_position[0], y=obstacle_position[1])[0]
                        except:
                            bhv_logger.error('\n*Free Space: obstacle out of projection domain. obstacle position: \n',
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
                                bhv_logger.debug('\n*Free Space: obstacle behind ego vehicle')
                                bhv_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                bhv_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                bhv_logger.debug('*Free Space: obstacle is not further behind than half the velocity distance\n')
                        # obstacle in front of ego vehicle
                        else:
                            # if not obstacle is further away than half the velocity distance.
                            if not (obstacle_pos_s > (self.BM_state.ref_position_s + ego_position_offset +
                                                      self.BM_state.vehicle_params.length +
                                                      self.BM_state.ego_state.velocity / 2 * risk_factor)):
                                free_space_on_target_lanelet = False
                                bhv_logger.debug('\n*Free Space: obstacle in front of ego vehicle')
                                bhv_logger.debug('*Free Space: obstacle position: ', obstacle_position)
                                bhv_logger.debug('*Free Space: obstacle velocity: ', obstacle_velocity)
                                bhv_logger.debug('*Free Space: obstacle is not further away than half the velocity distance\n')
                if ego_position_offset == 0:
                    if free_space_on_target_lanelet:
                        self.FSM_state.free_space_on_target_lanelet = True
                        break
                else:
                    if free_space_on_target_lanelet:
                        bhv_logger.debug('\n*Free Space: free space detected with offset: \n', ego_position_offset)
                        self.FSM_state.free_space_offset = ego_position_offset
                        self.FSM_state.change_velocity_for_lane_change = True
                        break
        if self.FSM_state.free_space_on_target_lanelet:
            bhv_logger.debug('\n*Free Space: free space detected\n')
        time2 = time.time()
        bhv_logger.debug('Free Space calc time: ', time2 - time1)


class PreparationsDone(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Preparations Done")


class InitiateLaneChange(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Initiating lane change")


class EgoVehicleBetweenTwoLanes(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Vehicle crossing lanes")


class LaneChangeComplete(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Lane change complete")


class BehaviorStateComplete(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Behavior State complete")


class EstimateMergingLaneLengthAndEmergencyStopPoint(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Estimating merging lane length and setting emergency stop point")


class InitiateLaneMerge(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Initiating lane merge")


class InitiateRoadExit(State):
    def execute(self):
        bhv_logger.debug("FSM Situation State: Preparations Done")


class ObservingIntersection(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Observing Intersection")


class IntersectionClear(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Intersection Clear")


class WaitingForIntersectionClearance(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Waiting for Intersection Clearance")


class TurnClear(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Turn Clear
        bhv_logger.debug("FSM Situation State: Turn Clear")


class WaitingForTurnClearance(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Waiting for Turn Clearance
        bhv_logger.debug("FSM Situation State: Waiting for Turn Clearance")


class AbortOvertake(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Aborting Overtake
        bhv_logger.debug("FSM Situation State: Aborting Overtake")


class Overtaking(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Overtaking
        bhv_logger.debug("FSM Situation State: Overtaking")


class OvertakeComplete(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        # TODO: implementation exec Overtaking completed
        bhv_logger.debug("FSM Situation State: Overtaking completed")


class ObservingTrafficLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Observing Traffic Light")


class SlowingDown(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.VP_state = BM_state.VP_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Slowing Down Car")
        self.FSM_state.slowing_car_for_traffic_light = True
        self.VP_state.dist_to_tlna = self.BM_state.current_static_goal.stop_point_s - self.BM_state.ref_position_s - \
                                   self.BM_state.vehicle_params.length
        self.VP_state.stop_distance = self.VP_state.dist_to_tl

        # check for other stopping cars in front of vehicle
        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.dist_preceding_veh - self.BM_state.vehicle_params.length - \
                    self.VP_state.closest_preceding_vehicle.obstacle_shape.length \
                    <= self.VP_state.dist_to_tl:
                self.VP_state.stop_distance = self.VP_state.dist_preceding_veh - \
                                              self.BM_state.vehicle_params.length - \
                                              self.VP_state.closest_preceding_vehicle.obstacle_shape.length
        bhv_logger.debug("FSM distance to stopping line is: ", self.VP_state.dist_to_tl)


class GreenLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Green Light")
        self.FSM_state.slowing_car_for_traffic_light = False


class Stopping(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.VP_state = BM_state.VP_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Stopping the Car")
        self.FSM_state.slowing_car_for_traffic_light = True
        self.VP_state.dist_to_tl = self.BM_state.current_static_goal.stop_point_s - self.BM_state.ref_position_s - \
                                   self.BM_state.vehicle_params.length
        self.VP_state.stop_distance = self.VP_state.dist_to_tl

        # check for other stopping cars in front of vehicle
        if self.VP_state.dist_preceding_veh is not None:
            if self.VP_state.dist_preceding_veh - self.BM_state.vehicle_params.length - \
                    self.VP_state.closest_preceding_vehicle.obstacle_shape.length \
                    <= self.VP_state.dist_to_tl:
                self.VP_state.stop_distance = self.VP_state.dist_preceding_veh - \
                                              self.BM_state.vehicle_params.length - \
                                              self.VP_state.closest_preceding_vehicle.obstacle_shape.length

        bhv_logger.debug("FSM distance to stopping line is: ", self.VP_state.dist_to_tl)


class WaitingForGreenLight(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Waiting for Green Light")


class ContinueDriving(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Continue driving")
        self.FSM_state.slowing_car_for_traffic_light = False
        self.FSM_state.waiting_for_green_light = False


class ObservingCrosswalk(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Observing Crosswalk")


class CrosswalkClear(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Crosswalk Clear")


class WaitingForCrosswalkClearance(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Waiting for Crosswalk Clearance")


class ObservingStopYieldSign(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Observing Target Lane at Stop or Yield Sign")


class StopYieldSignClear(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Stop or Yield Clear")


class WaitingForStopYieldSignClearance(State):
    def __init__(self, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state

    def execute(self):
        bhv_logger.debug("FSM Situation State: Waiting Target Lane Clearance at Stop or Yield Sign")


####################################################################################################
#                                         Transition State                                         #
####################################################################################################


class Transition(object):
    """transition base class"""

    def __init__(self, to_state):
        self.to_state = to_state

    def execute(self):
        bhv_logger.info("FSM Transitioning to: " + str(self.to_state))
