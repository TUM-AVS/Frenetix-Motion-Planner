__author__ = "Luca Troncone, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"


"""Logic Modules for the FSM Machines"""

from random import randint
from commonroad.scenario.lanelet import LineMarking
import logging

# get logger
msg_logger = logging.getLogger("Message_logger")


class LogicStreetSetting:
    """logic module for Ego FSM."""
    def __init__(self, BM_state):
        self.street_setting = BM_state.street_setting
        self.cur_state = BM_state.street_setting
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for Ego FSM

        Returns:
             transition (string): state to transition to or None if no change is required
             current state (string): current street setting state
        """
        self.cur_state = cur_state
        self.transition = None

        if self.street_setting != cur_state:
            if self.street_setting == 'Highway':
                self.transition = 'toHighway'
                self.cur_state = 'Highway'
            elif self.street_setting == 'Country':
                self.transition = 'toCountry'
                self.cur_state = 'Country'
            elif self.street_setting == 'Urban':
                self.transition = 'toUrban'
                self.cur_state = 'Urban'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


'''Street Setting FSM Logic Modules'''


class LogicBehaviorStatic:
    """logic module for street setting state to determine static behavior states."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.static_route_plan = self.BM_state.PP_state.static_route_plan
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for state Highway

        Returns:
             transition (string): state to transition to or None if no change is required
             current state (string): new current state
        """
        self.cur_state = cur_state
        self.transition = None

        for static_goal in self.static_route_plan:
            if static_goal.start_s <= self.BM_state.nav_position_s < static_goal.end_s:
                self.BM_state.current_static_goal = static_goal
                if self.cur_state != static_goal.goal_type:
                    self.cur_state = static_goal.goal_type
                    self.transition = 'to' + static_goal.goal_type

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicHighwayDynamic:
    """logic module for state Highway to determine dynamic behavior states."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for state Highway

        Returns:
             transition (string): state to transition to or None if no change is required
             current state (string): new current state
        """
        self.cur_state = cur_state
        self.transition = None

        # no lane change
        if self.cur_state != 'NoLaneChanges':
            if self.FSM_state.no_auto_lane_change:
                self.transition = 'toNoLaneChanges'
                self.cur_state = 'NoLaneChanges'

        if self.cur_state == 'NoLaneChanges':
            if self.cur_state == 'NoLaneChange':
                if self.FSM_state.no_auto_lane_change:
                    self.transition = 'toDynamicDefault'
                    self.cur_state = 'DynamicDefault'

        # initiate lane change preparations
        if self.cur_state == 'DynamicDefault' and not self.FSM_state.no_auto_lane_change \
                and self.BM_state.time_step > 0:
            # lane change due to navigation
            if self.BM_state.nav_lane_changes_right > 0:
                if self.BM_state.current_lanelet.adj_right is not None:
                    if self.BM_state.current_lanelet.adj_right_same_direction:
                        if self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('solid') and \
                                self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('broad_solid'):
                            self.transition = 'toPrepareLaneChangeRight'
                            self.cur_state = 'PrepareLaneChangeRight'
            if self.BM_state.nav_lane_changes_left > 0:
                if self.BM_state.current_lanelet.adj_left is not None:
                    if self.BM_state.current_lanelet.adj_left_same_direction:
                        if self.BM_state.current_lanelet.line_marking_left_vertices != LineMarking('solid') and \
                                self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('broad_solid'):
                            self.transition = 'toPrepareLaneChangeLeft'
                            self.cur_state = 'PrepareLaneChangeLeft'
        # TODO: add lane change due to Lane Merge, Road Exit, slow preceding vehicle ....

        # abort lane change preparations
        if self.cur_state == 'PrepareLaneChangeRight' and self.BM_state.current_lanelet.adj_right is None:
            self.cur_state = 'DynamicDefault'
            self.transition = 'toDynamicDefault'
        if self.cur_state == 'PrepareLaneChangeLeft' and self.BM_state.current_lanelet.adj_left is None:
            self.cur_state = 'DynamicDefault'
            self.transition = 'toDynamicDefault'

        # initiate lane change
        if self.cur_state == 'PrepareLaneChangeRight' and self.FSM_state.lane_change_right_ok:
            self.transition = 'toLaneChangeRight'
            self.cur_state = 'LaneChangeRight'
            self.FSM_state.lane_change_right_ok = None
            self.FSM_state.do_lane_change = True
        if self.cur_state == 'PrepareLaneChangeLeft' and self.FSM_state.lane_change_left_ok:
            self.transition = 'toLaneChangeLeft'
            self.cur_state = 'LaneChangeLeft'
            self.FSM_state.lane_change_left_ok = None
            self.FSM_state.do_lane_change = True

        # lane change completed
        if self.cur_state == 'LaneChangeRight' and self.FSM_state.lane_change_right_done:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_right_done = None
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            if self.BM_state.nav_lane_changes_right > 0:
                self.BM_state.nav_lane_changes_right -= 1
        if self.cur_state == 'LaneChangeLeft' and self.FSM_state.lane_change_left_done:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_left_done = None
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            if self.BM_state.nav_lane_changes_left > 0:
                self.BM_state.nav_lane_changes_left -= 1

        # lane change preparations aborted
        if self.cur_state == 'PrepareLaneChangeRight' and self.FSM_state.lane_change_prep_right_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            self.FSM_state.lane_change_prep_right_abort = False
        if self.cur_state == 'PrepareLaneChangeLeft' and self.FSM_state.lane_change_prep_left_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            self.FSM_state.lane_change_prep_left_abort = False

        # lane change aborted
        if self.cur_state == 'LaneChangeRight' and self.FSM_state.lane_change_right_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_right_abort = False
            self.FSM_state.undo_lane_change = True
        if self.cur_state == 'LaneChangeLeft' and self.FSM_state.lane_change_left_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_left_abort = False
            self.FSM_state.undo_lane_change = True

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicCountryDynamic:
    """logic module for state Country to determine dynamic behavior states."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for state Country

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        # no lane change
        if self.cur_state != 'NoLaneChanges':
            if self.FSM_state.no_auto_lane_change:
                self.transition = 'toNoLaneChanges'
                self.cur_state = 'NoLaneChanges'

        if self.cur_state == 'NoLaneChanges':
            if not self.FSM_state.no_auto_lane_change:
                self.transition = 'toDynamicDefault'
                self.cur_state = 'DynamicDefault'
        # TODO: add overtaking

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicUrbanDynamic:
    """logic module for state Urban to determine dynamic behavior states."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for state Urban

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        # no lane change
        if self.cur_state != 'NoLaneChanges':
            if self.FSM_state.no_auto_lane_change:
                self.transition = 'toNoLaneChanges'
                self.cur_state = 'NoLaneChanges'

        if self.cur_state == 'NoLaneChanges':
            if not self.FSM_state.no_auto_lane_change:
                self.transition = 'toDynamicDefault'
                self.cur_state = 'DynamicDefault'

        # initiate lane change preparations
        if self.cur_state == 'DynamicDefault' and not self.FSM_state.no_auto_lane_change \
                and self.BM_state.time_step > 0:
            # lane change due to navigation
            if self.BM_state.nav_lane_changes_right > 0:
                if self.BM_state.current_lanelet.adj_right is not None:
                    if self.BM_state.current_lanelet.adj_right_same_direction:
                        if self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('broad_solid') and \
                                self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('broad_solid'):
                            self.transition = 'toPrepareLaneChangeRight'
                            self.cur_state = 'PrepareLaneChangeRight'
            if self.BM_state.nav_lane_changes_left > 0:
                if self.BM_state.current_lanelet.adj_left is not None:
                    if self.BM_state.current_lanelet.adj_left_same_direction:
                        if self.BM_state.current_lanelet.line_marking_left_vertices != LineMarking('solid') and \
                                self.BM_state.current_lanelet.line_marking_right_vertices != LineMarking('broad_solid'):
                            self.transition = 'toPrepareLaneChangeLeft'
                            self.cur_state = 'PrepareLaneChangeLeft'

            # TODO: add lane change due to Lane Merge, Road Exit, slow preceding vehicle ....

        # initiate lane change
        if self.cur_state == 'PrepareLaneChangeRight' and self.FSM_state.lane_change_right_ok:
            self.transition = 'toLaneChangeRight'
            self.cur_state = 'LaneChangeRight'
            self.FSM_state.do_lane_change = True
            self.FSM_state.lane_change_right_ok = None
        if self.cur_state == 'PrepareLaneChangeLeft' and self.FSM_state.lane_change_left_ok:
            self.transition = 'toLaneChangeLeft'
            self.cur_state = 'LaneChangeLeft'
            self.FSM_state.do_lane_change = True
            self.FSM_state.lane_change_left_ok = None

        # lane change completed
        if self.cur_state == 'LaneChangeRight' and self.FSM_state.lane_change_right_done:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            if self.BM_state.nav_lane_changes_right > 0:
                self.BM_state.nav_lane_changes_right -= 1
            self.FSM_state.lane_change_right_done = None
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
        if self.cur_state == 'LaneChangeLeft' and self.FSM_state.lane_change_left_done:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            if self.BM_state.nav_lane_changes_left > 0:
                self.BM_state.nav_lane_changes_left -= 1
            self.FSM_state.lane_change_left_done = None
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None

        # lane change preparations aborted
        if self.cur_state == 'PrepareLaneChangeRight' and self.FSM_state.lane_change_prep_right_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            self.FSM_state.lane_change_prep_right_abort = False
        if self.cur_state == 'PrepareLaneChangeLeft' and self.FSM_state.lane_change_prep_left_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_target_lanelet_id = None
            self.FSM_state.lane_change_target_lanelet = None
            self.FSM_state.lane_change_prep_left_abort = False

        # lane change aborted
        if self.cur_state == 'LaneChangeRight' and self.FSM_state.lane_change_right_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_right_abort = False
            self.FSM_state.undo_lane_change = True
        if self.cur_state == 'LaneChangeLeft' and self.FSM_state.lane_change_left_abort:
            self.transition = 'toDynamicDefault'
            self.cur_state = 'DynamicDefault'
            self.FSM_state.lane_change_left_abort = False
            self.FSM_state.undo_lane_change = True

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


'''Behavior FSM Logic Modules'''


class LogicPrepareLaneChangeLeft:
    """logic module for state PrepareLaneChangeLeft."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for PrepareLaneChangeLeft

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'IdentifyTargetLaneAndVehiclesOnTargetLane':
            if self.FSM_state.obstacles_on_target_lanelet is not None:
                self.transition = 'toIdentifyFreeSpaceOnTargetLaneForLaneChange'
                self.cur_state = 'IdentifyFreeSpaceOnTargetLaneForLaneChange'

        elif self.cur_state == 'IdentifyFreeSpaceOnTargetLaneForLaneChange':
            if self.FSM_state.situation_time_step_counter > 4 and not self.FSM_state.free_space_on_target_lanelet:
                self.transition = 'toIdentifyTargetLaneAndVehiclesOnTargetLane'
                self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
                self.FSM_state.situation_time_step_counter = 0

            elif self.FSM_state.free_space_on_target_lanelet:
                self.transition = 'toPreparationsDone'
                self.cur_state = 'PreparationsDone'
                self.FSM_state.free_space_offset = 0
                self.FSM_state.change_velocity_for_lane_change = False
                self.FSM_state.lane_change_left_ok = True

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicLaneChangeLeft:
    """logic module for state LaneChangeLeft."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for LaneChangeLeft

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        # FSM_state.lane_change_left_abort
        if self.cur_state == 'InitiateLaneChange' and self.FSM_state.initiated_lane_change:
            self.FSM_state.initiated_lane_change = None
            self.FSM_state.do_lane_change = False

        if self.cur_state == 'InitiateLaneChange' and self.FSM_state.situation_time_step_counter > 16:
            self.FSM_state.lane_change_left_abort = True
            msg_logger.debug("FSM Dynamic Situation State: Aborting Lane Change")

        if self.FSM_state.detected_lanelets is not None:
            if len(self.FSM_state.detected_lanelets) > 1 \
                    and self.FSM_state.lane_change_target_lanelet_id in self.FSM_state.detected_lanelets:
                self.transition = 'toEgoVehicleBetweenTwoLanes'
                self.cur_state = 'EgoVehicleBetweenTwoLanes'

        if self.FSM_state.detected_lanelets is not None:
            if self.cur_state == 'EgoVehicleBetweenTwoLanes' and len(self.FSM_state.detected_lanelets) == 1 and \
                    self.BM_state.current_lanelet_id == self.FSM_state.lane_change_target_lanelet_id:
                self.transition = 'toLaneChangeComplete'
                self.cur_state = 'LaneChangeComplete'
                self.FSM_state.lane_change_left_done = True
                # reset temporary variables
                self.FSM_state.obstacles_on_target_lanelet = None
                self.FSM_state.lane_change_target_lanelet_id = None
                self.FSM_state.lane_change_target_lanelet = None
                self.FSM_state.free_space_on_target_lanelet = None
                self.FSM_state.initiated_lane_change = None

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicPrepareLaneChangeRight:
    """logic module for state PrepareLaneChangeRight."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for PrepareLaneChangeRight

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'IdentifyTargetLaneAndVehiclesOnTargetLane':
            if self.FSM_state.obstacles_on_target_lanelet is not None:
                self.transition = 'toIdentifyFreeSpaceOnTargetLaneForLaneChange'
                self.cur_state = 'IdentifyFreeSpaceOnTargetLaneForLaneChange'

        elif self.cur_state == 'IdentifyFreeSpaceOnTargetLaneForLaneChange':
            if self.FSM_state.situation_time_step_counter > 4 and not self.FSM_state.free_space_on_target_lanelet:
                self.transition = 'toIdentifyTargetLaneAndVehiclesOnTargetLane'
                self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'
                self.FSM_state.situation_time_step_counter = 0

            elif self.FSM_state.free_space_on_target_lanelet:
                self.transition = 'toPreparationsDone'
                self.cur_state = 'PreparationsDone'
                self.FSM_state.free_space_offset = 0
                self.FSM_state.change_velocity_for_lane_change = False
                self.FSM_state.lane_change_right_ok = True

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicLaneChangeRight:
    """logic module for state LaneChangeRight."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for LaneChangeRight

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'InitiateLaneChange' and self.FSM_state.initiated_lane_change:
            self.FSM_state.initiated_lane_change = None
            self.FSM_state.do_lane_change = False

        if self.cur_state == 'InitiateLaneChange' and self.FSM_state.situation_time_step_counter > 16:
            self.FSM_state.lane_change_left_abort = True
            msg_logger.debug("FSM Dynamic Situation State: Aborting Lane Change")

        if self.FSM_state.detected_lanelets is not None:
            if len(self.FSM_state.detected_lanelets) > 1 \
                    and self.FSM_state.lane_change_target_lanelet_id in self.FSM_state.detected_lanelets:
                self.transition = 'toEgoVehicleBetweenTwoLanes'
                self.cur_state = 'EgoVehicleBetweenTwoLanes'

        if self.FSM_state.detected_lanelets is not None:
            if self.cur_state == 'EgoVehicleBetweenTwoLanes' and len(self.FSM_state.detected_lanelets) == 1 and \
                    self.BM_state.current_lanelet_id == self.FSM_state.lane_change_target_lanelet_id:
                self.transition = 'toLaneChangeComplete'
                self.cur_state = 'LaneChangeComplete'
                self.FSM_state.lane_change_right_done = True
                # reset temporary variables
                self.FSM_state.obstacles_on_target_lanelet = None
                self.FSM_state.lane_change_target_lanelet_id = None
                self.FSM_state.lane_change_target_lanelet = None
                self.FSM_state.free_space_on_target_lanelet = None
                self.FSM_state.initiated_lane_change = None

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicPrepareLaneMerge:
    """logic module for state PrepareLaneMerge."""
    def __init__(self, start_state):
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for PrepareLaneMerge

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'EstimateMergingLaneLengthAndEmergencyStopPoint':
            if randint(0, 2):
                self.transition = 'toIdentifyTargetLaneAndVehiclesOnTargetLane'
                self.cur_state = 'IdentifyTargetLaneAndVehiclesOnTargetLane'

        elif self.cur_state == 'IdentifyTargetLaneAndVehiclesOnTargetLane':
            if randint(0, 2):
                self.transition = 'toIdentifyFreeSpaceOnTargetLaneForLaneMerge'
                self.cur_state = 'IdentifyFreeSpaceOnTargetLaneForLaneMerge'

        elif self.cur_state == 'IdentifyFreeSpaceOnTargetLaneForLaneMerge':
            if randint(0, 2):
                self.transition = 'toPreparationsDone'
                self.cur_state = 'PreparationsDone'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicLaneMerge:
    """logic module for state LaneMerge."""
    def __init__(self, start_state):
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for LaneMerge

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'InitiateLaneMerge':
            if randint(0, 2):
                self.transition = 'toEgoVehicleBetweenTwoLanes'
                self.cur_state = 'EgoVehicleBetweenTwoLanes'

        elif self.cur_state == 'EgoVehicleBetweenTwoLanes':
            if randint(0, 2):
                self.transition = 'toBehaviorStateComplete'
                self.cur_state = 'BehaviorStateComplete'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicPrepareRoadExit:
    """logic module for state PrepareRoadExit."""
    def __init__(self, start_state):
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for PrepareRoadExit

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'IdentifyTargetLaneAndVehiclesOnTargetLane':
            if randint(0, 2):
                self.transition = 'toIdentifyFreeSpaceOnTargetLaneForLaneMerge'
                self.cur_state = 'IdentifyFreeSpaceOnTargetLaneForLaneMerge'

        elif self.cur_state == 'IdentifyFreeSpaceOnTargetLaneForLaneMerge':
            if randint(0, 2):
                self.transition = 'toPreparationsDone'
                self.cur_state = 'PreparationsDone'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicRoadExit:
    """logic module for state RoadExit."""
    def __init__(self, start_state):
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for RoadExit

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'InitiateRoadExit':
            if randint(0, 2):
                self.transition = 'toEgoVehicleBetweenTwoLanes'
                self.cur_state = 'EgoVehicleBetweenTwoLanes'

        elif self.cur_state == 'EgoVehicleBetweenTwoLanes':
            if randint(0, 2):
                self.transition = 'toBehaviorStateComplete'
                self.cur_state = 'BehaviorStateComplete'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicPrepareTrafficLight:
    """logic module for state PrepareTrafficLight."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.VP_state = BM_state.VP_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for PrepareTrafficLight

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'ObservingTrafficLight':
            if self.FSM_state.traffic_light_state != 'green':
                self.transition = 'toSlowingDown'
                self.cur_state = 'SlowingDown'

        elif self.cur_state == 'SlowingDown':
            if self.FSM_state.traffic_light_state == 'green' or self.FSM_state.traffic_light_state == 'redYellow':
                self.transition = 'toObservingTrafficLight'
                self.cur_state = 'ObservingTrafficLight'

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state


class LogicTrafficLight:
    """logic module for state TrafficLight."""
    def __init__(self, start_state, BM_state):
        self.BM_state = BM_state
        self.FSM_state = BM_state.FSM_state
        self.VP_state = BM_state.VP_state
        self.cur_state = start_state
        self.transition = None

    def execute(self, cur_state):
        """call function to execute logic module for TrafficLight

        Returns:
             string: state to transition to or None if no change is required
        """
        self.cur_state = cur_state
        self.transition = None

        if self.cur_state == 'GreenLight':
            if self.FSM_state.traffic_light_state != 'green':
                self.transition = 'toStopping'
                self.cur_state = 'Stopping'

        elif self.cur_state == 'Stopping':
            if self.FSM_state.traffic_light_state == 'green' or self.FSM_state.traffic_light_state == 'redYellow':
                self.transition = 'toGreenLight'
                self.cur_state = 'GreenLight'

            if self.BM_state.ego_state.velocity <= 0.5:
                self.transition = 'toWaitingForGreenLight'
                self.cur_state = 'WaitingForGreenLight'
                self.FSM_state.waiting_for_green_light = True

        elif self.cur_state == 'WaitingForGreenLight':
            if self.FSM_state.traffic_light_state == 'green' or self.FSM_state.traffic_light_state == 'redYellow':
                self.transition = 'toContinueDriving'
                self.cur_state = 'ContinueDriving'
                self.FSM_state.waiting_for_green_light = False

        return self.transition, self.cur_state

    def reset_state(self, state):
        self.cur_state = state
