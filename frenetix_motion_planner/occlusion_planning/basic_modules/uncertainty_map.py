__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
This module contains the uncertainty map of the occlusion module which stores information about occluded areas and
their history.

Author: Korbinian Moller, TUM
Date: 17.04.2023
"""

# imports
import numpy as np
import logging
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as ohf

# get logger
msg_logger = logging.getLogger("Message_logger")


class OccUncertaintyMap:
    def __init__(self, occ_visible_area=None, occ_occluded_area=None):
        self.occ_visible_area = occ_visible_area
        self.occ_occluded_area = occ_occluded_area
        self._np_init_len = 3
        self._threshold_low = -100
        self._threshold_high = 20
        self.error = False
        self.map_detail = None
        self.map = None
        self.log = []

    def step(self):

        self.error = False
        points_visible, points_occluded = self._check_points(self.occ_visible_area.points,
                                                             self.occ_occluded_area.points)

        if self.map is None:
            self._initialize(points_visible, points_occluded)
        else:
            if points_visible is None:
                msg_logger.debug("No visible area available, Occlusion map will be increased by 1")
                self._increase_map()
            else:
                self._update(points_visible, points_occluded)

        self.log.append(self.map)

    def _check_points(self, points_visible, points_occluded):
        if points_occluded is None:
            points_occluded = np.empty([0, self._np_init_len])
        return points_visible, points_occluded

    def _increase_map(self):
        self.map_detail[:, 3] += 1
        # Limit updated map to upper threshold
        self.map_detail = ohf.np_replace_max_value_with_new_value(self.map_detail, 3, self._threshold_high,
                                                                  self._threshold_high)
        self.map_detail[:, 4] = 1 - self.map_detail[:, 3] / self._threshold_high
        self.map = self.map_detail[:, [1, 2, 3]]

    def _update(self, np_visible, np_occluded):
        # new visible area has higher priority than old area --> -100, if value is added it will
        # still be below zero --> visible
        np_visible = np.c_[
            np_visible, np.ones(np_visible.shape[0]) * self._threshold_low, np.zeros(np_visible.shape[0])]
        np_occluded = np.c_[np_occluded, np.ones(np_occluded.shape[0]), np.zeros(np_occluded.shape[0])]

        # create a new occlusion map --> it's needed to identify the area of interest
        occ_map_new = np.concatenate((np_visible, np_occluded), axis=0)

        # save hash values in variable --> needed for the coordinate comparison (np only supports comparison in 1d)
        hash_map = self.map_detail[:, 0]
        hash_map_new = occ_map_new[:, 0]

        # find hashes (representation of coordinates) and corresponding indices that exist in both arrays
        hash_both, idx_map_new, idx_map = np.intersect1d(hash_map_new, hash_map, return_indices=True)

        # get values at shared coordinates from OLD occlusion map
        common_values_old_map = self.map_detail[idx_map, :]

        # get values at shared coordinates from NEW occlusion map
        common_values_new_map = occ_map_new[idx_map_new, :]

        # add occlusion values (stored in column 3) from old map to new map
        common_values_new_map[:, 3] += common_values_old_map[:, 3]
        common_values_new_map_0 = ohf.np_replace_negative_with_zero(common_values_new_map, 3)

        # find hashes and the index of values that only exist in NEW map
        hash_only_new = np.setdiff1d(hash_map_new, hash_map)
        idx_only_occ_map_new = np.where(np.isin(occ_map_new[:, 0], hash_only_new))[0]

        # get coordinates and corresponding values that ONLY exist in NEW map (and replace negative values with 0)
        occ_map_new_only = occ_map_new[idx_only_occ_map_new, :]

        # replace occlusion values of new points with the highest possible number (occ_threshold)
        occ_map_new_only = ohf.np_replace_non_negative_with_value(occ_map_new_only, 3, self._threshold_high)

        # replace values smaller than 0 with 0 (--> visible at this time_step)
        occ_map_new_only_0 = ohf.np_replace_negative_with_zero(occ_map_new_only, 3)

        # Create updated map
        occ_map_updated = np.concatenate((common_values_new_map_0, occ_map_new_only_0), axis=0)

        # Limit updated map to upper threshold
        occ_map_updated = ohf.np_replace_max_value_with_new_value(occ_map_updated, 3,
                                                                  self._threshold_high, self._threshold_high)

        # Calculate relative occlusion value
        occ_map_updated[:, 4] = 1 - occ_map_updated[:, 3] / self._threshold_high

        self.map_detail = occ_map_updated
        self.map = self.map_detail[:, [1, 2, 3]]

    def _initialize(self, np_visible, np_occluded):
        # hash, x, y, absolute value from 0(visible) to threshold (occluded), relative value from 0(unknown) to 1
        # (visible)

        if np_visible is None:
            msg_logger.debug("No visible area available, Occlusion cannot be initialized!")
            self.error = True
            return

        np_visible = np.c_[np_visible, np.zeros(np_visible.shape[0]), np.ones(np_visible.shape[0])]
        np_occluded = np.c_[np_occluded, np.ones(np_occluded.shape[0]) * self._threshold_high,
                            np.zeros(np_occluded.shape[0])]
        # if no occlusion map exists, the initial occlusion map is ready after this step
        self.map_detail = np.concatenate((np_visible, np_occluded), axis=0)
        self.map = self.map_detail[:, [1, 2, 3]]
# eof
