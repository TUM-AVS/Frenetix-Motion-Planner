__author__ = "Maximilian Geisslinger, Tobias Markus"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
This is the main script of the predictions software.
It contains the WaleNet class which should be used to deploy the prediction.

wale_net_lite is based on wale-net, but only contains ONNX-based inference.
Support for online learning has been removed.
"""


# Standard imports
import math
import copy
import logging

import importlib.resources

from typing import List, Any

# Third party imports
import numpy as np

# Custom imports
from .geometry import transform_trajectories, transform_back
from .preprocessing import (
    generate_self_rendered_sc_img,
    generate_nbr_array,
)

import onnxruntime


class Prediction(object):
    """General prediction class.
    All prediction methods should inherit from this class.
    """

    def __init__(self, scenario, common_args=None, multiprocessing=False):
        """Initialize prediction

        Args:
            scenario ([commonroad.Scenario]) -- [CommonRoad scenario object]
            common_args ([type], optional): [Arguments for the predction]. Defaults to None.
            multiprocessing (bool, optional): [True if multiprocessing should be used]. Defaults to False.
        """
        assert not multiprocessing
        self.scenario = scenario

        self.online_args = dict()

    def step(self, time_step, obstacle_id_list=None, scenario=None):
        """Step function that executes the main function of the prediction.

        Arguments:
            scenario ([commonroad.Scenario]) -- [CommonRoad scenario object]
            time_step {[int]} -- [time_step of CommonRoad scenario]
            obstacle_id_list {[list]} -- [list of obstacle ids that should be predicted]

        Keyword Arguments:
            multiprocessing {bool} -- [True if every predicted object should start an own process] (default: {False})

        Returns:
            prediction_result [dict] -- [dictionary with obstacle ids as keys and x,y position and covariance matrix as values]
        """
        # Update scenario
        if scenario:
            self.scenario = scenario

        # Take all obstacle ids if non are given
        if obstacle_id_list is None:
            obstacle_id_list = list(self.scenario._dynamic_obstacles.keys())

        self.prediction_result = {}
        self.time_step = time_step

        # Check if all obstacles are still in the scenario
        obstacle_id_list = self._obstacles_in_scenario(time_step, obstacle_id_list)

        obstacle_id_list.sort(
            key=lambda x: len(
                self.scenario._dynamic_obstacles[x].prediction.trajectory.state_list
            ),
            reverse=True,
        )

        self.obstacle_id_list = obstacle_id_list

        # we can only process multiple vehicles trough the network if online learning is not used
        # self.step_multi(obstacle_id_list)
        
        # NOTE: step_multi works fine for me, however torch warns against using ONNX models using GRU
        # with a batch size other than 1 unless the initial state is supplied to the model
        # (which is not the case here).
        for obstacle_id in obstacle_id_list:
            self.step_single(obstacle_id)

        return self.prediction_result

    def step_single(self, obstacle_id):
        """Main function for the prediction of a single object.

        Arguments:
            obstacle_id {[int]} -- [CommonRoad obstacle ID]

        Returns:
            prediction_result [dict] -- [result of prediction in a dict]
        """
        fut_pos = self._predict_GT(self.time_step, obstacle_id)
        fut_cov = np.zeros((50, 2, 2))

        self.prediction_result[obstacle_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

        return [obstacle_id, {"pos_list": fut_pos, "cov_list": fut_cov}]

    def step_multi(self, _obstacle_id_list):
        raise NotImplementedError

    def get_positions(self):
        """Returns the position list of the prediction result

        Returns:
            [list]: [List of predicted postions]
        """
        self.pos_list = [
            list(self.prediction_result.values())[i]["pos_list"]
            for i in range(len(self.prediction_result))
        ]
        return self.pos_list

    def get_covariance(self):
        """Returns a list of covariant matrices of the last prediction result

        Returns:
            [list]: [List of covariance matrices]
        """
        self.cov_list = [
            list(self.prediction_result.values())[i]["cov_list"]
            for i in range(len(self.prediction_result))
        ]
        return self.cov_list

    def _predict_GT(self, time_step, obstacle_id, pred_horizon=50):
        """Returns the ground truth from the scenario as a prediction

        Args:
            time_step ([int]): [Current time step in CommonRoad scenario]
            obstacle_id ([int]): [Obstacle ID that should be predicted with ground truth]
            pred_horizon (int, optional): [Number of timesteps that should be predicted]. Defaults to 50.

        Returns:
            [np.array]: [Positions of ground truth predictions]
        """
        fut_GT = [
            self.scenario._dynamic_obstacles[obstacle_id]
            .prediction.trajectory.state_list[i]
            .position
            for i in range(time_step + 1, time_step + pred_horizon + 1)
            if len(
                self.scenario._dynamic_obstacles[
                    obstacle_id
                ].prediction.trajectory.state_list
            )
            > i
        ]

        return np.array(fut_GT)

    def _obstacles_in_scenario(self, time_step, obstacle_id_list):
        obstacle_id_list_new = [
            obst
            for obst in obstacle_id_list
            if self.scenario._dynamic_obstacles[obst].prediction.final_time_step
            > time_step and (self.scenario._dynamic_obstacles[obst].prediction.initial_time_step - 1)  # -1 because its the first time step of prediction
            <= time_step
        ]
        return obstacle_id_list_new


class WaleNet(Prediction):
    """Class for LSTM prediction method.

    Arguments:
        Prediction {[class]} -- [General Prediction class]
    """

    @staticmethod
    def execution_providers() -> List[str]:
        available_providers = onnxruntime.get_available_providers()
        logging.info(f"providers: {available_providers}")

        providers = []

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")

        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
        else:
            logging.warn("CPUExecutionProvider not available in onnxruntime")

        return providers

    def __init__(self, scenario):
        super().__init__(scenario)

        # Fixed network parameters
        self.grid_size = [13, 3]
        self.watch_radius = 64

        model_path = "wale-net.onnx"

        pkg_files = importlib.resources.files(__package__)
        model_file = pkg_files.joinpath(model_path)
        model = model_file.read_bytes()

        providers = self.execution_providers()

        self.inf_session = onnxruntime.InferenceSession(model, providers=providers)
        inputs = self.inf_session.get_inputs()
        
        input_dict = {}
        for input in inputs:
            input_dict[input.name] = input

        self.in_length = input_dict["hist"].shape[0]

        self.rotation_dict: dict[int, Any] = {}
        self.translation_dict: dict[int, Any] = {}

    def step_single(self, obstacle_id):
        """Main function for the prediction of a single object.

        Arguments:
            obstacle_id {[int]} -- [CommonRoad obstacle ID]

        Returns:
            prediction_result [dict] -- [result of prediction in a dict]
        """

        self.obstacle_id = obstacle_id

        # Preprocessing
        hist, nbrs, sc_img = self._preprocessing(obstacle_id)

        # Neural Network
        self.fut_pred = self._predict(hist, nbrs, sc_img)

        # Post processing
        fut_pos, fut_cov = self._postprocessing(self.fut_pred, obstacle_id)

        self.prediction_result[obstacle_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

        return [obstacle_id, {"pos_list": fut_pos, "cov_list": fut_cov}]

    def step_multi(self, obstacle_id_list):
        """This function makes multiple predictions at the same time based on a list of obstacles.
        This should reduce computational effort.

        Args:
            obstacle_id_list ([list]): [List of obstacle IDs to be predicted]
        """

        # Create tensors
        hist_batch = np.zeros(
            [self.in_length, len(obstacle_id_list), 2]
        )
        no_nbrs_cells = (
            self.grid_size[0]
            * self.grid_size[1]
        )

        nbrs_batch = np.zeros(
            [
                self.in_length,
                no_nbrs_cells * len(obstacle_id_list),
                2,
            ]
        )

        sc_img_batch = np.zeros([len(obstacle_id_list), 1, 256, 256])

        for obst_num, obst_id in enumerate(obstacle_id_list):

            hist, nbrs, sc_img = self._preprocessing(obst_id)  # results[obst_num][0]

            hist_batch[:, obst_num, :] = hist[:, 0, :]
            nbrs_batch[
                :, (obst_num * no_nbrs_cells) : ((obst_num + 1) * no_nbrs_cells), :
            ] = nbrs
            sc_img_batch[obst_num, :, :, :] = sc_img

        # Neural Network
        self.fut_pred = self._predict(hist_batch, nbrs_batch, sc_img_batch)

        # Post Processing
        for obst_num, obst_id in enumerate(obstacle_id_list):
            fut_pred = self.fut_pred[:, obst_num, :]

            fut_pos, fut_cov = self._postprocessing(
                np.expand_dims(fut_pred, axis=1), obst_id
            )

            self.prediction_result[obst_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

    def _predict(self, hist: np.ndarray, nbrs: np.ndarray, sc_img: np.ndarray) -> np.ndarray:
        """[Processing trough the neural network]

        Args:
            hist ([torch.Tensor]): [Past positions of the vehicle being predicted. Shape: [in_length, batch_size, 2]]
            nbrs ([torch.Tensor]): [Neighbor array of the vehicle being predicted. Shape: [in_length, grid_size * batch_size, 2]]
            sc_img ([torch.Tensor]): [Scene image for the prediction. Shape: [batch_size, 1, 256, 256]]

        Returns:
            [torch.Tensor]: [Network output. Shape: [out_length, batch_size, 5]]
        """

        hist = hist.astype(np.float32)
        nbrs = nbrs.astype(np.float32)
        sc_img = sc_img.astype(np.float32)

        hist_input = onnxruntime.OrtValue.ortvalue_from_numpy(hist)
        nbrs_input = onnxruntime.OrtValue.ortvalue_from_numpy(nbrs)
        sc_img_input = onnxruntime.OrtValue.ortvalue_from_numpy(sc_img)

        output_names = [ "predictions" ]

        results = self.inf_session.run_with_ort_values(output_names, {
            "hist": hist_input,
            "nbrs": nbrs_input,
            "sc_img": sc_img_input,
        })
        assert len(results) == 1
        
        fut_pred = results[0].numpy()

        return fut_pred

    def _postprocessing(self, fut_pred, obstacle_id):
        """Transforming the neural network output to a prediction format in world coordinates

        Args:
            fut_pred ([torch.Tensor]): [Network output. Shape: [50, batch_size, 5]]
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]

        Returns:
            [tuple]: [Storing fut_pos, fut_cov in real world coordinates]
        """
        # avoid changing fut_pred
        fut_pred_copy = copy.deepcopy(fut_pred)
        fut_pred_copy = np.squeeze(
            fut_pred_copy, 1
        )  # use batch size axes for list axes in transform function
        fut_pred_trans = transform_back(
            fut_pred_copy,
            self.translation_dict[obstacle_id],
            self.rotation_dict[obstacle_id],
        )

        return fut_pred_trans

    def _preprocessing(self, obstacle_id, time_step=None):
        """Prepare the input for the PredictionNet

        Args:
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]

        Returns:
            [list]: [hist, nbrs, sc_img as inputs for the neural network. See _predict for further Information]
        """
        logging.info("preprocessing")

        if time_step is None:
            time_step = self.time_step

        traj_state_list = self.scenario._dynamic_obstacles[
            obstacle_id
        ].prediction.trajectory.state_list

        # Get initial time step
        initial_time_step = self.scenario._dynamic_obstacles[obstacle_id].initial_state.time_step

        # Generate history
        hist = []
        for i in reversed(range(self.in_length)):
            # State list starts with the initial time step, which is not necessarily = 0!
            if time_step - (i + initial_time_step) >= 0:
                hist.append(traj_state_list[time_step - (i + initial_time_step)].position)
            else:
                hist.append([np.nan, np.nan])

        translation = hist[-1]
        rotation = (
            self.scenario._dynamic_obstacles[obstacle_id]
            .prediction.trajectory.state_list[time_step-initial_time_step]
            .orientation
        )

        # Adapt rotation
        rotation -= math.pi / 2

        self.translation_dict[obstacle_id] = translation
        self.rotation_dict[obstacle_id] = rotation

        hist = transform_trajectories([hist], translation, rotation)[0]

        # Generate neighbor array
        traj_list = [
            [
                self.scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .position
                for j in range(
                    0,
                    len(
                        self.scenario.dynamic_obstacles[
                            i
                        ].prediction.trajectory.state_list
                    ),
                )
            ]
            for i in range(0, len(self.scenario.dynamic_obstacles))
        ]
        initial_time_step_list = [self.scenario.dynamic_obstacles[i].initial_state.time_step for i in range(0, len(self.scenario.dynamic_obstacles))]
        trans_traj_list = transform_trajectories(traj_list, translation, rotation)
        nbrs, _, _, _ = generate_nbr_array(
            trans_traj_list, time_step, pp=self.in_length, initial_time_step_list=initial_time_step_list
        )
        nbrs = nbrs.reshape(nbrs.shape[0] * nbrs.shape[1], nbrs.shape[2], nbrs.shape[3])
        nbrs = np.swapaxes(nbrs, 0, 1)

        sc_img = generate_self_rendered_sc_img(
            self.watch_radius,
            self.scenario,
            translation,
            rotation,
        )

        # Create torch tensors and add batch dimension
        hist = np.expand_dims(hist, axis=1)
        sc_img = np.expand_dims(sc_img, axis=0)
        sc_img = np.expand_dims(sc_img, axis=0)

        # All NaN to zeros
        hist = np.nan_to_num(hist)
        nbrs = np.nan_to_num(nbrs)

        return hist, nbrs, sc_img

