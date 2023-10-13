__author__ = "Rainer Trauth, Gerald Würsching"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# standard imports
from typing import List, Union
import os

# third party
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import imageio.v3 as iio

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import KSState
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.mp_renderer import MPRenderer, DynamicObstacleParams, ShapeParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import MPDrawParams
from frenetix_motion_planner.utility.helper_functions import green_to_red_colormap
# commonroad_dc
from commonroad_dc import pycrcc

# commonroad-rp
from cr_scenario_handler.utils.configuration import Configuration


"""Visualization functions for the frenét planner."""
from wale_net_lite.visualization import draw_uncertain_predictions as draw_uncertain_predictions_wale


def visualize_scenario_and_pp(scenario: Scenario, planning_problem: PlanningProblem, cosy=None):
    """Visualizes scenario, planning problem and (optionally) the reference path"""
    plot_limits = None
    ref_path = None
    if cosy is not None:
        ref_path = cosy.reference
        x_min = np.min(ref_path[:, 0]) - 50
        x_max = np.max(ref_path[:, 0]) + 50
        y_min = np.min(ref_path[:, 1]) - 50
        y_max = np.max(ref_path[:, 1]) + 50
        plot_limits = [x_min, x_max, y_min, y_max]

    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    rnd.draw_params.time_begin = 0
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19,
                    linewidth=0.8, label='reference path')
        proj_domain_border = np.array(cosy.ccosy.projection_domain())
        rnd.ax.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], color="orange", linewidth=0.8)
    plt.show(block=True)


def visualize_collision_checker(scenario: Scenario, cc: pycrcc.CollisionChecker):
    """
    Visualizes the collision checker, i.e., all collision objects and, if applicable, the road boundary.
    :param scenario CommonRoad scenario object
    :param cc pycrcc.CollisionChecker object
    """
    rnd = MPRenderer(figsize=(20, 10))
    scenario.lanelet_network.draw(rnd)
    cc.draw(rnd)
    rnd.render(show=True)


def visualize_planner_at_timestep(scenario: Scenario, planning_problem: PlanningProblem, ego: DynamicObstacle,
                                  timestep: int, config: Configuration, log_path: str,
                                  traj_set=None, optimal_traj = None, ref_path: np.ndarray = None,
                                  rnd: MPRenderer = None, predictions: dict = None, plot_window: int = None,
                                  visible_area=None, cluster=None, occlusion_map=None):
    """
    Function to visualize planning result from the reactive planner for a given time step
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param ego: Ego vehicle as CommonRoad DynamicObstacle object
    :param pos: positions of planned trajectory [(nx2) np.ndarray]
    :param timestep: current time step of scenario to plot
    :param log_path: Log path where to save the plots
    :param config: Configuration object for plot/save settings
    :param traj_set: List of sampled trajectories (optional)
    :param optimal_traj: Optimal Trajectory selected
    :param ref_path: Reference path for planner as polyline [(nx2) np.ndarray] (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object; otherwise it
    :param predictions: Predictions used to run the planner
    :param plot_window: Window size to plot (optional)
    :param visible_area: Visible Area for plotting (optional)
    :param cluster: cluster number in this time step (optional)
    :param occlusion_map: Occlusion map information to plot (optional)
    will visualize on the existing object)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    if rnd is None:
        if plot_window > 0:
            rnd = MPRenderer(plot_limits=[-plot_window + ego.prediction.trajectory.state_list[0].position[0],
                                          plot_window + ego.prediction.trajectory.state_list[0].position[0],
                                          -plot_window + ego.prediction.trajectory.state_list[0].position[1],
                                          plot_window + ego.prediction.trajectory.state_list[0].position[1]],
                                          figsize=(10, 10))
        else:
            rnd = MPRenderer(figsize=(20, 10))

    # set ego vehicle draw params
    ego_params = DynamicObstacleParams()
    ego_params.time_begin = timestep
    ego_params.draw_icon = config.debug.draw_icons
    ego_params.show_label = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.debug.draw_icons
    obs_params.dynamic_obstacle.show_label = True
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    obs_params.static_obstacle.show_label = True
    obs_params.static_obstacle.occupancy.shape.facecolor = "#a30000"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

    # visualize scenario, planning problem, ego vehicle
    scenario.draw(rnd, draw_params=obs_params)
    planning_problem.draw(rnd)
    ego.draw(rnd, draw_params=ego_params)

    rnd.render()

    # visualize optimal trajectory
    rnd.ax.plot([i.position[0] for i in optimal_traj.state_list],
                [i.position[1] for i in optimal_traj.state_list],
                color='k', marker='x', markersize=1.5, zorder=21, linewidth=2.0, label='optimal trajectory')

    # draw visible sensor area
    if visible_area is not None:
        if visible_area.geom_type == "MultiPolygon":
            for geom in visible_area.geoms:
                rnd.ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
        elif visible_area.geom_type == "Polygon":
            rnd.ax.fill(*visible_area.exterior.xy, "g", alpha=0.2, zorder=10)
        else:
            for obj in visible_area.geoms:
                if obj.geom_type == "Polygon":
                    rnd.ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10)

    # draw occlusion map - first version
    if occlusion_map is not None:
        cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"], N=10)
        scatter = rnd.ax.scatter(occlusion_map[:, 1], occlusion_map[:, 2], c=occlusion_map[:, 4], cmap=cmap, zorder=25, s=5)
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        rnd.ax.legend(handles, labels, loc="upper right", title="Occlusion")

    # visualize sampled trajectory bundle
    if traj_set is not None:
        valid_traj = [obj for obj in traj_set if obj.valid is True and obj.feasible is True]
        invalid_traj = [obj for obj in traj_set if obj.valid is False or obj.feasible is False]
        norm = matplotlib.colors.Normalize(
            vmin=0,
            vmax=len(valid_traj),
            clip=True,
        )
        mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())
        step = int(len(invalid_traj) / 100) if int(len(invalid_traj) / 100) > 2 else 1
        for idx, val in enumerate(valid_traj):
            if not val._coll_detected:
                color = mapper.to_rgba(idx)
                plt.plot(val.cartesian.x, val.cartesian.y,
                         color=color, zorder=20, linewidth=1.0, alpha=1.0, picker=False)
            else:
                plt.plot(val.cartesian.x, val.cartesian.y,
                         color='cyan', zorder=20, linewidth=1.0, alpha=0.8, picker=False)
        for ival in range(0, len(invalid_traj), step):
            plt.plot(invalid_traj[ival].cartesian.x, invalid_traj[ival].cartesian.y,
                     color="#808080", zorder=19, linewidth=0.8, alpha=0.4, picker=False)

    # visualize predictions
    if predictions is not None:
        if config.prediction.mode == "walenet":
            draw_uncertain_predictions_wale(predictions, rnd.ax)

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    if cluster is not None:
        rnd.ax.text(traj_set[0].cartesian.x[0]+(plot_window + 5),
                    traj_set[0].cartesian.y[0]+(plot_window + 5), str(cluster), fontsize=40)

    # save as .png file
    if config.debug.save_plots:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if config.debug.gif:
            plt.axis('off')
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    # show plot
    if config.debug.show_plots:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def plot_final_trajectory(scenario: Scenario, planning_problem: PlanningProblem, state_list: List[KSState],
                          config: Configuration, log_path: str, ref_path: np.ndarray = None):
    """
    Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param state_list: List of trajectory States
    :param config: Configuration object for plot/save settings
    :param ref_path: Reference path as [(nx2) np.ndarray] (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.facecolor = '#E37222'
    occ_params.edgecolor = '#9C4100'
    occ_params.opacity = 1.0
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)
    # visualize planning problem
    planning_problem.draw(rnd)
    # visualize occupancies of trajectory
    for i in range(len(state_list)):
        state = state_list[i]
        occ_pos = Rectangle(length=config.vehicle.length, width=config.vehicle.width, center=state.position,
                            orientation=state.orientation)
        if i >= 1:
            occ_params.opacity = 0.3
            occ_params.zorder = 50
        occ_pos.draw(rnd, draw_params=occ_params)
    # render scenario and occupancies
    rnd.render()

    # visualize trajectory
    pos = np.asarray([state.position for state in state_list])
    rnd.ax.plot(pos[:, 0], pos[:, 1], color='k', marker='x', markersize=3.0, markeredgewidth=0.4, zorder=21,
                linewidth=0.8)

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    # save as .png file
    if config.debug.save_plots:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.svg", format='svg',
                    bbox_inches='tight')

    # show plot
    if config.debug.show_plots:
        matplotlib.use("TkAgg")
        # plt.show(block=False)
        # plt.pause(0.0001)


def make_gif(config: Configuration, scenario: Scenario, time_steps: Union[range, List[int]], log_path: str, duration: float = 0.1):
    """
    Function to create from single images of planning results at each time step
    Images are saved in output path specified in config.general.path_output
    :param config Configuration object
    :param scenario CommonRoad scenario object
    :param time_steps list or range of time steps to create the GIF
    :param duration
    """
    if not config.debug.save_plots:
        # only create GIF when saving of plots is enabled
        print("...GIF not created: Enable config.debug.save_plots to generate GIF.")
        pass
    else:
        print("...Generating GIF")
        images = []
        filenames = []

        # directory, where single images are outputted (see visualize_planner_at_timestep())
        path_images = os.path.join(log_path, "plots")

        for step in time_steps:
            im_path = os.path.join(path_images, str(scenario.scenario_id) + "_{}.png".format(step))
            filenames.append(im_path)

        for filename in filenames:
            images.append(iio.imread(filename))

        iio.imwrite(os.path.join(log_path, str(scenario.scenario_id) + ".gif"),
                        images, duration=duration)
