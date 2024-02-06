__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os
from typing import List, Union, Dict

import matplotlib
from matplotlib import pyplot as plt
import imageio.v3 as iio
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import State, CustomState
from commonroad.planning.planning_problem import PlanningProblem
from cr_scenario_handler.utils.agent_status import AgentStatus
# commonroad_dc
from commonroad_dc import pycrcc

from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle

from cr_scenario_handler.utils.configuration import Configuration

from wale_net_lite.visualization import draw_uncertain_predictions

# Color palette for plotting
colors_spec = ["#e3af22", "#d6e322", "#96e322", "#55e322", "#22e32f",
               "#22e36f", "#22e3af", "#22d6e3", "#2296e3", "#2255e3",
               "#2f22e3", "#6f22e3", "#af22e3", "#e322d6", "#e32296"]
darkcolors = ["#9c0d00", "#8f9c00", "#5b9c00", "#279c00", "#009c0d",
              "#009c41", "#009c75", "#008f9c", "#005b9c", "#00279c",
              "#0d009c", "#41009c", "#75009c", "#9c008f", "#9c005b"]
lightcolors = ["#ffd569", "#f8ff69", "#c6ff69", "#94ff69", "#69ff70",
               "#69ffa3", "#69ffd5", "#69f8ff", "#69c6ff", "#6993ff",
               "#7069ff", "#a369ff", "#d569ff", "#ff69f8", "#ff69c5"]


def visualize_agent_at_timestep(scenario: Scenario, planning_problem: PlanningProblem, ego: DynamicObstacle,
                                  timestep: int, config, log_path: str,
                                  traj_set=None, optimal_traj=None, ref_path: np.ndarray = None,
                                  rnd: MPRenderer = None, predictions: dict = None, plot_window: int = None,
                                  visible_area=None, occlusion_map=None, save: bool = False, show: bool = False, gif: bool = False):
    """
    Function to visualize planning result from the reactive planner for a given time step
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param ego: Ego vehicle as CommonRoad DynamicObstacle object
    :param pos: positions of planned trajectory [(nx2) np.ndarray]
    :param timestep: current time step of scenario to plot
    :param log_path: Log path where to save the plots
    :param config_visu: Configuration object for plot/save settings
    :param traj_set: List of sampled trajectories (optional)
    :param optimal_traj: Optimal Trajectory selected
    :param ref_path: Reference path for planner as polyline [(nx2) np.ndarray] (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object; otherwise it
    :param predictions: Predictions used to run the planner
    :param plot_window: Window size to plot (optional)
    :param visible_area: Visible Area for plotting (optional)
    :param occlusion_map: Occlusion map information to plot (optional)
    will visualize on the existing object)
    :param save_path: Path to save plot as .png (optional)
    """
    # Only create renderer if not passed in
    if rnd is None:
        # Assuming ego.prediction.trajectory.state_list[0].position returns a constant value
        ego_start_pos = ego.prediction.trajectory.state_list[0].position
        if plot_window > 0:
            plot_limits = [-plot_window + ego_start_pos[0], plot_window + ego_start_pos[0],
                           -plot_window + ego_start_pos[1], plot_window + ego_start_pos[1]]
            rnd = MPRenderer(plot_limits=plot_limits, figsize=(10, 10))
        else:
            rnd = MPRenderer(figsize=(20, 10))

    # set ego vehicle draw params
    ego_params = DynamicObstacleParams()
    ego_params.time_begin = timestep
    ego_params.draw_icon = config.visualization.draw_icons
    ego_params.show_label = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.visualization.draw_icons
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
    if optimal_traj:
        optimal_traj_positions = np.array([(state.position[0], state.position[1]) for state in optimal_traj.state_list])
        rnd.ax.plot(optimal_traj_positions[:, 0], optimal_traj_positions[:, 1], 'kx-', markersize=1.5, zorder=21, linewidth=2.0)

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
        cmap = colors.LinearSegmentedColormap.from_list('rg', ["r", "y", "g"], N=10)
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
        step = int(len(invalid_traj) / 50) if int(len(invalid_traj) / 50) > 2 else 1
        for idx, val in enumerate(reversed(valid_traj)):
            if not val._coll_detected:
                color = mapper.to_rgba(len(valid_traj) - 1 - idx)
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
        predictions_copy = predictions
        if ego.obstacle_id in predictions.keys():
            predictions_copy = predictions.copy()
            # Remove the entry with key 60000 from the copy
            predictions_copy.pop(ego.obstacle_id, None)
        if config.prediction.mode == "walenet":
            draw_uncertain_predictions(predictions_copy, rnd.ax)

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    # save as .png file
    if config.visualization.save_plots or save or gif:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if config.visualization.save_gif or gif:
            plt.axis('off')
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    # show plot
    if config.visualization.show_plots or show:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def visualize_multiagent_scenario_at_timestep(scenario: Scenario,
                                              agent_list: List,
                                              timestep: int,
                                              config: Configuration, log_path: str,
                                              orig_pp_id: List=None,
                                              predictions: dict = None,
                                              rnd: MPRenderer = None,
                                              plot_window: int = None,
                                              save: bool = False, show: bool = False, gif: bool = False):
    """
    Function to visualize planning result from the reactive planner for a given time step
    for all agents in a multiagent simulation.
    Replaces visualize_visualize_planner_at_timestep from visualization.py

    :param scenario: CommonRoad scenario object containing no dummy obstacles
    :param agent_list: Dummy obstacles for all agents. Assumed to include the recorded path in the trajectory
    :param timestep: Time step of the scenario to plot
    :param config: Configuration object for plot/save settings
    :param log_path: Path to save the plot to (optional, depending on the config)
    :param predictions: Dictionary of all predictions (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object;
                otherwise it will visualize on the existing object)
    :param plot_window: Size of the margin of the plot, or minimum distance between the
                        plot border and all agents (optional)
    """

    # create renderer object (if no existing renderer is passed)
    if rnd is None:
        rnd = MPRenderer(figsize=(20, 10))

        if plot_window is not None:
            # focus on window around all agents
            left = np.inf
            right = - np.inf
            top = - np.inf
            bottom = np.inf
            for agent in agent_list:
                try:
                    ini_pos = agent.planning_problem.initial_state.position
                    if hasattr(agent.planning_problem.goal.state_list[0].position, "shapes"):
                        goal_pos = agent.planning_problem.goal.state_list[0].position.shapes[0].center
                    elif hasattr(agent.planning_problem.goal.state_list[0].position, "center"):
                        goal_pos = agent.planning_problem.goal.state_list[0].position.center
                    # else:
                    #     raise ValueError

                    left = min(left, goal_pos[0], ini_pos[0])
                    right = max(right, goal_pos[0], ini_pos[0])
                    top = max(top, goal_pos[1], ini_pos[1])
                    bottom = min(bottom, goal_pos[1], ini_pos[1])
                except AttributeError:
                    pass

            rnd.plot_limits = [-plot_window + left,
                               plot_window + right,
                               -plot_window + bottom,
                               plot_window + top]

    # Set obstacle parameters
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.visualization.draw_icons
    obs_params.dynamic_obstacle.show_label = config.visualization.show_labels
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    obs_params.static_obstacle.show_label = config.visualization.show_labels
    obs_params.static_obstacle.occupancy.shape.facecolor = "#A30000"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#756F61"

    # visualize scenario
    scenario.draw(rnd, draw_params=obs_params)

    # Visualize agents and planning problems
    # for i in range(len(agent_list)):
    for agent in agent_list:
        if agent.status != AgentStatus.RUNNING:
            continue
        # set ego vehicle draw params
        ego_params = DynamicObstacleParams()
        ego_params.time_begin = timestep
        ego_params.draw_icon = config.visualization.draw_icons
        ego_params.show_label = config.visualization.show_labels

        # Use standard colors for single-agent plots
        if len(agent_list) == 1:
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
            ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
        elif orig_pp_id:
            if agent.id in orig_pp_id:
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
                ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
            else:
                obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
                obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
        else:
            ego_params.vehicle_shape.occupancy.shape.facecolor = \
                colors_spec[agent.id % len(colors_spec)]
            ego_params.vehicle_shape.occupancy.shape.edgecolor = \
                darkcolors[agent.id % len(darkcolors)]
        ego_params.vehicle_shape.occupancy.shape.zorder = 50
        ego_params.vehicle_shape.occupancy.shape.opacity = 1

        # Visualize planning problem and agent
        # if multiple_agents:
        #     planning_problems.find_planning_problem_by_id(agent.obstacle_id).draw(rnd)
        # else:

        agent.vehicle_history[-1].draw(rnd, draw_params=ego_params)
        agent.planning_problem.draw(rnd)
    rnd.render()
    for agent in agent_list:
        if agent.status != AgentStatus.RUNNING:
            continue
        # visualize optimal trajectory
        history = [i for i in agent.vehicle_history if i.initial_state.time_step == timestep]
        if history:
            history = history[0]

            if history.prediction is not None and config.visualization.draw_calc_traj:
                rnd.ax.plot([i.position[0] for i in history.prediction.trajectory.state_list],
                            [i.position[1] for i in history.prediction.trajectory.state_list],
                            color='k' , marker='x', markersize=1.5, zorder=21, linewidth=2.0, label='optimal trajectory')

        if agent.all_trajectories is not None and agent.config_planner.debug.draw_traj_set:
            valid_traj = [obj for obj in agent.all_trajectories if obj.valid is True and obj.feasible is True]
            invalid_traj = [obj for obj in agent.all_trajectories if obj.valid is False or obj.feasible is False]
            norm = matplotlib.colors.Normalize(
                vmin=0,
                vmax=len(valid_traj),
                clip=True,
            )
            mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())
            # step_inv = int(len(invalid_traj) / 50) if int(len(invalid_traj) / 50) > 2 else 1
            step_val = int(len(valid_traj)/50)  if int(len(valid_traj) / 50) > 2 else 1
            step_inv=step_val
            for idx in range(0, len(valid_traj), step_val):
                val = valid_traj[idx]
                if not val._coll_detected:
                    color = mapper.to_rgba(idx)
                    color = lightcolors[idx % len(lightcolors)]
                    plt.plot(val.cartesian.x, val.cartesian.y,
                             color=color, zorder=20, linewidth=1.0, alpha=1.0, picker=False)
                else:
                    plt.plot(val.cartesian.x, val.cartesian.y,
                             color='cyan', zorder=20, linewidth=1.0, alpha=0.8, picker=False)
            for ival in range(0, len(invalid_traj), step_inv):
                plt.plot(invalid_traj[ival].cartesian.x, invalid_traj[ival].cartesian.y,
                         color="#808080", zorder=19, linewidth=0.8, alpha=0.4, picker=False)

        if agent.reference_path is not None and config.visualization.draw_reference_path:

            rnd.ax.plot(agent.reference_path[:, 0], agent.reference_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                        label='reference path')

    # visualize predictions
    if predictions is not None and config.visualization.draw_predictions:
        draw_uncertain_predictions(predictions, rnd.ax)

    if save or gif:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if gif:
            plt.axis('off')
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300, bbox_inches='tight',
                        pad_inches=0)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    if show:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def make_gif(scenario: Scenario, time_steps: Union[range, List[int]],
             log_path: str, duration: float = 0.1):
    """ Create an animated GIF from saved image files.

    Images are assumed to be saved as <log_path>/plots/<scenario_id>_<timestep>.png
    Does not check the plotting configuration in order to simplify independent
    configurations on agent and simulation level. This has to be done by the caller.

    :param scenario: CommonRoad scenario object.
    :param time_steps: List or range of time steps to include in the GIF
    :param log_path: Base path containing the plots folder with the input images
    :param duration: Duration of the individual frames (default: 0.1s)
    """

    print("Generating GIF")
    images = []
    filenames = []

    # directory, where single images are outputted (see visualize_planner_at_timestep())
    path_images = os.path.join(log_path, "plots")

    for step in time_steps:
        im_path = os.path.join(path_images, str(scenario.scenario_id) + f"_{step}.png")
        filenames.append(im_path)

    for filename in filenames:
        images.append(iio.imread(filename))

    iio.imwrite(os.path.join(log_path, str(scenario.scenario_id) + ".gif"),
                images, duration=duration)


def collision_vis(scenario: Scenario, ego_vehicle: DynamicObstacle, destination: str,
                  ego_harm: float, ego_type: ObstacleType, ego_v: float, ego_mass: float,
                  obs_harm: float, obs_type: ObstacleType, obs_v: float, obs_mass: float,
                  pdof: float, ego_angle: float, obs_angle: float,
                  time_step: int, modes: Dict, planning_problem: PlanningProblem = None,
                  global_path: np.ndarray = None, driven_traj: List[State] = None):
    """ Create a report for visualization of the collision and respective harm.

    Creates a collision report and saves the file in the destination folder
    in a subdirectory called "collisions".
    Replaces risk_assessment.visualization.collision_visualization.collision_vis
    to allow using the full ego obstacle.

    :param scenario: Considered Scenario.
    :param ego_vehicle: Ego obstacle, may contain both past and future trajectories.
    :param destination: Path to save output to.
    :param ego_type: Type of the ego vehicle (usually CAR).
    :param ego_harm: Harm for the ego vehicle.
    :param ego_v: Impact speed of the ego vehicle.
    :param ego_mass: Mass of the ego vehicle.
    :param obs_harm: Harm for the obstacle.
    :param obs_type: Type of obstacle.
    :param obs_v: Velocity of the obstacle.
    :param obs_mass: Estimated mass of the obstacle.
    :param pdof: Principle degree of force.
    :param ego_angle: Angle of impact area for the ego vehicle.
    :param obs_angle: Angle of impact area for the obstacle.
    :param time_step: Current time step.
    :param modes: Risk modes. Read from risk.json.
    :param planning_problem: Considered planning problem. Defaults to None.
    :param global_path: Global path for the planning problem. Defaults to None.
    :param driven_traj: Already driven trajectory of the ego vehicle. Defaults to None.
    """

    # clear everything
    plt.cla()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5.833, 8.25)

    ego_params = DynamicObstacleParams()
    ego_params.time_begin = time_step
    ego_params.draw_icon = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    # set plot limits to show the road section around the ego vehicle
    position = ego_vehicle.state_at_time(time_step).position
    plot_limits = [position[0] - 20,
                   position[0] + 20,
                   position[1] - 20,
                   position[1] + 20]

    scenario_params = MPDrawParams()
    scenario_params.time_begin = time_step
    scenario_params.dynamic_obstacle.show_label = True
    scenario_params.dynamic_obstacle.draw_icon = True
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#0065BD"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 1

    rnd = MPRenderer(ax=ax2, plot_limits=plot_limits)
    # plot the scenario at the current time step
    scenario.draw(rnd, draw_params=scenario_params)

    plt.gca().set_aspect('equal')

    # draw the planning problem
    if planning_problem is not None:
        planning_problem.draw(rnd)

    # mark the ego vehicle
    if ego_vehicle is not None:
        ego_vehicle.draw(rnd, draw_params=ego_params)

    rnd.render()

    # Draw global path
    if global_path is not None:
        plt.plot(global_path[:, 0], global_path[:, 1], color='blue',
                 zorder=20, label='global path')

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

    # get the target time to show it in the title
    if hasattr(planning_problem.goal.state_list[0], 'time_step'):
        target_time_string = ('Target-time: %.1f s - %.1f s' %
                              (planning_problem.goal.state_list[0].
                               time_step.start * scenario.dt,
                               planning_problem.goal.state_list[0].
                               time_step.end * scenario.dt))
    else:
        target_time_string = 'No target-time'

    plt.legend()
    plt.title('Time: {0:.1f} s'.format(time_step * scenario.dt) + '    ' +
              target_time_string)

    # get mode description
    if modes["harm_mode"] == "log_reg":
        mode = "logistic regression"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "ref_speed":
        mode = "reference speed"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "gidas":
        mode = "GIDAS P(MAIS 2+)"
        harm = "P(MAIS 2+)"
    else:
        mode = "None"
        harm = "None"

    # get angle mode
    if modes["ignore_angle"] is True or modes["harm_mode"] == "gidas":
        angle = "ignoring impact areas"
    else:
        if modes["reduced_angle_areas"]:
            angle = "considering impact areas reduced on front, side, and " \
                    "rear crashes"
        else:
            angle = "considering impact areas according to the clock system"

        if modes["sym_angle"]:
            angle += " with symmetric coefficients"
        else:
            angle += " with asymmetric coefficients"

    # description of crash
    description = "Collision at {:.1f} s in ". \
                      format(time_step * scenario.dt) + \
                  str(scenario.scenario_id) + \
                  "\n\nCalculate harm using the " + mode + " model by " + angle + \
                  "\n\nEgo vehicle harm " + harm + ": {:.3f}".format(ego_harm) + \
                  "\nObstacle harm " + harm + ": {:.3f}".format(obs_harm) + \
                  "\n\nCrash parameters:\n\nEgo type: " + str(ego_type)[13:] + \
                  "\nEgo velocity: {:.2f}m/s". \
                      format(ego_v) + \
                  "\nEgo mass: {:.0f}kg".format(ego_mass) + \
                  "\nImpact angle for the ego vehicle: {:.2f}°". \
                      format(ego_angle * 180 / np.pi) + \
                  "\n\nObstacle type: " + str(obs_type)[13:]
    if obs_mass is not None and obs_v is not None and obs_angle is not None:
        description += "\nObstacle velocity: {:.2f}m/s".format(obs_v) + \
                       "\nObstacle mass: {:.0f}kg".format(obs_mass) + \
                       "\nImpact angle for the obstacle: {:.2f}°". \
                           format(obs_angle * 180 / np.pi)
    description += "\n\nPrinciple degree of force: {:.2f}°". \
        format(pdof * 180 / np.pi)

    # add description of crash
    ax1.axis('off')
    ax1.text(0, 1, description, verticalalignment='top', fontsize=8,
             wrap=True)

    fig.suptitle("Collision in " + str(scenario.scenario_id) + " detected",
                 fontsize=16)

    # Create directory for pictures
    destination = os.path.join(destination, "collisions")
    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.savefig(destination + "/" + str(scenario.scenario_id) + ".svg", format="svg")
    plt.close()


def plot_final_trajectory(scenario: Scenario, planning_problem: PlanningProblem, state_list: List[CustomState],
                          config: Configuration, log_path: str, ref_path: np.ndarray = None, save=None, show=None):
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
    if config.visualization.save_plots or save:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.svg", format='svg',
                    bbox_inches='tight')

    # show plot
    if config.visualization.show_plots or show:
        matplotlib.use("TkAgg")
        # plt.show(block=False)
        # plt.pause(0.0001)


def plot_multiagent_scenario_final_trajectories(scenario, agent_list,
                                       config,
                                       log_path,
                                       show, save):
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
    for agent in agent_list:
        # visualize planning problem
        agent.planning_problem.draw(rnd)
        # visualize occupancies of trajectory
        for i, state in enumerate(agent.record_state_list):
            # state = agent.record_state_list[i]
            occ_pos = Rectangle(length=config.vehicle.length, width=config.vehicle.width, center=state.position,
                                orientation=state.orientation)
            if i >= 1:
                occ_params.opacity = 0.3
                occ_params.zorder = 50
            occ_pos.draw(rnd, draw_params=occ_params)

    # render scenario and occupancies
    rnd.render()

    for agent in agent_list:
        # visualize trajectory
        pos = np.asarray([state.position for state in agent.record_state_list])
        rnd.ax.plot(pos[:, 0], pos[:, 1], color='k', marker='x', markersize=3.0, markeredgewidth=0.4, zorder=21,
                    linewidth=0.8)
        # visualize reference path
        if agent.reference_path is not None:
            rnd.ax.plot(agent.reference_path[:, 0], agent.reference_path[:, 1], color='g', marker='.', markersize=1, zorder=19,
                            linewidth=0.8,
                            label='reference path')

    # save as .png file
    if save:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.svg", format='svg',
                    bbox_inches='tight')

    # show plot
    if show:
        matplotlib.use("TkAgg")
        # plt.show(block=False)
        # plt.pause(0.0001)


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


def visualize_scenario_and_pp(scenario: Scenario, planning_problem: PlanningProblem=None, save_path=None, cosy=None, t = 0):
    """Visualizes scenario, planning problem and (optionally) the reference path"""
    plot_limits = None
    ref_path = None
    if cosy is not None:
        try:
            ref_path = cosy.reference
        except:
            ref_path = np.array(cosy.reference_path())
        # x_min = np.min(ref_path[:, 0]) - 150
        # x_max = np.max(ref_path[:, 0]) + 150
        # y_min = np.min(ref_path[:, 1]) - 150
        # y_max = np.max(ref_path[:, 1]) + 150
        # plot_limits = [x_min, x_max, y_min, y_max]
        plot_limits = None

    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    rnd.draw_params.time_begin = t
    rnd.draw_params.traffic_light.draw_traffic_lights = False
    rnd.draw_params.traffic_sign.draw_traffic_signs = False
    scenario.draw(rnd)
    if planning_problem is not None:
        planning_problem.draw(rnd)
    rnd.render()
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=20,
                    linewidth=0.8, label='reference path')
        if hasattr(cosy, "ccosy"):
            proj_domain_border = np.array(cosy.ccosy.projection_domain())
        else:
            proj_domain_border = np.array(cosy.projection_domain())
        rnd.ax.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], color="orange", linewidth=0.8)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "reference_path.svg"), format='svg', bbox_inches='tight')
    else:
        plt.show()


def green_to_red_colormap():
    """Define a colormap that fades from green to red."""
    # This dictionary defines the colormap
    cdict = {
        "red": (
            (0.0, 0.0, 0.0),  # no red at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.8, 0.8),
        ),  # set to 0.8 so its not too bright at 1
        "green": (
            (0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no green at 1
        "blue": (
            (0.0, 0.0, 0.0),  # no blue at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no blue at 1
    }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap("GnRd", cdict)

    return GnRd
