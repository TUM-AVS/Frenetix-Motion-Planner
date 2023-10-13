__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
This file contains the OccPlot class, which provides plot methods for the OcclusionModule.

"""

# imports
import os
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import frenetix_motion_planner.occlusion_planning.utils.occ_helper_functions as hf
import numpy as np
import pandas as pd
from commonroad.visualization.icons import get_obstacle_icon_patch
from commonroad.scenario.obstacle import ObstacleType

# get logger
msg_logger = logging.getLogger("Message_logger")


class OccPlot:
    def __init__(self, config=None, log_path=None, scenario_id=None, occ_scenario=None):
        self.occ_scenario = occ_scenario
        self.scenario_id = scenario_id
        self.log_path = log_path
        self.interactive_plot = config.occlusion.interactive_plot
        self.plot_window = config.occlusion.plot_window_dyn
        self.plot_backend = config.occlusion.plot_backend
        self.fig = None
        self.ax = None
        self.occ_cmap = LinearSegmentedColormap.from_list('gr', ["g", "y", "r"], N=10)
        self.time_step = None

        if self.interactive_plot:
            mpl.use(self.plot_backend)
            plt.ion()

    def step_plot(self, time_step=0, ego_state=None, visible_area_vm=None,
                  obstacles=None, visible_area=None, occluded_area=None, obstacle_id=False, x_cl=None, x_0=None,
                  ego_vehicle=None):

        self.time_step = time_step

        if self.ax is None:
            self._create_occ_figure()
        else:
            self.ax.clear()

        s = round(x_cl[0][0], 2) if x_cl is not None else 'unknown'
        d = round(x_cl[1][0], 2) if x_cl is not None else 'unknown'
        x = round(x_0.position[0], 2) if x_0 is not None else 'unknown'
        y = round(x_0.position[1], 2) if x_0 is not None else 'unknown'
        v = round(x_0.velocity, 2) if x_0 is not None else 'unknown'

        self.ax.set_title('Occlusion Module Plot of Timestep {} -- x: {} m, y: {} m, v: {} m/s -- s: {} m,  d: {} m'
                          .format(time_step, x, y, v, s, d))

        self.ax.set_xlim([ego_state.initial_state.position[0] - self.plot_window, ego_state.initial_state.position[0] + self.plot_window])
        self.ax.set_ylim([ego_state.initial_state.position[1] - self.plot_window, ego_state.initial_state.position[1] + self.plot_window])

        ##################
        # Plot Scenario
        ##################

        self._plot_scenario(self.ax)

        ##################
        # Plot Ego Vehicle
        ##################

        # plot ego vehicle
        if ego_state is not None:
            if ego_vehicle is None:
                ego_length = 5
                ego_width = 2
            else:
                ego_length = ego_vehicle.length
                ego_width = ego_vehicle.width

            pos_x = ego_state.initial_state.position[0]
            pos_y = ego_state.initial_state.position[1]
            try:
                ego_patch = get_obstacle_icon_patch(obstacle_type=ObstacleType('car'),
                                                    pos_x=pos_x,
                                                    pos_y=pos_y,
                                                    orientation=ego_state.initial_state.orientation,
                                                    vehicle_length=ego_length,
                                                    vehicle_width=ego_width,
                                                    vehicle_color='blue',
                                                    edgecolor='black',
                                                    zorder=10)

                self._add_patch(ego_patch)
            except:
                self.ax.plot(pos_x, pos_y, 'o', markersize=10)

        ##################
        # Plot visible area from visibility module
        ##################

        if visible_area_vm is not None:
            hf.plot_polygons(self.ax, visible_area_vm, 'g', zorder=2)

        ##################
        # Plot Obstacles
        ##################

        if obstacles is not None:
            for obst in obstacles:

                if obst.pos is None:
                    continue

                # define color
                if obst.visible_at_timestep:
                    color = 'green'
                    alpha = 1
                else:
                    color = "orange"
                    alpha = 0.5

                # create and plot patch
                try:
                    if obst.obstacle_role == "DYNAMIC":
                        pos_x = obst.pos[time_step][0]
                        pos_y = obst.pos[time_step][1]
                        orientation = obst.orientation[time_step]
                    else:
                        pos_x = obst.pos[0]
                        pos_y = obst.pos[1]
                        orientation = obst.orientation

                    if obst.obstacle_type.name == "UNKNOWN":
                        obst_type = ObstacleType('car')
                    else:
                        obst_type = obst.obstacle_type

                    obst_patch = get_obstacle_icon_patch(obstacle_type=obst_type,
                                                         pos_x=pos_x,
                                                         pos_y=pos_y,
                                                         orientation=orientation,
                                                         vehicle_length=obst.obstacle_shape.length,
                                                         vehicle_width=obst.obstacle_shape.width,
                                                         vehicle_color=color,
                                                         edgecolor='black',
                                                         zorder=10)
                    self._add_patch(obst_patch, alpha)
                    hf.fill_polygons(self.ax, obst.polygon, color, zorder=1, opacity=0.1)

                # plot polygon
                except:
                    if obst.obstacle_type.name == "PEDESTRIAN":
                        hf.fill_polygons(self.ax, obst.polygon.buffer(0.2), 'orangered', zorder=10, opacity=1)
                        hf.plot_polygons(self.ax, obst.polygon.buffer(0.2), 'k', zorder=11, opacity=1)
                    else:
                        hf.fill_polygons(self.ax, obst.polygon, color, zorder=1, opacity=alpha)

                if obstacle_id:
                    # plot obstacle id
                    x = obst.pos_point.x
                    y = obst.pos_point.y
                    self.ax.annotate(obst.obstacle_id, xy=(x, y), xytext=(x, y), zorder=100, color='white')

        if self.interactive_plot:
            plt.show(block=False)
            plt.pause(0.1)

    def _create_occ_figure(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('equal')
        self.fig.set_size_inches((10, 10))

    def save_plot_to_file(self, file_format='svg'):
        plot_dir = os.path.join(self.log_path, "occlusion_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{self.scenario_id}_{self.time_step}.{file_format}", format=file_format, dpi=300)

    def pause(self, t=0.1):
        plt.pause(t)

    def plot_trajectories(self, trajectories, color='k', label=None, zorder=1):
        if trajectories is not None:
            if type(trajectories) == list:
                for traj in trajectories:
                    self.ax.plot(traj.cartesian.x, traj.cartesian.y, color=color)
            else:
                if label is not None:
                    self.ax.plot(trajectories.cartesian.x, trajectories.cartesian.y, label=label)
                    plt.legend(loc="upper left")
                else:
                    self.ax.plot(trajectories.cartesian.x, trajectories.cartesian.y, color=color, zorder=zorder)

    def debug_trajectory_point_distances(self, occ_map, trajectory, traj_coords, distance, distance_weights):
        # debug plot for the distance calculation of the occ_uncertainty_map_evaluator
        self.plot_trajectories(trajectory, 'k')
        self.ax.plot(traj_coords[:, 0], traj_coords[:, 1], 'ko')
        for i, traj in enumerate(traj_coords):
            for j, map_coord in enumerate(occ_map[:, :2]):

                if distance[i, j] > 0:
                    dist = distance[i, j]
                    dist_weight = distance_weights[i, j]
                    x = [traj[0], map_coord[0]]
                    y = [traj[1], map_coord[1]]
                    self.ax.plot(x, y, 'b')
                    x = np.sum(x) / 2
                    y = np.sum(y) / 2
                    self.ax.annotate(str(dist) + "-" + str(dist_weight), xy=(x, y),
                                     xytext=(x + 0.2, y + 0.2), zorder=100)

    def plot_trajectories_cost_color(self, trajectories, costs, min_costs=None, max_costs=None, step_size=1,
                                     legend=False, harm=None, risk=None, print_costs=False):
        if costs is None:
            return

        # plot trajectories with color according to their costs
        if min_costs is None:
            min_costs = min(costs)
        if max_costs is None:
            max_costs = max(costs)

        if max_costs == 0:
            return

        if legend:
            # create Legend
            line_c = Line2D([0], [0], label='Costs 0% of max. Costs', color='c')
            line_g = Line2D([0], [0], label='Costs in the range of 1%-25% of max. Costs', color='g')
            line_y = Line2D([0], [0], label='Costs in the range of 26%-50% of max. Costs', color='y')
            line_o = Line2D([0], [0], label='Costs in the range of 51%-75% of max. Costs', color='orange')
            line_r = Line2D([0], [0], label='Costs in the range of 76%-100% of max. Costs', color='r')

            # access legend objects automatically created from data
            handles, labels = self.ax.get_legend_handles_labels()
            handles.extend([line_c, line_g, line_y, line_o, line_r])
            self.ax.legend(handles=handles, loc='upper left')

        for i, traj in enumerate(trajectories):
            if i % step_size != 0:
                continue
            if costs[i] == min_costs:
                self.plot_trajectories(traj, color='c')
            elif costs[i] <= 0.25 * max_costs:
                self.plot_trajectories(traj, color='g')
            elif 0.25 * max_costs < costs[i] <= 0.5 * max_costs:
                self.plot_trajectories(traj, color='y')
            elif 0.5 * max_costs < costs[i] <= 0.75 * max_costs:
                self.plot_trajectories(traj, color='orange')
            else:
                self.plot_trajectories(traj, color='r')

            if print_costs:
                if harm is not None:
                    msg_logger.debug('trajectory {}: harm {} -- risk {} -- costs {}'.format(i, harm[i], risk[i], costs[i]))
                else:
                    msg_logger.debug('trajectory {}: costs {} '.format(i, costs[i]))

    def plot_phantom_ped_trajectory(self, peds):
        # plot phantom ped trajectories
        for ped in peds:
            hf.fill_polygons(self.ax, ped.polygon.buffer(0.2), 'orangered', zorder=1000)
            self.ax.plot(ped.trajectory[:, 0], ped.trajectory[:, 1], 'coral', zorder=999)
            #self.ax.plot(ped.goal_position[0], ped.goal_position[1], 'bo')

    def plot_uncertainty_map(self, occlusion_map):
        if occlusion_map is not None:
            hf.plot_occ_map(self.ax, occlusion_map, self.occ_cmap)

    def final_evaluation_plot(self, crash):

        # Colorlist with HexCodes (TUM Colors)
        color_list = ['#0065BD', '#E37222', '#A2AD00', '#000000', '#98C6EA', '#7F7F7F']

        # Change matplotlib standard colors
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

        # Create figure and axis
        fig = plt.figure(figsize=(12, 6))

        ax0 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), rowspan=3)
        ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 1))
        ax2 = plt.subplot2grid(shape=(3, 2), loc=(1, 1))
        ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1))

        axes = [ax1, ax2, ax3]

        # load log
        df = pd.read_csv(self.log_path + "/logs.csv", encoding='utf-8', delimiter=";", decimal='.')

        # plot commonroad scenario into axis 0
        self._plot_scenario(ax0)

        # plot the driven route into axis 0
        ax0.plot(df['x_position_vehicle_m'], df['y_position_vehicle_m'], label='Vehicle Position in global coordinates')
        ax0.set_xlabel('Coordinate x in m')
        ax0.set_ylabel('Coordinate y in m')
        ax0.axis("equal")

        plot_window = 10
        min_x = min(df['x_position_vehicle_m'])
        max_x = max(df['x_position_vehicle_m'])
        min_y = min(df['y_position_vehicle_m'])
        max_y = max(df['y_position_vehicle_m'])

        ax0.set(xlim=(min_x - plot_window, max_x + plot_window), ylim=(min_y - plot_window, max_y + plot_window))

        # plot a red x at the position of collision
        if crash:
            ax0.plot(df['x_position_vehicle_m'].iloc[-1], df['y_position_vehicle_m'].iloc[-1], c='red', marker='x', markersize=10)

        # plot log information for ax1, ax2, ax3
        selection = ['d_position_m', 'velocities_mps', 'accepted_occ_harm']
        labels = ['Coordinate d in m', 'Velocity in m/s', 'P_MAIS3+ in %']

        # select columns from dataframe
        selected_columns = df[selection]

        # plot selected columns
        for i, col in enumerate(selected_columns):
            if type(df[col][0]) is str:
                y = df[col].str.split(',').str[0].astype(float)
            else:
                y = df[col]
            axes[i].plot(df['s_position_m'], y)
            axes[i].set_xlabel('Coordinate s in m')
            axes[i].set_ylabel(labels[i])

            if crash:
                axes[i].plot(df['s_position_m'].iloc[-1], y.iloc[-1], c='red', marker='x', markersize=10)

        # save plot
        plot_dir = os.path.join(self.log_path, "occlusion_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{self.scenario_id}_{'Final_Evaluation'}.svg", format='svg', dpi=300)

        return

    def _add_patch(self, patch, alpha=1):
        # add patch to plot -- for plotting of vehicles
        for p in patch:
            p.set_alpha(alpha)
            self.ax.add_patch(p)

    def _plot_scenario(self, ax):
        if self.occ_scenario.ref_path is not None:
            ax.plot(self.occ_scenario.ref_path[:, 0], self.occ_scenario.ref_path[:, 1], c='y')

        if self.occ_scenario.lanelets_combined is not None:
            hf.fill_polygons(ax, self.occ_scenario.lanelets_combined, opacity=0.5, color='gray')
            hf.plot_polygons(ax, self.occ_scenario.lanelets_single, opacity=0.5, color='k')

        if self.occ_scenario.sidewalk_combined is not None:
            hf.plot_polygons(ax, self.occ_scenario.sidewalk_combined, opacity=0.5, color='black')

        if self.occ_scenario.lanelets_along_path_combined is not None:
            hf.fill_polygons(ax, self.occ_scenario.lanelets_along_path_combined, opacity=0.5, color='dimgrey')

# eof
