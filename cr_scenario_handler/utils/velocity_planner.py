from typing import Optional, Tuple, List
from shapely.geometry import Point
from commonroad.geometry.shape import Rectangle, Circle, Polygon, Shape, ShapeGroup


class VelocityPlanner:
    def __init__(self, scenario, planning_problem, coordinate_system):
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.DT = scenario.dt
        self.coordinate_system = coordinate_system
        self.default_goal_velocity = self._calculate_default_goal_velocity(planning_problem)
        self.used_goal_metric, self.goal_s_position, self.goal_centers, self.goal_shape = self._determine_goal_metrics(scenario,
                                                                                                      planning_problem)

    @staticmethod
    def _calculate_default_goal_velocity(planning_problem) -> Optional[float]:
        """Calculate the default goal velocity if velocity attributes are present."""
        goal_state = planning_problem.goal.state_list[0]
        if hasattr(goal_state, 'velocity'):
            start_velocity = max(goal_state.velocity.start, 0.01)
            end_velocity = max(goal_state.velocity.end, 0.01)
            return (start_velocity + end_velocity) / 2
        return None

    def _determine_goal_metrics(self, scenario, planning_problem):
        """Determine the goal metrics based on planning problem attributes."""
        goal_metric = None
        goal_centers = []
        goal_s_position = None
        goal_shape = None

        if self._is_lanelet_goal(planning_problem):
            goal_metric, goal_centers, goal_shape = self._process_lanelet_goal(scenario, planning_problem)

        elif hasattr(planning_problem.goal.state_list[0], "position"):
            goal_metric = "center" if hasattr(planning_problem.goal.state_list[0].position, "shapely_object") else None
            if goal_metric:
                goal_centers.append(planning_problem.goal.state_list[0].position.center)
                goal_shape = planning_problem.goal.state_list[0].position.shapely_object

        elif hasattr(planning_problem.goal.state_list[0], "time_step"):
            goal_metric = "time_step"

        if goal_metric != "time_step":
            goal_s_position = self._calculate_goal_s_position(goal_centers)

        return goal_metric, goal_s_position, goal_centers, goal_shape

    def _is_lanelet_goal(self, planning_problem) -> bool:
        """Check if the planning problem's goal is defined by lanelets."""
        return hasattr(planning_problem.goal, "lanelets_of_goal_position") and planning_problem.goal.lanelets_of_goal_position is not None

    def _process_lanelet_goal(self, scenario, planning_problem):
        """Process the lanelet-based goal."""
        goal_centers = []
        goal_lanelet_ids = planning_problem.goal.lanelets_of_goal_position[0]
        for lanelet_id in goal_lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            n_center_vertices = len(lanelet.center_vertices)
            goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
        goal_lanelet_id = self.planning_problem.goal.lanelets_of_goal_position[0][0]
        goal_polygon = self.scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id).polygon.shapely_object
        return "lanelets_of_goal_position", goal_centers, goal_polygon

    def _calculate_goal_s_position(self, goal_centers: List) -> Optional[float]:
        """Calculate the goal's s position based on goal centers."""
        goal_s_position = None
        for goals in goal_centers:
            try:
                curvilinear_coords = self.coordinate_system.ccosy.convert_to_curvilinear_coords(goals[0], goals[1])[0]
            except:
                curvilinear_coords = None
                print("Convert to curvilinear coords failed due to projection domain!")

            if curvilinear_coords is None or goal_s_position is None or (curvilinear_coords < goal_s_position):
                goal_s_position = curvilinear_coords

        return goal_s_position

    def set_new_scenario_and_planning_problem(self, scenario, planning_problem, coordinate_system):
        # Reinitializing with new scenario and planning problem
        self.__init__(scenario, planning_problem, coordinate_system)

    @staticmethod
    def clip_velocity(theoretically_desired_velocity, current_velocity, max_value=50, clip_value=5):
        # Define the lower and upper bounds
        lower_bound = max(current_velocity - clip_value, 0)
        upper_bound = min(current_velocity + clip_value, max_value)

        # Clip the value x within the bounds
        return max(min(theoretically_desired_velocity, upper_bound), lower_bound)

    def calculate_desired_velocity(self, x_0, s_position) -> float:
        """
        Calculate the desired velocity based on the vehicle's position and the goal.

        Args:
            x_0: Current state of the vehicle.
            s_position: Current position in the coordinate system.
            clip_value: clips velocity to prevent exploding velocity offset costs

        Returns:
            float: The calculated desired velocity.
        """
        if self.used_goal_metric != "time_step":
            if self._is_in_goal(x_0):
                if self.default_goal_velocity:
                    return self.clip_velocity(self.default_goal_velocity, x_0.velocity)
                else:
                    return x_0.velocity

        if self.used_goal_metric == "time_step":
            return x_0.velocity

        if not self.goal_s_position and self.default_goal_velocity:
            return self.clip_velocity(self.default_goal_velocity, x_0.velocity)
        elif not self.goal_s_position:
            return x_0.velocity

        distance_to_goal = self.goal_s_position - s_position
        remaining_time = round(self._calculate_remaining_time(x_0), 3)

        if remaining_time > 0.0:
            return self.clip_velocity(distance_to_goal / remaining_time, x_0.velocity)
        elif self.default_goal_velocity:
            return self.clip_velocity(self.default_goal_velocity, x_0.velocity)
        else:
            return x_0.velocity

    def _is_in_goal(self, x_0) -> bool:
        """Check if the vehicle is within the goal region."""
        return Point(x_0.position).within(self.goal_shape)

    def _calculate_remaining_time(self, x_0) -> float:
        """
        Calculate the remaining time to reach the goal.

        Args:
            x_0: Current state of the vehicle.

        Returns:
            float: Remaining time in seconds.
        """
        remaining_time_steps = self.calc_remaining_time_steps(
            ego_state_time=x_0.time_step,
            t=0.0,
        )
        return remaining_time_steps * self.DT

    def calc_remaining_time_steps(self, ego_state_time: float, t: float) -> int:
        """
        Calculate the minimum and maximum amount of remaining time steps.

        Args:
            ego_state_time (float): Current time of the state of the ego vehicle.
            t (float): Checked time.

        Returns:
            Tuple[int, int]: Minimum and maximum remaining time steps.
        """
        considered_time_step = int(ego_state_time + t / self.DT)
        if hasattr(self.planning_problem.goal.state_list[0], "time_step"):
            min_remaining_time = self.planning_problem.goal.state_list[0].time_step.start - considered_time_step
            max_remaining_time = self.planning_problem.goal.state_list[0].time_step.end - considered_time_step
            return int((max_remaining_time+min_remaining_time)/2)


