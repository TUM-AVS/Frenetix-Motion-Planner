
# Behavior Planner

Design a Behavior Planner concept suitable for a frenet based reactive planner to controle driving dynamics and stabilization

## Documentation

### BehaviorModule:
#### Tasks:
-	Coordination of three submodules: ```PathPlanner```, ```VelocityPlanner```, Finite State Machine (```FSM```)
-	Executed each time step do generate necessary output for the reactive planner
-	Parameter Management with parameter classes ```BM_state```, ```PP_state```, ```VP_state```, ```FSM_state```
#### Inputs:
-	Initialization: project path, scenario path, initialized ego state, vehicle parameters
-   Execution: predictions, current ego state, current time step
#### Outputs:
-	Reference path for reactive planner
-	Desired velocity for reactive planner
#### Generated important Parameters:
-	Load CR Scenario: ```BM_state.scenario```
-	Country ID of scenario: ```BM_state.country```
-	Navigation information like navigation path and necessary lane change maneuvers from navigation: ```BM_state.global_nav_route```, ```BM_state. nav_lane_changes_left```, ```BM_state. nav_lane_changes_right```
-	S-coordinate of current ego_position in current curvililinear coordinate system: ```BM_state.position_s```
-	Current general road information like current lanelet, current legal speed limit, current street setting like ```‘Highway’``` or ```‘Urban’```: ```BM_state.current_lanelet_id```, ```BM_state.current_lanelet```, ```BM_state.speed_limit```, ```BM_state.street_setting_scenario```
Changeable Parameters:
-	Future factor (velocity dependent factor to determine a necessary plan ahead time for the behavior planner to not destabilize the reactive planner): ```BM_state.future_factor (= (current ego velocity // 4) + 1)```

### PathPlanner:
#### Tasks:
-	Creation of static route plan, including detection of traffic lights, yield and stop signs, merging lanes, intersections, road exits, left and right turns
-	Creation of the reference paths, straight base reference path at the beginning
-	Implementation of lane change maneuvers withing the reference path
-	Updating of curvilinear coordinate system after reference path changes
#### Inputs:
-	Initialization: ```BM_state```

### RoutePlan:
#### Tasks:
-	Creation of Static Route plan with planning horizon of one CR Scenario
-	Detection of Static goals
#### Inputs:
-	Initialization: lanelet network, navigation route
-	Execution: 
#### Generated important Parameters:
-	Static route plan: ```PP_state.static_route_plan```
-	List of Yield Signs: ```yield_signs```
-	List of Stop Signs: ```stop_signs```
-	List of Traffic Lights: ```traffic_lights```
-	List of Turns: ```turns```
-	List of Road Exits: ```road_exits```
-	List of Lane Merges: ```lane_merges```
-	List of Intersections: ```intersections```

### ReferencePath:
#### Tasks:
-	Creation of straight base reference path with initialiszation
-	Implementation of lane change maneuvers and creation of updated reference path
-	Updating curvilinear coordinate system
#### Inputs:
-	Initialization: lanelet network, navigation route, ```BM_state```
-	Execution: current ego state, goal lanelet id for lane changes, future factor
#### Generated important Parameters:
-	Reference path: ```PP_state.reference_path```
-	Curvilinear coordinate system: ```PP_state.cl_ref_coordinate_system```
#### Changeable Parameters:
-	number of vertices for lane change maneuver: ```number_vertices_lane_change=6```

### VelocityPlanner:
#### Tasks:
-	Calculation of desired velocity
-	Calculation of a Time-To-Collision (TTC) dependent velocity if a preceding vehicle is detected
-	Calculation of a driving condition dependent max velocity with a maximum of the current legal speed limit
-	Calculation of a necessary safety distance to a preceding vehicle
-	Estimation of a longitudinal driving dynamics conditions
-	Estimation of a lateral driving dynamics conditions
-	Estimation of visual driving conditions
-	Executed each time step do estimate driving conditions and desired velocity
#### Inputs:
-	Initialization: BM_state
#### Generated important Parameters:
-	Information of closest preceding vehicle like distance and velocity: ```VP_state.closest_preceding_vehicle```, ```VP_state.dist_preceding_veh, VP_state```vel_prececing_veh
-	General condition factor: ```VP_state.condition_factor```
-	Default speed limit if no sign is indicating: V```P_state.speed_limit_default```
-	Condition dependent max velocity: ```VP_state.MAX```
-	TTC and safety distance dependent velocity: ```VP_state.TTC```
-	Chosen velocity mode (```MAX``` or ```TTC```): ```VP_state.velocity_mode```
-	goal velocity (```TTC``` or ```MAX```): ```VP_state.goal_velocity```
-	desired velocity, may be different from goal velocity depended of Behavior Planner input: ```VP_state.desired_velocity```
#### Changeable Parameters:
-	minimum safety distance: at the moment 1.5 * vehicle length
-	calculation of safety distance: ```d_safe_2``` (based on ‘Verifying the Safety of Lane Change Maneuvers of Self-driving Vehicles Based on Formalized Traffic Rules’ Christian Pek, Peter Zahn, and Matthias Althoff)
-	reaction time of vehicle: ```delta = 0.3```

### DrivingDynamicsConditions:
#### Tasks:
-	Estimate longitudinal and lateral driving dynamics conditions: grip, vehicle parameters, maintenance, street setting, curve parameters

### VisibilityConditions:
#### Tasks:
-	Estimate visibility conditions: human visibility, sensor visibility, unobstructed view, curve view

### FSM_model:
#### Tasks:
-	Mirror the current state of the ego vehicle with a finite state machine
-	States are organized in three layers: Street Setting, Behavior State, Situation State
-	The FSM is built in a recursive fashion to simplify the possible number of states and transitions for each FSM
-	Behavior States are split Dynamic and Static Behavior States
-	Static Behavior States represent the static goals from the static route plan of the path planner. Static Behavior is dependent on spatial coordinates like yield and stop signs or traffic lights
-	Dynamic Behavior is independent from spatial coordinates like lane change maneuvers
-	Situation States include ```IdentifyTargetLaneAndVehiclesOnTargetLane``` and ```IdentifyFreeSpaceOnTargetLaneForLaneChange```
#### Inputs:
-	Initialization: ```BM_state```
#### Generated important Parameters:
-	Street setting: ```FSM_state.street_setting```
-	Behavior state static: ```FSM_state.behavior_state_static```
-	Behavior dynamic static: ```FSM_state. behavior_state_dynamic```
-	Situation state static: ```FSM_state.situation_state_static```
-	Situation state dynamic: ```FSM_state.situation_state_dynamic```

### IdentifyTargetLaneAndVehiclesOnTargetLane(State):
#### Tasks:
-	Identify target lanelet for lane change maneuvers
-	Identify vehicles on target lanelet
#### Generated important Parameters:
-	List of obstacles on target lanelet: ```FSM_state.obstacles_on_target_lanelet```
-	Id of target lanelet: ```FSM_state.lane_change_target_lanelet_id```
-	Target lanelet object: ```FSM_state.lane_change_target_lanelet```
#### Changeable Parameters:
-	Search distance for vehicles on target lanelet: ```search_distance=speed_limit_default * 2```

### IdentifyFreeSpaceOnTargetLaneForLaneChange(State):
#### Tasks:
-	Identify direct free space on target lane
-	Identify free space on a target lane with an offset
-	Iterates over all obstacles on target lanelet and checks if they are further away than half the velocity distance + vehicle length
#### Generated important Parameters:
-	Free space on target lanelet flag to give permission for lane change: ```FSM_state.free_space_on_target_lanelet```
#### Changeable Parameters:
-	Risk factor to increase or decrease the minimum distance to vehicles on the target lanelet : ```risk_factor = 1```

### FSM_logic_modules:
#### Tasks:
-	Logic modules to determine the switching between the single states of the FSMs
#### Inputs:
-	Initialization: ```BM_state```
-   Execution: current state: ```cur_state```
#### Outputs:
-   new current state: ```cur_stat```
-   transition to new state: ```transition```


