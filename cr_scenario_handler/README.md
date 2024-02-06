# Multi-agent Simulation
The simulation framework can be used to run [CommonRoad](https://commonroad.in.tum.de/) scenarios with multiple agents. 
Recorded vehicles are replaced by intelligent agents each controlled by advanced trajectory planners.
An interface is provided in [cr_scenario_handler/planner_interfaces](../cr_scenario_handler/planner_interfaces) to integrate and test different planning algorithms.
The [Frenetix Motion Planner](../frenetix_motion_planner) is implemented as default trajectory planner. 


## 🚧 Requirements
The software is  developed and tested on recent versions of Linux. We strongly recommend to use [Ubuntu 22.04](https://ubuntu.com/download/desktop) or higher.
For the python installation, we suggest the usage of Virtual Environment with Python 3.10 or Python 3.9
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)

## 🔧 Installation
For the installation, please follow the installation steps [here](../README.md#pre-installation-steps)

## Getting Started
These instructions should help you to install the scenario handler and use it for development and testing purposes.
The full documentation can be built by calling `doxygen` inside the `cr_scenario_handler/doxygen` directory.


## Run Code
When the requirements are fulfilled and the installation was successful, the multi-agent simulation can directly be started 
with the Frenetix planner.
* The main entry point to the scenario handler is `main_multiagent.py`. Adjust the path within the python file to select the scenario you want to execute.
* Change the configurations if you want to run a scenario with a different setup under [configurations...](../configurations)
* By default, a multi-agent simulation is performed, using all dynamic obstacles from the scenario as agents. Adapt the `simulation.yaml` configuration file for other agent selections.

## Adjust Multi-agent Simulation

## Integrate Different Trajectory Planner


