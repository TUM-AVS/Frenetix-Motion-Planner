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
The full documentation can be built by calling _doxygen_ inside the _cr_scenario_handler/doxygen_ directory.


## Run Code
When the requirements are fulfilled and the installation was successful, the multi-agent simulation can directly be started 
with the Frenetix planner.
* The main entry point to the scenario handler is `main_multiagent.py`. Adjust the path within the python file to select the scenario you want to execute.
* Change the configurations if you want to run a scenario with a different setup under [configurations](../configurations)
* By default, a multi-agent simulation is performed, using all dynamic obstacles from the scenario as agents. Adapt the **simulation.yaml** configuration file for other agent selections.

## Integration of external Trajectory Planner 
1. A **base class** with all attributes necessary for the simulation is provided in  [planner_interface ](/cr_scenario_handler/planner_interfaces)
2. Create a new file with an interface to fit your planner and save it in _cr_scenario_handler/planner_interface_\
    The new **interface** must be a **subclass** of _PlannerInterface_.
3. In _configurations/simulation/simulation_ adjust **used_planner_interface** with the **class-name** of your interface 

## Citation
- [Multi-agent Simulation Framework](https://arxiv.org/abs/2402.04720)
```bibtex
@misc{multiagent2024,
      title={Investigating Driving Interactions: A Robust Multi-Agent Simulation Framework for Autonomous Vehicles}, 
      author={Marc Kaufeld and Rainer Trauth and Johannes Betz},
      year={2024},
      eprint={2402.04720},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```


