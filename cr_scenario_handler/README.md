# Scenario Handler
This project allows running different motion planners on 
[CommonRoad](https://commonroad.in.tum.de/) scenarios.

## Getting Started
These instructions should help you to install the scenario handler and use it for development and testing purposes.
The full documentation can be built using doxygen in `doxygen/`.

### Requirements
The software is  developed and tested on recent versions of Linux and OS X.

For the python installation, we suggest the usage of Virtual Environment with Python 3.10.
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)

### Installation
1. Clone this repository, and 
   create a new virtual environment `python3.10 -m venv venv`

2. Install the package:
    * Source & Install the package via pip: `source venv/bin/activate` & `pip install -r requirements.txt`

3. Download Scenarios:
    * Clone commonroad scenarios on the same level as commonroad-reactive-planner 
      (not into commonroad-reactive-planner or sc_scenario_handler) with: 
      * `git clone https://gitlab.lrz.de/tum-cps/commonroad-scenarios.git`

4. (Optional: Build documentation)
    * Call `doxygen` inside the `cr_scenario_handler/doxygen` directory.

### Run Code
* The main entry point to the scenario handler is `main.py`. Adjust the path within the python file to select the scenario you want to execute.
* Change the configurations if you want to run a scenario with a different setup under `configurations/defaults/...`
* By default, a multi-agent simulation is performed, using all dynamic obstacles from the scenario as agents. Adapt the `multiagent.yaml` configuration file for other agent selections or for running a single-agent simulation.