[![DOI](https://zenodo.org/badge/700239470.svg)](https://zenodo.org/records/10078062)

[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


# Frenetix Motion Planner

This repository includes a Frenet trajectory planning algorithm in the [CommonRoad](https://commonroad.in.tum.de/) scenario format.
The Repo provides a python-based and a C++-accelerated Motion Planner [Frenetix](https://github.com/TUM-AVS/Frenetix/) implementation.

# Occlusion-aware Motion Planning


<img src="doc/images/pedestrians.png" alt="reactive-planner" width="600" />


To try the Occlusion-Ware module, you can find the following readme script here:

 * [frenetix_motion_planner/occlusion_planning/README.md](https://github.com/TUM-AVS/Frenetix-Motion-Planner/blob/master/frenetix_motion_planner/occlusion_planning/README.md).

# Standard Planner Initialization

### Requirements
The software is  developed and tested on recent versions of Linux. We strongly recommend to use [Ubuntu 22.04](https://ubuntu.com/download/desktop) or higher.
For the python installation, we suggest the usage of Virtual Environment with Python 3.10 or Python 3.9
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)


<details>
<summary> <h2> 🖥 How to Use the Package </h2> </summary>

### Installation & Run Code
1. Make sure that the following **dependencies** are installed on your system for the C++ implementation:
   * [Eigen3](https://eigen.tuxfamily.org/dox/) 
     * On Ubuntu: `sudo apt-get install libeigen3-dev`
   * [Boost](https://www.boost.org/)
     * On Ubuntu: `sudo apt-get install libboost-all-dev`
   * [OpenMP](https://www.openmp.org/) 
     * On Ubuntu: `sudo apt-get install libomp-dev`
   * [python3.10-full](https://packages.ubuntu.com/jammy/python3.10-full) 
        * On Ubuntu: `sudo apt-get install python3.10-full` and `sudo apt-get install python3.10-dev`

2. **Clone** this repository & create a new virtual environment `python3.10 -m venv venv`

3. **Install** the package:
    * Source & Install the package via pip: `source venv/bin/activate` & `pip install -r requirements.txt`
    * [Frenetix](https://pypi.org/project/frenetix/) should be installed automatically. If not please write [rainer.trauth@tum.de](mailto:rainer.trauth@tum.de).

4. **Optional**: Download additional Scenarios:
    * Clone commonroad scenarios on the **same level** as commonroad-reactive-planner --> not into commonroad-reactive-planner with: 
      * `git clone https://gitlab.lrz.de/tum-cps/commonroad-scenarios.git`

5. **Change** Configurations in _configurations/defaults/*.yaml_ if needed. 

6. **Change** Settings in **main.py** if needed. Note that not all configuration combinations may work. The following options are available:
   1. **use_cpp**: If _True_: The C++ Frenet Implementations will be used. 
   2. **start_multiagent**: If _True_: Start a multiagent run. For runtime reasons, C++ is automatically used.
   3. **evaluation_pipeline**: If _True_: Run many scenarios in a row. Set **scenario folder** accordingly.
   4. **use_specific_scenario_list**: If _True_: Run a specific scenario list. Example in _example_scenarios/scenario_list.csv_. Make sure all scnearios in the list are in the scenario folder.

7. **Run** the planner with `python3 main.py`
8. **Logs** and **Plots** can be found in _/logs/<scenario_name>_


<figure style="border: 2px solid #cccccc; padding: 10px; display: inline-block;">
<img src="doc/images/ZAM_Tjunction-1_8_T-1_038.png" alt="reactive-planner" width="500" />
</figure>

</details>


<details>
<summary> <h2> 📈 Test Data </h2> </summary>

Additional scenarios can be found [here](https://commonroad.in.tum.de/scenarios).

</details>

<details>
<summary> <h2> 🔧 Modules </h2> </summary>

Detailed documentation of the functionality behind the single modules can be found below.

1. [General Planning Algorithm](README.md)

2. [Frenetix C++ Trajectory Handler](https://github.com/TUM-AVS/Frenetix)

3. [Commonroad Scenario Handler](cr_scenario_handler/README.md)

4. [Behavior Planner](behavior_planner/README.md)

5. [Occlusion-aware Module](frenetix_motion_planner/occlusion_planning/README.md)

6. [Wale-Net](https://github.com/TUMFTM/Wale-Net)

7. [Risk-Assessment](https://github.com/TUMFTM/EthicalTrajectoryPlanning)

</details>

<details>
<summary> <h2> 📇 Contact Info </h2> </summary>

[Rainer Trauth](mailto:rainer.trauth@tum.de),
Institute of Automotive Technology,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

[Johannes Betz](mailto:johannes.betz@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

</details>

<details>
<summary> <h2> 📃 Citation </h2> </summary>
   
If you use this repository for any academic work, please cite our code:

```bibtex
@misc{GitHubRepo,
  author = {Rainer Trauth},
  title = {Frenetix Motion Planner},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.10078062},
  url = {https://github.com/TUM-AVS/Frenetix-Motion-Planner}
}
```
</details>
