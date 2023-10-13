__author__ = "Korbinian Moller, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""
Simple function to visualize CommonRoad Scenarios

"""

# import functions to read xml file and visualize commonroad objects
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

# generate path of the file to be opened
file_path = "ZAM_leereKreuzung-1_1_T-1.xml"

mpl.use('TkAgg')

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# plot the planning problem and the scenario for the fifth time step
x_min = 80
x_max = 160
y_min = 125
y_max = 185
plot_limits = [x_min, x_max, y_min, y_max]

plt.figure(figsize=(25, 10))
rnd = MPRenderer(plot_limits=plot_limits)

for i in range(0, 2):
    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    planning_problem_set.draw(rnd)
    rnd.render()
    plt.show(block=False)
    plt.pause(0.1)
    print(i)
    plt.axis('off')
    plt.savefig(f"{file_path}-{i}.pdf", format='pdf', pad_inches=0)

print('Done')
