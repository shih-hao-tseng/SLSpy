from slspy.system_model import LTISystem
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
import numpy as np

def test():
    x = [np.ones([3,1]),np.ones([3,1]),np.ones([3,1])]
    Bu = [np.ones([3,1]),np.ones([3,1]),np.ones([3,1])]
    xDes = [np.ones([3,1]),np.ones([3,1]),np.ones([3,1])]
    plot_time_trajectory(x, Bu, xDes)

if __name__ == '__main__':
    test()