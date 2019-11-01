from slspy.system_model import LTISystem
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
import numpy as np

def test():
    adj = np.eye(3)
    adj[2,1] = 1

    coord = [np.array([1,2]),np.array([-1,2]),np.array([0,0])]
    plot_graph (adj,coord,'b')

if __name__ == '__main__':
    test()