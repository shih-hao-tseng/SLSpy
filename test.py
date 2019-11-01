from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import *
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
from sls_sim.VisualizationTools import *
import numpy as np

def test():
    cSpeedx = [2, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    cSpeedy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    Plot_Line_Chart(cSpeedx,cSpeedy,'test')

if __name__ == '__main__':
    test()