from slspy.system_model import LTISystem
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import *
from slspy.sls_objective import *
from slspy.sls_constraint import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
import numpy as np

def state_fdbk_rfd_example():
    # specify system matrices
    sys = LTISystem (
        Nx = 10, Nw = 10
    )

    np.random.seed(0)

    # generate sys._A, sys._B2
    generate_random_chain (
        system_model = sys,
        rho = 0.8,
        actuator_density = 1
    )
    generate_BCD_and_zero_initialization(sys)

    synthesizer = SLS (FIR_horizon = 15)
    # objective function
    obj_H2 = SLSObj_H2 ()
    synthesizer += obj_H2

    rfdCoeffs = [0.01, 0.1, 1, 10, 100, 1000]

    ## (1) basic sls (centralized controller) with rfd
    num_acts = []
    clnorms = []

    # add RFD regulator
    obj_rfd = SLSObj_RFD()

    for rfdCoeff in rfdCoeffs:
        synthesizer.setSystemModel (sys)
        obj_rfd._rfdCoeff = rfdCoeff
        synthesizer += obj_rfd
        synthesizer.synthesizeControllerModel ()

        new_act_ids = obj_rfd.getActsRFD()
        num_acts.append(len(new_act_ids))
        
        # check performance with rfd-designed system
        sysAfterRFD = sys.updateActuation(new_act_ids=new_act_ids)

        # only H2
        synthesizer.setSystemModel(sysAfterRFD)
        synthesizer.setObjOrCons(obj_H2)
        synthesizer.synthesizeControllerModel ()
        
        clnorms.append(synthesizer.getOptimalObjectiveValue())

    plot_line_chart(
        list_x=num_acts,
        list_y=clnorms,
        line_format='*-',
        title='Centralized RFD tradeoff curve',
        xlabel='Number of actuators',
        ylabel='Close loop norm'
    )

    ## (2) d-localized sls with rfd
    num_acts = []
    clnorms = []

    dlocalized = SLSCons_dLocalized (
        actDelay = 1,
        cSpeed = 2,
        d = 3
    )
    synthesizer += dlocalized

    for rfdCoeff in rfdCoeffs:
        synthesizer.setSystemModel (sys)
        obj_rfd._rfdCoeff = rfdCoeff
        synthesizer += obj_rfd
        synthesizer.synthesizeControllerModel ()

        new_act_ids = obj_rfd.getActsRFD()
        num_acts.append(len(new_act_ids))
        
        # check performance with rfd-designed system
        sysAfterRFD = sys.updateActuation(new_act_ids=new_act_ids)

        # only H2
        synthesizer.setSystemModel(sysAfterRFD)
        synthesizer.setObjOrCons(obj_H2)
        synthesizer.synthesizeControllerModel ()
        
        clnorms.append(synthesizer.getOptimalObjectiveValue())

    plot_line_chart(
        list_x=num_acts,
        list_y=clnorms,
        line_format='*-',
        title='d-localized RFD tradeoff curve',
        xlabel='Number of actuators',
        ylabel='Close loop norm'
    )

    ## (3) approximate d-localized sls with rfd
    num_acts = []
    clnorms = []

    approx_dlocalized = SLSCons_ApproxdLocalized (
        base = dlocalized,
        robCoeff = 10e4
    )
    synthesizer.setObjOrCons(approx_dlocalized)

    for rfdCoeff in rfdCoeffs:
        synthesizer.setSystemModel (sys)
        obj_rfd._rfdCoeff = rfdCoeff
        synthesizer += obj_rfd
        synthesizer.synthesizeControllerModel ()

        new_act_ids = obj_rfd.getActsRFD()
        num_acts.append(len(new_act_ids))
        
        # check performance with rfd-designed system
        sysAfterRFD = sys.updateActuation(new_act_ids=new_act_ids)

        # only H2
        synthesizer.setSystemModel(sysAfterRFD)
        synthesizer.setObjOrCons(obj_H2)
        synthesizer.synthesizeControllerModel ()
        
        clnorms.append(synthesizer.getOptimalObjectiveValue())

    plot_line_chart(
        list_x=num_acts,
        list_y=clnorms,
        line_format='*-',
        title='Approx d-localized RFD tradeoff curve',
        xlabel='Number of actuators',
        ylabel='Close loop norm'
    )

if __name__ == '__main__':
    state_fdbk_rfd_example()