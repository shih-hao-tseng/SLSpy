from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import *
from sls_sim.SLSObjective import *
from sls_sim.SLSConstraint import *
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
from sls_sim.VisualizationTools import *
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
    synthesizer.setSystemModel (sys)
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
        obj_rfd._rfdCoeff = rfdCoeff
        synthesizer += obj_rfd
        controller = synthesizer.synthesizeControllerModel ()

        num_acts.append(len(obj_rfd.getActsRFD()))
        print (num_acts)
        
        ## check performance with rfd-designed system
        #sysAfterRFD1     = updateActuation(sys, slsOutsRFD1)
        
        # only H2
        synthesizer.setObjOrCons(obj_H2)
        #slsOutsAfterRFD1 = state_fdbk_sls(sysAfterRFD1, slsParams)
        
        clnorms.append(synthesizer.getOptimalObjectiveValue())

    plot_line_chart(
        list_x=num_acts,
        list_y=clnorms,
        line_format='*-',
        title='Centralized RFD tradeoff curve',
        xlabel='Number of actuators',
        ylabel='Close loop norm'
    )

#    ## (2) d-localized sls with rfd
#    num_acts = []
#    clnorms = []
#
#    slsParams.mode_      = SLSMode.DLocalized;
#    slsParams.actDelay_  = 1;
#    slsParams.cSpeed_    = 2;
#    slsParams.d_         = 3;
#
#    for rfdCoeff in rfdCoeffs:
#        slsParams.rfdCoeff_ = rfdCoeff;
#        slsParams.rfd_      = true;
#        slsOutsRFD2         = state_fdbk_sls(sys, slsParams);
#
#         % check performance with rfd-designed system
#        sysAfterRFD2     = updateActuation(sys, slsOutsRFD2);
#        slsParams.rfd_   = false;
#        slsOutsAfterRFD2 = state_fdbk_sls(sysAfterRFD2, slsParams);
#
#        num_acts         = [num_acts; length(slsOutsRFD2.acts_)];
#        clnorms          = [clnorms; slsOutsAfterRFD2.clnorm_];
#
#    plot_line_chart(
#        list_x=num_acts,
#        list_y=clnorms,
#        line_format='*-',
#        title='d-localized RFD tradeoff curve',
#        xlabel='Number of actuators',
#        ylabel='Close loop norm'
#    )
#
#    ## (3) approximate d-localized sls with rfd
#    num_acts = []
#    clnorms = []
#
#    slsParams.mode_      = SLSMode.ApproxDLocalized;
#    slsParams.robCoeff_  = 10^4;
#
#    for rfdCoeff in rfdCoeffs:
#        slsParams.rfdCoeff_ = rfdCoeff
#        slsParams.rfd_      = true;
#        slsOutsRFD3         = state_fdbk_sls(sys, slsParams);
#
#        % check performance with rfd-designed system
#        sysAfterRFD3     = updateActuation(sys, slsOutsRFD3);
#        slsParams.rfd_   = false;
#        slsOutsAfterRFD3 = state_fdbk_sls(sysAfterRFD3, slsParams);
#
#        num_acts         = [num_acts; length(slsOutsRFD3.acts_)];
#        clnorms          = [clnorms; slsOutsAfterRFD3.clnorm_];
#    end
#
#    plot_line_chart(
#        list_x=num_acts,
#        list_y=clnorms,
#        line_format='*-',
#        title='Approx d-localized RFD tradeoff curve',
#        xlabel='Number of actuators',
#        ylabel='Close loop norm'
#    )

if __name__ == '__main__':
    state_fdbk_rfd_example()