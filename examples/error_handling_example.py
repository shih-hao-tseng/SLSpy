from slspy import *
import numpy as np

def error_handling_example():
    sys = LTI_System (
        Nx = 1, Nu = 1, Nw = 1
    )

    # generate sys._A, sys._B2
    generate_doubly_stochastic_chain (
        system_model = sys,
        rho = 1,
        actuator_density = 1,
        alpha = 0.2
    )
    generate_BCD_and_zero_initialization(sys)
    
    # use SLS controller synthesis algorithm
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 3
    )
    # set SLS objective
    synthesizer << SLS_Obj_H2()

    # using the default solver SLS_Sol_CVX with verbose = True
    sls_solver = synthesizer.getSolver()
    sls_solver.setOptions(verbose=True)

    # using maximize instead
    sls_solver.setOptimizationDirection('max') # it is equivalent to sls_solver.setOptimizationDirection(1)

    # the optimization problem is unbounded, and SLS will fail
    synthesizer.synthesizeControllerModel ()

if __name__ == '__main__':
    error_handling_example()