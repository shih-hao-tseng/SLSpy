# Simple State Space Example
#   author: Shih-Hao Tseng (shtseng@caltech.edu)
# Synopsys:
#   1. generate the following (state space) system,
#     x(t+1) = [ 1  2 ] x(t) + [ 1 1 ] u(t) + [ 1 0 ] w(t)
#              [ 3  0 ]        [ 0 1 ]        [ 0 1 ]
#     where x is the state, u is the control, and w is the noise.
#   2. synthesize the controller by SLS
#     The SLS synthesizes the controller that minimizes the following L2 objective:
#       || Phi_x + 2 Phi_u ||_L2
#     (Equivalently, it minimizes the l2 norm of the regulated output
#       z_(t) = x(t) + 2 u(t)
#     )
#   3. simulate the controlled system given 
#     the initial state 
#       x(0) = [ 0 ]
#              [ 0 ]
#     and the noise 
#       n(0) = [ 0  ]
#              [ 10 ]
#       n(t) = 0 for all t > 0
from slspy import *
import numpy as np

def simple_state_space_example():
    # step-by-step programming flow:

    # 1. generate the following (state space) system:
    # the LTI system is as follows
    #   x(t+1)= A*x(t)  + B1*w(t)  + B2*u(t)
    #   z_(t) = C1*x(t) + D11*w(t) + D12*u(t)
    #   y(t)  = C2*x(t) + D21*w(t) + D22*u(t)
    # where x is the state, z_ is the output, y is the measurement, u is the control, and w is the noise.
    sys = LTI_System (
        Nx = 2, # dimension of x
        Nu = 2, # dimension of u
        Nw = 2  # dimension of w
    )

    # fill in the corresponding matrices
    sys._A   = np.array([[1,2],
                         [3,0]])
    sys._B2  = np.array([[1,1],
                         [0,1]])
    sys._B1  = np.array([[1,0],
                         [0,1]])

    sys._C1  = np.array([[1,0],
                         [0,1]])
    sys._D12 = np.array([[2,0],
                         [0,2]])


    # 2. synthesize the controller by SLS:
    # use SLS controller synthesis algorithm
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 20
    )
    # set SLS objective
    synthesizer << SLS_Obj_H2()
    
    # synthesize controller (the generated controller is actually initialized)
    SLS_controller = synthesizer.synthesizeControllerModel ()


    # 3. simulate the controlled system given the noise:
    # initialize x
    sys.initialize (x0 = np.zeros([sys._Nx, 1]))

    sim_horizon = 25
    # generate noise
    # n(0) = [ 0  ]
    #        [ 10 ]
    noise = FixedNoiseVector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][1] = 10

    simulator = Simulator (
        system = sys,
        controller = SLS_controller,
        noise = noise,
        horizon = sim_horizon
    )

    # run the simulation
    x_history, y_history, z_history, u_history, w_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Step-by-step Synthesis')


def simple_state_space_alternative_flow_example():
    # This function shows an alternative programming flow which is equivalent to the 
    #   step-by-step programming flow above and is mostly used in the other examples

    # we usually declare all the variables and perform controller synthesis at the end
    sys = LTI_System (
        Nx = 2, # dimension of x
        Nu = 2, # dimension of u
        Nw = 2  # dimension of w
    )

    # fill in the corresponding matrices
    sys._A   = np.array([[1,2],
                         [3,0]])
    sys._B2  = np.array([[1,1],
                         [0,1]])
    sys._B1  = np.array([[1,0],
                         [0,1]])

    sys._C1  = np.array([[1,0],
                         [0,1]])
    sys._D12 = np.array([[2,0],
                         [0,2]])

    # initialize x
    sys.initialize (x0 = np.zeros([sys._Nx, 1]))

    sim_horizon = 25
    # generate noise
    # n(0) = [ 0  ]
    #        [ 10 ]
    noise = FixedNoiseVector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][1] = 10

    simulator = Simulator (
        system = sys,
        noise = noise,
        horizon = sim_horizon
    )

    # use SLS controller synthesis algorithm
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 20
    )
    # set SLS objective
    synthesizer << SLS_Obj_H2()

    # synthesize controller (the generated controller is actually initialized)
    # and use the synthesized controller in simulation
    simulator.setController (
        controller = synthesizer.synthesizeControllerModel ()
    )

    # run the simulation
    x_history, y_history, z_history, u_history, w_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Synthesis at the End')

if __name__ == '__main__':
    simple_state_space_example()
    simple_state_space_alternative_flow_example()
    keep_showing_figures()