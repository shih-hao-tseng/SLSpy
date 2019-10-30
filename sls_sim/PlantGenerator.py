from SystemModel import LTISystem
from math import floor, ceil
import numpy as np

'''
Some helper functions to generate the LTI system plant matrices
'''

def GenerateDoubleStochasticChain(system_model=None, rho=0, actuator_density=1, alpha=0):
    '''
    Populates (A, B2) of the specified system with these dynamics:
    x_1(t+1) = rho*[(1-alpha)*x_1(t) + alpha x_2(t)] + B(1,1)u_1(t)
    x_i(t+1) = rho*[alpha*x_{i-1}(t) + (1-2*alpha)x_i(t) + alpha*x_{i+1}(t)] + B(i,i)u_i(t)
    x_N(t+1) = rho*[alpha*x_{N-1}(t) + (1-alpha)x_N(t)] + B(N,N)u_N(t)
    Also sets Nu of the system accordingly
    Inputs
       system_model     : LTISystem containing system matrices
       rho              : stability of A; choose rho=1 for dbl stochastic A
       actuator_density : actuation density of B, in (0, 1]
                          this is approximate; only exact if things divide exactly
       alpha            : how much state is spread between neighbours
    '''
    if not isinstance(system_model,LTISystem):
        # only modify LTISystem plant
        return
    
    if system_model._Nx == 0:
        return 

    Nx = system_model._Nx
    Nu = int(ceil(Nx*actuator_density))
    system_model._Nu = Nu

    system_model._A = (1-2*alpha)*np.eye(Nx)
    system_model._A[0,0] += alpha
    system_model._A[Nx-1,Nx-1] += alpha
    tmp = alpha*np.eye(Nx-1)
    system_model._A[0:-1,1:] += tmp
    system_model._A[1:,0:-1] += tmp
    system_model._A *= rho

    system_model._B2 = np.zeros([Nx,Nu])
    for i in range (0,Nu):
        x = int(floor(i/actuator_density)) % Nx
        system_model._B2[x,i] = 1
