from ..system_models import LTI_System
from math import floor, ceil
import numpy as np

'''
Some helper functions to generate the LTI system plant matrices
'''
def generate_matrices_from_ABCD (system_model=None, A=None, B=None, C=None, D=None):
    '''
    Given A, B, C, and D, partition them into the corresponding matrices
    '''
    # This function simply serves as an abbreviation
    # The user has to ensure that system_model is LTI_System, 
    # and the dimensions of the provided matrices are correct
    Nx = system_model._Nx
    Ny = system_model._Ny
    Nz = system_model._Nz
    Nu = system_model._Nu
    Nw = system_model._Nw

    system_model._A = A
    if B is None:
        system_model._B1 = None
        system_model._B2 = None
    else:
        system_model._B1 = B[:,0:Nw]
        system_model._B2 = B[:,Nw:Nw+Nu]

    if C is None:
        system_model._C1 = None
        system_model._C2 = None
    else:
        system_model._C1 = C[0:Nz,:]
        system_model._C2 = C[Nz:Nz+Ny,:]
    
    if D is None:
        system_model._D11 = None
        system_model._D12 = None
        system_model._D21 = None
        system_model._D22 = None
    else:
        system_model._D11 = D[0:Nz,0:Nw]
        system_model._D12 = D[0:Nz,Nw:Nw+Nu]
        system_model._D21 = D[Nz:Nz+Ny,0:Nw]
        system_model._D22 = D[Nz:Nz+Ny,Nw:Nw+Nu]

def generate_BCD_and_zero_initialization (system_model=None):
    # This function simply serves as an abbreviation
    # The user has to ensure that system_model is LTI_System
    system_model._Nw = system_model._Nx
    system_model._Nz = system_model._Nx + system_model._Nu

    system_model._B1  = np.eye (system_model._Nx, system_model._Nw)
    system_model._C1  = np.eye (system_model._Nz, system_model._Nx)
    system_model._D12 = np.concatenate ((np.zeros([system_model._Nx, system_model._Nu]), np.eye(system_model._Nu)), axis = 0)

    if not system_model._state_feedback:
        # assign the matrices for y as well
        system_model._C2  = np.eye(system_model._Ny, system_model._Nx)
        #system_model._D22 = np.eye(system_model._Ny, system_model._Nu)

    system_model.initialize (x0 = np.zeros([system_model._Nx, 1]))

def generate_doubly_stochastic_chain (system_model=None, rho=0, actuator_density=1, alpha=0):
    '''
    Populates (A, B2) of the specified system with these dynamics:
    x_1(t+1) = rho*[(1-alpha)*x_1(t) + alpha x_2(t)] + B(1,1)u_1(t)
    x_i(t+1) = rho*[alpha*x_{i-1}(t) + (1-2*alpha)x_i(t) + alpha*x_{i+1}(t)] + B(i,i)u_i(t)
    x_N(t+1) = rho*[alpha*x_{N-1}(t) + (1-alpha)x_N(t)] + B(N,N)u_N(t)
    Also sets Nu of the system accordingly
    Inputs
       system_model     : LTI_System containing system matrices
       rho              : stability of A; choose rho=1 for dbl stochastic A
       actuator_density : actuation density of B, in (0, 1]
                          this is approximate; only exact if things divide exactly
       alpha            : how much state is spread between neighbours
    '''
    if not isinstance(system_model,LTI_System):
        # only modify LTI_System plant
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
    for i in range (Nu):
        x = int(floor(i/actuator_density)) % Nx
        system_model._B2[x,i] = 1

def generate_random_chain (system_model=None, rho=1, actuator_density=1):
    '''
    Populates (A, B2) of the specified system with a random chain 
    (tridiagonal A matrix) and a random actuation (B) matrix
    Also sets Nu of the system accordingly
    Inputs
       system_model     : LTI_System containing system matrices
       rho              : normalization value; A is generated s.t. max |eig(A)| = rho
       actuator_density : actuation density of B, in (0, 1]
                          this is approximate; only exact if things divide exactly
    '''
    if not isinstance(system_model,LTI_System):
        # only modify LTI_System plant
        return

    if system_model._Nx == 0:
        return

    Nx = system_model._Nx
    Nu = int(ceil(Nx*actuator_density))
    system_model._Nu = Nu

    system_model._A = np.eye(Nx)

    if Nx > 1:
        system_model._A[0:-1,1:] += np.diag(np.random.randn(Nx-1))
        system_model._A[1:,0:-1] += np.diag(np.random.randn(Nx-1))

    eigenvalues, eigenvectors = np.linalg.eig(system_model._A)
    largest_eigenvalue = np.max(np.absolute(eigenvalues))

    # normalization
    system_model._A /= largest_eigenvalue
    system_model._A *= rho

    system_model._B2 = np.zeros([Nx,Nu])
    for i in range (Nu):
        x = int(floor(i/actuator_density)) % Nx
        system_model._B2[x,i] = np.random.randn ()