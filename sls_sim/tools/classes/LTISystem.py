class LTISystem:
    '''
    Contains all matrices of an LTI system as per (3.1)
    Inherits handle class with deep copy functionality
    '''

    def __init__(self):
        # initialize to zero instead of empty array

        # state       : x(t+1)= A*x(t)  + B1*w(t)  + B2*u(t)
        self.A  = 0
        self.B1 = 0
        self.B2 = 0

        # reg output  : z_(t) = C1*x(t) + D11*w(t) + D12*u(t)
        self.C1  = 0
        self.D11 = 0
        self.D12 = 0

        # measurement : y(t)  = C2*x(t) + D21*w(t) + D22*u(t)
        self.C2  = 0
        self.D21 = 0
        self.D22 = 0

        # number of states and number of actuators
        self.Nx = 0
        self.Nu = 0

    # make a new system with the dynamics of the old system and updated
    # actuation (based on rfd output)
    @staticmethod
    def updateActuation (oldObj, slsOuts):
        obj = oldObj
    
        # extract columns based on slsOuts.acts_
        obj.B2  = []
        for row in oldObj.B2:
            obj.B2.append ([row[i] for i in slsOuts.acts_])
    
        obj.D12  = []
        for row in oldObj.D12:
            obj.D12.append ([row[i] for i in slsOuts.acts_])

        obj.Nu  = len(slsOuts.acts_)
    
        return obj