class SimParams:
    '''
    Contains parameters for simulating the system
    Inherits handle class with deep copy functionality
    '''

    def __init__(self):
        # initialize to zero instead of empty array
        self.tSim_ = 0;         # amount of time to simulate
        self.w_ = 0;            # disturbance w(t) (as per (3.1)
        self.openLoop_ = False; # whether to simulate the system open loop only

    def sanity_check(self):
        modestr = 'closed-loop'
        if self.openLoop_:
            modestr = 'open-loop'
        
        statusTxt = 'tSim= %d, %s' % (self.tSim_, modestr)
        
        # sanity check
        if len(self.w_) < 1:
            print('[SLS ERROR] The disturbance (w) is not specified!')
            return

        if len(self.w_[0]) < self.tSim_:
            print('[SLS ERROR] The specified length of the disturbance (w) is less than tSim!')
        
        return statusTxt