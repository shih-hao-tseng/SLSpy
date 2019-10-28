class SLSParams:
    '''
    Contains parameters for SLS 
    Note that depending on the solver called, not all params may be used
    Inherits handle class with deep copy functionality
    '''

    def __init__(self):
        self.mode_     = 0     # an SLSMode (i.e. Basic, DLocalized)
        self.rfd_      = False # whether we want rfd
        
        # used for all modes
        self.tFIR_     = 0 # finite impulse response horizon

        # used for DLocalized and ApproxDLocalized modes only
        self.actDelay_ = 0 # actuation delay
        self.cSpeed_   = 0 # communication speed
        self.d_        = 0 # d-hop locality constraint
        
        self.robCoeff_ = 0 # regularization coeff for robust stability (used in approx-d sls only) 
        self.rfdCoeff_ = 0 # regularization coeff for rfd (used in rfd only)
        
        self.obj_      = 0 # an Objective (i.e. H2, HInf)

    def sanity_check(self):
        if self.tFIR_ == 0:
            print("[SLS ERROR] tFIR=0. Did you forget to specify it?")
        
        paramStr = "\ntFIR=%d" % self.tFIR_
        
        if self.mode_ == SLSMode.Basic:
            modeStr = 'basic'
        elif self.mode_ == SLSMode.DLocalized:
            paramStr = self.check_d_local(self, paramStr)
            modeStr = 'd-localized'
        elif self.mode_ == SLSMode.ApproxDLocalized:
            if self.robCoeff_ == 0:
                print('[SLS ERROR] Solving with approximate d-localized SLS but robCoeff=0. Did you forget to specify it?')
            paramStr = self.check_d_local(self, paramStr)
            paramStr += ', robCoeff=%.2f' % self.robCoeff_
            modeStr  = 'approx d-localized'
        else:
            print('[SLS ERROR] SLS mode unknown or unspecified!')

        # check objective & needed params
        if self.obj_ == Objective.H2:
            objStr = 'H2'
        elif self.obj_ == Objective.HInf:
            objStr = 'HInf'
        elif self.obj_ == Objective.L1:
            objStr = 'L1'
        else:
            objStr = 'constant'
    
        statusTxt = modeStr + ' SLS with ' + objStr + ' objective'

        if self.rfd_:
            if self.rfdCoeff_ == 0:
                print('[SLS WARNING] Solving with RFD but rfdCoeff=0. Did you forget to specify it?')
            statusTxt += ' and RFD, rfdCoeff=%0.2f' % self.rfdCoeff_

        statusTxt += paramStr

        return statusTxt


    def check_d_local(self, paramStr):
        # ensure all the needed parameters for localized sls are specified
        if self.d_ == 0:
            print('[SLS WARNING] Solving with locality constraints but d=0. Did you forget to specify it?')

        if self.cSpeed_ == 0:            
            print('[SLS WARNING] Solving with locality constraints but cSpeed=0. Did you forget to specify it?')

        if self.actDelay_ == 0:
            print('[SLS WARNING] Solving with locality constraints but actDelay=0. Did you forget to specify it?')
        
        paramStr += ', d=%d, cSpeed=%0.2f, actDelay=%d' % (self.d_, self.cSpeed_, self.actDelay_)

        return paramStr