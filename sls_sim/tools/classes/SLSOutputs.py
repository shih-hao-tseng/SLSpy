class SLSOutputs:
    '''
    Contains outputs of SLS 
    Depending on the solver / mode called, not all outputs are set 
    Inherits handle class with deep copy functionality
    '''
    def __init__(self):
        # initialize to zero instead of empty array

        # based on (5.1),(5.2),(5.3)
        # reminder: the two disturbances are dx = B1*w, dy = D21*w
        self.R_ = 0 # dx to x transfer matrix (phi_xx)
        self.M_ = 0 # dx to u transfer matrix (phi_ux)

        # output feedback only
        self.N_ = 0 # dy to x transfer matrix (phi_xy)
        self.L_ = 0 # dy to u transfer matrix (phi_uy)
        
        self.clnorm_     = 0 # final (optimal value) of original objective
                             # doesn't include regularization terms
        self.acts_       = 0 # indices of actuators (u) kept after rfd
        self.robustStab_ = 0 # cvx_status

        # TODO: this enforces small gain on l1->l1; should generalize
        self.robustStab_; # inf norm of delta from (2.24), (4.22)
                          # <1 means we can guarantee stab