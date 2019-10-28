def get_objective(sys, params, R, M):
    '''
    Gets the objective function value based on the 
    Returns 
       objective : objective function value
    Inputs
       sys       : LTISystem containing system matrices
       params    : SLSParams containing parameters
    '''

    if params.obj_ == Objective.H2:
        objective = compute_H2(sys, params, R, M)
    elif params.obj_ == Objective.HInf:
        objective = compute_Hinf(sys, params, R, M)
    elif params.obj_ == Objective.L1:
        objective = compute_L1(sys, params, R, M)
    else:
        objective = 0
        print('[SLS WARNING] Objective = constant, only finding feasible solution')

    return objective


# local functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_H2(sys, params, R, M):
    '''
    return ||[C1,D12][R;M]||_H2^2 as per (4.20)
    '''

    objective = 0
    for t = 1:params.tFIR_
        # need to do the vect operation because of quirk in cvx
        vect = vec([sys.C1, sys.D12]*[R{t};M{t}])
        objective = objective + vect'*vect

    return objective

def objective = compute_Hinf(sys, params, R, M)
% return max singular value of [C1,D12][R;M]

mtx = [];
for t = 1:params.tFIR_
    mtx = blkdiag(mtx, [sys.C1, sys.D12]*[R{t};M{t}]);
end

objective = sigma_max(full(mtx));
end


function objective = compute_L1(sys, params, R, M)
% return max row sum of [C1,D12][R;M]

mtx = [];
for t = 1:params.tFIR_
    mtx = blkdiag(mtx, [sys.C1, sys.D12]*[R{t};M{t}]);
end

objective = norm(mtx, Inf); % note: L1 is induced inf-inf norm
end
