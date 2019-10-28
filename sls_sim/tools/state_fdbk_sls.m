function slsOuts = state_fdbk_sls(sys, params)
% System level synthesis with state feedback
% Can optionally use regularization for design (rfd) as well
% Returns 
%    slsOuts: SLSOutputs containing system responses and other info
% Inputs
%    sys    : LTISystem containing system matrices
%    params : SLSParams containing parameters

statusTxt = params.sanity_check();
statusTxt = [char(10), 'Solving ', statusTxt];
disp(statusTxt);

cvx_begin quiet

% decision variables
if params.mode_ ~= SLSMode.Basic
    expression Rs(sys.Nx, sys.Nx, params.tFIR_)
    expression Ms(sys.Nu, sys.Nx, params.tFIR_)
    if params.mode_ == SLSMode.ApproxDLocalized
        expression Delta(sys.Nx, sys.Nx * params.tFIR_)
    end
else % basic SLS
    variable Rs(sys.Nx, sys.Nx, params.tFIR_)
    variable Ms(sys.Nu, sys.Nx, params.tFIR_)
end

% populate decision variables
% not totally necessarily but makes code easier to understand / use
for t = 1:params.tFIR_
    R{t} = Rs(:,:,t);
    M{t} = Ms(:,:,t);
end

% locality constraints
% automatically enforced by limiting support of R, M
if params.mode_ ~= SLSMode.Basic  
    [RSupport, MSupport, count] = get_localized_supports(sys, params);
    variable X(count)

    spot = 0;
    for t = 1:params.tFIR_
        suppR = find(RSupport{t});
        num = sum(sum(RSupport{t}));
        R{t}(suppR) = X(spot+1:spot+num);
        spot = spot + num;

        suppM = find(MSupport{t});
        num = sum(sum(MSupport{t}));
        M{t}(suppM) = X(spot+1:spot+num);
        spot = spot + num;
    end
end
 
objective  = get_objective(sys, params, R, M);
actPenalty = get_act_penalty(sys, params, M);
robustStab = 0;

% achievability  / approx achievability constraints
R{1} == eye(sys.Nx);
R{params.tFIR_} == zeros(sys.Nx, sys.Nx);

if params.mode_ == SLSMode.ApproxDLocalized
    for t=1:params.tFIR_-1
        Delta(:,(t-1)*sys.Nx+1:t*sys.Nx) = R{t+1} - sys.A*R{t} - sys.B2*M{t};
    end
    robustStab = norm(Delta, inf); % < 1 means we can guarantee stab
    objective = objective + params.robCoeff_ * robustStab;    
else
    for t=1:params.tFIR_-1
        R{t+1} == sys.A*R{t} + sys.B2*M{t};
    end
end

if params.rfd_
    objective = objective + params.rfdCoeff_ * actPenalty;
end

% solve minimization problem
minimize(objective);
cvx_end

% outputs
slsOuts              = SLSOutputs;
slsOuts.acts_        = get_acts_rfd(sys, params, M); % rfd actuator selection
slsOuts.R_           = R;
slsOuts.M_           = M;
slsOuts.robustStab_  = robustStab;
slsOuts.solveStatus_ = cvx_status;

if strcmp(cvx_status, 'Solved')
    statusTxt = 'Solved!';
else
    statusTxt = sprintf('[SLS WARNING] Solver exited with status %s', cvx_status);
end
disp([char(9), statusTxt]);

% optimal value of objective function without regularization terms
slsOuts.clnorm_     = get_objective(sys, params, R, M);

end


% local functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RSupport, MSupport, count] = get_localized_supports(sys, params)
commsAdj  = abs(sys.A) > 0;
localityR = commsAdj^(params.d_-1) > 0;

count = 0;
for t = 1:params.tFIR_
    RSupport{t} = min(commsAdj^(floor(max(0, params.cSpeed_*(t-params.actDelay_)))),localityR) > 0;
    MSupport{t} = (abs(sys.B2)'*RSupport{t}) > 0;
    count       = count + sum(sum(RSupport{t})) + sum(sum(MSupport{t}));
end
end


function actPenalty = get_act_penalty(sys, params, M)
actPenalty = 0;
if params.rfd_
    for i = 1:sys.Nu
        Mi = [];
        for t = 1:params.tFIR_
            Mi = [Mi, M{t}(i,:)];
        end    
        actPenalty = actPenalty + norm(Mi, 2);
    end
end
end


function acts = get_acts_rfd(sys, params, M)
tol = 1e-4;

acts = [];
if params.rfd_
    for i=1:sys.Nu
        if norm(vec(M{1}(i,:)),2) > tol
            acts = [acts; i];
        end
    end    
end
end