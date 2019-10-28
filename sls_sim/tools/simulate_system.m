function [x, u] = simulate_system(sys, slsParams, slsOuts, simParams)
% Simulate system as per equation (2.8)
% Returns 
%    x, u      : state and actuation values
% Inputs
%    sys       : LTISystem containing system matrices
%    slsParams : SLSParams containing parameters
%    slsOuts   : SLSOutputs containing system responses and other info
%    simParams : SimParams; parameters for the simulation

statusTxt = simParams.sanity_check();
statusTxt = [char(10), 'Simulating with ', statusTxt];
disp(statusTxt);
        
x     = zeros(sys.Nx, simParams.tSim_); 
u     = zeros(sys.Nu, simParams.tSim_);
x_hat = zeros(sys.Nx, simParams.tSim_); 
w_hat = zeros(sys.Nx, simParams.tSim_);

for t=1:1:simParams.tSim_-1
    if (simParams.openLoop_ ~= 1) % closed loop simulation
        for tau=1:1:min(t-1, slsParams.tFIR_)
           u(:,t) = u(:,t) + slsOuts.M_{tau}*w_hat(:,t-tau);
        end

        for tau=1:1:min(t-1, slsParams.tFIR_-1)
           x_hat(:,t+1) = x_hat(:,t+1) + slsOuts.R_{tau+1}*w_hat(:,t-tau);       
        end 
    end
    
    x(:,t+1) = sys.A*x(:,t) + sys.B1*simParams.w_(:,t)+ sys.B2*u(:,t);
    w_hat(:,t) = x(:,t+1) - x_hat(:,t+1);
end
