function generate_dbl_stoch_chain(sys, rho, actDens, alpha)
% Populates (A, B2) of the specified system with these dynamics:
% x_1(t+1) = rho*[(1-alpha)*x_1(t) + alpha x_2(t)] + B(1,1)u_1(t)
% x_i(t+1) = rho*[alpha*x_{i-1}(t) + (1-2*alpha)x_i(t) + alpha*x_{i+1}(t)] + B(i,i)u_i(t)
% x_N(t+1) = rho*[alpha*x_{N-1}(t) + (1-alpha)x_N(t)] + B(N,N)u_N(t)
% Also sets Nu of the system accordingly
% Inputs
%    sys     : LTISystem containing system matrices
%    rho     : stability of A; choose rho=1 for dbl stochastic A
%    actDens : actuation density of B, in (0, 1]
%              this is approximate; only exact if things divide exactly
%    alpha   : how much state is spread between neighbours

if not(sys.Nx)
    error('[SLS ERROR] Nx = 0 in the LTISystem! Please specify it first');
end

sys.Nu = ceil(sys.Nx * actDens);

A = (1-2*alpha)*speye(sys.Nx);
A(1,1) = A(1,1) + alpha;
A(sys.Nx,sys.Nx) = A(sys.Nx,sys.Nx) + alpha;
A(1:end-1,2:end) = A(1:end-1,2:end)+ alpha*speye(sys.Nx-1);
A(2:end,1:end-1) = A(2:end,1:end-1)+ alpha*speye(sys.Nx-1);

B = sparse(sys.Nx, sys.Nu);
for i=1:1:sys.Nu
    B(mod(floor(1/actDens*i-1), sys.Nx)+1,i) = 1;
end

sys.A = A*rho;
sys.B2 = B;