function generate_rand_chain(sys, rho, actDens)
% Populates (A, B2) of the specified system with a random chain 
% (tridiagonal A matrix) and a random actuation (B) matrix
% Also sets Nu of the system accordingly
% Inputs
%    sys     : LTISystem containing system matrices
%    rho     : normalization value; A is generated s.t. max |eig(A)| = rho
%    actDens : actuation density of B, in (0, 1]
%              this is approximate; only exact if things divide exactly

if not(sys.Nx)
    error('[SLS ERROR] Nx = 0 in the LTISystem! Please specify it first');
end

sys.Nu = ceil(sys.Nx * actDens);

A                = speye(sys.Nx);
A(1:end-1,2:end) = A(1:end-1,2:end)+ diag(randn(sys.Nx-1, 1));
A(2:end,1:end-1) = A(2:end,1:end-1)+ diag(randn(sys.Nx-1, 1));

B = sparse(sys.Nx, sys.Nu);
for i=1:1:sys.Nu
    B(mod(floor(1/actDens*i-1), sys.Nx)+1,i) = randn();
end

sys.A  = A / max(abs(eigs(A))) * rho;
sys.B2 = B;