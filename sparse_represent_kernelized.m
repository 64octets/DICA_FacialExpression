function [a]=sparse_represent_kernelized(K_y,K_Tr,K_te,sp_level)
%Find the sparse representation using kernels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let the input signal be y in R^{1 x p} 
% and the data dictionary be X in R^{n x p} 
% where p is signal dimensionality and 
% n is number of datapoints available for reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[a]=sparse_represent_kernelized(K_Tr,K_te,sparsity)
% where a is the solution to
%   minimize ||a||_1);
%   ||y-a*X||_2<=E
% Since p>>n for our case we require kernelized input such that
% K_y=y*y'
% K_Tr=X*X';
% K_te=X*y'
% sp_level (Sparsity level) parameter has to be between [0,1] 
% Sparsity level of 1 ---> Pseudo inverse solution(for Y=Test and X=Train) A=Y*X'*inv(X*X')
% Sparsity level of 0 ---> As close as possible to the nearest neighbor solution
% For ADNI sparsity below 0.05 actually adds sparse coefficients to the solution 

n=max(size(K_Tr));

%E_min=norm(y-y*X'*inv(X*X')*X,2);%The best we can do is the pseudo-inverse
%E_min=abs(sqrt(K_y-K_te'*(K_te'*inv(K_Tr))'-K_te'*inv(K_Tr)*K_te+K_te'*inv(K_Tr)*K_Tr*(K_te'*inv(K_Tr))'));%The kernel version
%[min_val,min_index]=min(K_te);
%E_max=norm(y-X(min_index,:),2);%If we do worse than the mean there is no sense in proceeding
%E_max= abs(sqrt(K_y-2*min_val+K_Tr(min_index,min_index)));%The kernel version
%E_max=K_y;
%E=E_min+sp_level*(E_max-E_min);

N=sp_level*n;
cvx_begin
  cvx_quiet(true);
  variable a(1,n);
  minimize a*K_Tr*transpose(a)+K_y-transpose(K_te)*transpose(a)-a*K_te;
  norm(a,1)<=N;
cvx_end
