function [A]=sparse_represent(Test,Train,sp_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find sparse representation of Test data in terms of training data
% The following specifications of the data are required to be followed
% Train- a matrix in R^{n x p} :  n = no. of training signals
%                              :  p = dimensionality of training signals
% Test - a matrix in R^{k x p} :  k = no. of training signals
%                              :  p = dimensionality of training signals
% sp_level (Sparsity level) parameter has to be between [0,1] 
% Sparsity level of 1 ---> Pseudo inverse solution(for Y=Test and X=Train) A=Y*X'*inv(X*X')
% Sparsity level of 0 ---> As close as possible to the nearest neighbor solution
% For ADNI sparsity below 0.05 actually adds sparse coefficients to the solution 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%OUTPUT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A - a sparse matrix in R^{k x n} such that
% Test = A*Train (approximately) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%USAGE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[A]=sparse_represent(Test,Train,sp_level)

[n,p]=size(Train);
[k,pt]=size(Test);
if(p~=pt)
    sprintf('training data and test data must have the same dimensionality')
else
    X=Train;
    K_Tr=X*X';
    A=[];
    for i=1:k
        y=Test(i,:);
        K_te=X*y';
        K_y=y*y';
        [min_val,min_index]=min(K_te);
        a=sparse_represent_kernelized(K_y,K_Tr,K_te,sp_level);
        A=[A;a];
    end    
end
