function [predictions,src_scores,uniqlabels]=src(Traindata,Trainlabels,Testdata,sp_level)
%The sparse representations classifier (SRC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Traindata- a matrix in R^{n x p}   :  n = no. of training signals
%                                    :  p = dimensionality of training signals
% Trainlabels - a vector in R^{n x 1}:  n = no. of training signals
% Testdata - a matrix in R^{k x p}   :  k = no. of training signals
%                                    :  p = dimensionality of training signals
% sp_level (Sparsity level) parameter has to be between [0,1] 
% Sparsity level of 1 ---> Pseudo inverse solution(for Y=Test and X=Train) A=Y*X'*inv(X*X')
% Sparsity level of 0 ---> As close as possible to the nearest neighbor solution
% For ADNI sparsity below 0.05 actually adds sparse coefficients to the solution 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%OUTPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%predictions  : class predictions using sparse representations classifier
%src_scores   : Residual matrix in R{n x q} for q unique labels are present in Trainlabels
%uniqlabels   : src_scores column 2 corresponds to the label in uniqlabels(2)
A=sparse_represent(Testdata,Traindata,sp_level);
uniqlabels=unique(Trainlabels);
c=max(size(uniqlabels));
for i=1:c
    R=Testdata-A(:,find(Trainlabels==uniqlabels(i)))*Traindata(find(Trainlabels==uniqlabels(i)),:);
    src_scores(:,i)=sqrt(sum(R.*R,2));
end
[maxval,indices]=min(src_scores');
predictions=uniqlabels(indices);
