function Y = normaliseColumns(X)
% normalise columns of a matrix to unit ell_2 norm. 
Y = zeros(size(X));
numObservations = size(X,2);
for i=1:numObservations
     ytilde = X(:,i); 
	 ell2 = norm(ytilde);
     if ell2==0
        Y(:,i) = ytilde; 
     else
        Y(:,i) = ytilde/ell2; 
     end   
end

end

