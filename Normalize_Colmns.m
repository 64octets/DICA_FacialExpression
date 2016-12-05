function B = Normalize_Colmns(A)
% Normalize each column of the given matrix to 1.
% B = NORMALIZE_COLUMNS(A).
% A is m x n input matrix.
% B is m x n output matrix with l2 norm of each column 1.
% i.e., ||b_i||_2 = 1.
B = A/(diag(sqrt(diag(A'*A))));
