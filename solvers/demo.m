% Demo using DICA on synthetic data
clear; clc
%% Synthetic Data 
d1 = 2; % number of low-rank subspaces (attribute 1)
d2 = 1; % number of sparse subspaces (attribute 2)
D = 600; % 1000 dimension of ambient space.
N = 300; % 300 number of points in each low-rank subspace.
% Low-rank Components
[U,~,V] = svd(rand(D));
cids = [];
U1 = U(:,1:d1);
V1 = rand(d1,N);
X1 = U1*V1; % X1 - 1st Low-Rank Component
R = orth(rand(D));
U2 = R*U1;
V2 = rand(d1,N);
X2 = U2*V2; % X2 - 2nd Low-Rank Component
% visualization
X1X2 = [X1 X2];
% Sparse Component
X3 = insertShape(zeros(size(X1X2)), 'FilledRectangle', [1,350,2*N,100], 'Color', 'green','Opacity',1);
X3 = rgb2gray(X3);
% Clean Data (Y): Superposition of Low-Rank Components and Sparse Component
Y = X1X2 + X3;
% Noisy Data (Ytilde): Y Corrupted with gross, sparse noise
E0x = sign(randn(D,2*N));
inds = rand(D,2*N)<0.8;
E0x(inds) = 0;
Ytilde = Y + E0x;
figure; 
subplot(2,1,1); imagesc(Y); title('Clean Data');
subplot(2,1,2); imagesc(Ytilde); title('Noisy Data');

%% Run the DICA
% params
options = struct; 
% necessary fields
options.Labels{1} = [ones(1,N), 2*ones(1,N)]; % Class Labels w.r.t. attribute 1 
options.Labels{2} = ones(1,2*N); % Class Labels w.r.t. attribute 1 
% optional fields
options.eta = 0.1; % mutual incoherence param
options.rank1 = d1; % dimension of subspace corresponding to attribute 1
options.rank2 = d2; % dimension of subspace corresponding to attribute 2
options.normStyle1 = '*'; % nuclear norm ---> low-rank components for attribute 1
options.normStyle2 = '1'; % ell_1 norm   ---> sparse components for attribute 2
options.lambda2 = 0.001; % lambda for the sparse component (need to experiment with this to achieve good results)
% execute
S = DICA(Ytilde,options);

%% Visualize Results - Compute Reconstruction Errors
fprintf('Computing reconstruction errors for noisy data and individual components...\n');
figure;
subplot(2,1,1); imagesc(Ytilde); title('Noisy Data');
subplot(2,1,2); imagesc(S.Rec); title('Reconstruction by the DICA');
error = norm(S.Rec-Ytilde,'fro')/norm(Ytilde,'fro');
fprintf('Noisy Data: %f\n',error);
figure; 
subplot(2,1,1); imagesc(X1); title('Low-Rank Component 1');
subplot(2,1,2); imagesc(S.Dictionary1{1}); title('Reconstruction by the DICA'); % low-rank 1
error = norm(S.Dictionary1{1}-X1,'fro')/norm(X1,'fro');
fprintf('Low-Rank Component 1: %f\n',error);
figure; 
subplot(2,1,1); imagesc(X2); title('Low-Rank Component 2');
subplot(2,1,2); imagesc(S.Dictionary1{2}); title('Reconstruction by the DICA'); % low-rank 2
error = norm(S.Dictionary1{2}-X2,'fro')/norm(X2,'fro');
fprintf('Low-Rank Component 2: %f\n',error);
figure; 
subplot(2,1,1); imagesc(X3); title('Sparse Component 1');
subplot(2,1,2); imagesc(S.Dictionary2{1}); title('Reconstruction by the DICA'); % sparse 1
error = norm(S.Dictionary2{1}-X3,'fro')/norm(X3,'fro');
fprintf('Sparse Component 1: %f\n',error);