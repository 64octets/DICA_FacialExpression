function Y = DICA(X,U,V,options)

% This function implements the Discriminant Incoherent Component Analysis (DICA) proposed in [1]. 
% If you use the code please cite [1]. 

% Inputs: 
% - 'X': the data matrix containing the observations stacked as column vectors. (size: dimensionality x numObservations). 
% - 'options': struct variable used to set the experimental options and parameters for the DICA. 
% It has to contain at least the field 'Labels'. The field 'Labels' is a cell with Labels{1} containing 
% numeric labels w.r.t. the 1st structured attribute (e.g., identity),  and Labels{2} containing numeric labels for the 2nd attribute (e.g., expression). 
% The code can be easily modified to account for labelled data w.r.t. just one or more than two structured attributes.
% For the other fields of the input argument 'options', see below. 
% Outputs:
% - 'Y': struct variable containing the outputs of the DICA.  

% DICA, version 0.9, Apr 12th 2016
% ________________/Author\___________________
% Christos Georgakis, iBUG Group, 
% Computing Dept., Imperial College London
% email: christos.georgakis@imperial.ac.uk
% ___________________________________________

% References
% [1] Discriminant Incoherent Component Analysis. 
% C. Georgakis, Y. Panagakis, M. Pantic. IEEE Transactions on Image Processing. 25(5): pp. 2021 - 2034, May 2016. 

% Data
[d,n] = size(X);
normXfro = norm(X,'fro');
% Normalise each column to ell_2 norm
X = normaliseColumns(X);
% Labels
if ~isfield(options, 'Labels') 
   error('Second argument has to contain the field ''Labels''!');
end
if ~iscell(options.Labels) 
   error('options.Labels has to be a cell array!');
end
Labels = struct;
Labels.attr1 = options.Labels{1};
Labels.attr2 = options.Labels{2};
if (numel(Labels.attr1)~=n) || (numel(Labels.attr2)~=n)
    error('Number of Labels{1} and Labels{2} has to be equal to the number of columns in the data matrix!');
end
% attribute 1 (e.g., identity)
uniqueLabels1 = unique(Labels.attr1);
numClasses1 = numel(uniqueLabels1);
% attribute 2 (e.g., expression)
uniqueLabels2 = unique(Labels.attr2);
numClasses2 = numel(uniqueLabels2);
% all
numClasses = numClasses1 + numClasses2;

% options
if ~isfield(options, 'lambda1') % lambda for the norm of attribute 1 (e.g., identity) classes -  needs tuning
    options.lambda1 =  1; 
end
if ~isfield(options, 'lambda2') % lambda for the norm of attribute 2 (e.g., expression) classes -  needs tuning
    options.lambda2 =  1;   
end
if ~isfield(options, 'lambda0') % lambda for the norm of outliers -  needs tuning
    options.lambda0 =  1/sqrt(max(d,n));   
end
if ~isfield(options, 'rank1') % specify number of components for attribute 1 subspace - needs tuning
    options.rank1 =  numClasses1;    
end
if ~isfield(options, 'rank2') % specify number of components for attribute 2 subspace - needs tuning
    options.rank2 =  numClasses2;    
end
if ~isfield(options, 'eta') % mutual incoherence parameter - needs tuning
    options.eta =  1e-1;
end
if ~isfield(options, 'weightLambdas') % weight lambdas according to the frequency of occurence of each attr1 and attr2 label in the data matrix
    options.weightLambdas = false; 
end
if ~isfield(options, 'tol') % convergence parameter
    options.tol = 1e-6; 
end
if ~isfield(options, 'rho')  % ADMM parameter 
    options.rho = 1.1;     
end
if ~isfield(options, 'mu')  % ADMM parameter 
    options.mu = 1/norm(X,2); 
end
if ~isfield(options, 'mu_bar')  % ADMM parameter 
    options.mu_bar = 1e10; 
end
if ~isfield(options, 'displayFlag')  % verbose (caution: verbosity makes the code much slower, as the rank of all components are computed.)
    options.displayFlag = false; 
end
if ~isfield(options, 'maxIter')  
    options.maxIter = 1000; 
end
if ~isfield(options, 'normStyle1') % norm for the attribute 1 classes. '1': for ell_1 norm, '*': for nuclear norm.  
    options.normStyle1 = '*';  
end
if ~isfield(options, 'normStyle2') % norm for the attribute 2 classes. '1': for ell_1 norm, '*': for nuclear norm. 
    options.normStyle2 = '1';  
end
% pass options to variables
lambda1 = options.lambda1;
lambda2 = options.lambda2;
lambda0 = options.lambda0;
tol = options.tol;
k1 = options.rank1;
k2 = options.rank2;
rho = options.rho;
mu = options.mu;
mu_bar = options.mu_bar;
displayFlag = options.displayFlag;
maxIter = options.maxIter;
weightLambdas = options.weightLambdas;
eta = options.eta;
normStyle1 = options.normStyle1;
normStyle2 = options.normStyle2;

% Initialization
fprintf('Initializing the DICA...\n');
E = zeros(d,n); % outliers
Y = zeros(d,n); % Lagrangian
N = zeros(d,n); %  (sum_{UiViXDci})
D = cell(numClasses,1); % the indicator matrices
% U = cell(numClasses,1); % subspaces
% V = cell(numClasses,1); % coefficients
VtranspV = cell(numClasses,1); % auxiliary variable to improve efficiency
indices = cell(numClasses,1);
frequency = zeros(numClasses,1);

% Initialize Us and Vs and summand N. 
% attribute 1 classes
for i=1:numClasses1
    indices{i} = find(Labels.attr1==uniqueLabels1(i));
    frequency(i) = numel(indices{i});
    D{i} = zeros(n);
    D{i}(indices{i},indices{i}) = eye(frequency(i));     
    %U{i} = eye(d,k1); % note: you can use PCA or Robust PCA initialization for the Us and Vs, instead
    %V{i} = U{i}';
    VtranspV{i} = V{i}'*V{i};     
    N = N + U{i}*V{i}*X*D{i};
end
% attribute 2 classes
for i=1:numClasses2
    indices{i+numClasses1} = find(Labels.attr2==uniqueLabels2(i));
    frequency(i+numClasses1) = numel(indices{i+numClasses1});
    D{i+numClasses1} = zeros(n);
    D{i+numClasses1}(indices{i+numClasses1},indices{i+numClasses1}) = eye(frequency(i+numClasses1));
    U{i+numClasses1} = U{i}; % note: you can use PCA or Robust PCA initialization for the Us and Vs, instead
    V{i+numClasses1} = U{i+numClasses1}';
    VtranspV{i+numClasses1} = V{i+numClasses1}'*V{i+numClasses1};    
    N = N + U{i+numClasses1}*V{i+numClasses1}*X*D{i+numClasses1};
end
% weighting 
if weightLambdas
    lambdas1 = (lambda1/n)*frequency(1:numClasses1);
    lambdas2 = (lambda2/n)*frequency(numClasses1+1:end);
else
    lambdas1 = lambda1*ones(numClasses1,1);
    lambdas2 = lambda2*ones(numClasses2,1);
end
lambdas = [lambdas1 ; lambdas2];
%% Main loop
iter = 0;
fprintf('Running main loop of the DICA...\n');
tic
while iter<maxIter
    
        iter = iter + 1; 

        % V{i}s for attribute 1 (e.g., Identity Classes)
        for i=1:numClasses1
            temp = {VtranspV{setdiff(1:numClasses,i)}}';                    
            summand = sum(cat(3,temp{:}),3);
            Vi_old = V{i};
            grad = (-U{i}')*(X - N - E + Y/mu)*(X*D{i})' +  2*(eta/mu)*V{i}*summand; 
            lipsch = 1.02*max(eig(X(:,indices{i})*X(:,indices{i})'+ 2*(eta/mu)*summand));            
            temp = V{i} - grad/lipsch;  
            thresh = lambdas(i)/(mu*lipsch);    
            if strcmp(normStyle1,'*') % nuclear norm (default)
                [U1,sigma1,V1] = svd(temp,'econ'); 
                sigma1 = diag(sigma1);
                svp = length(find(sigma1>thresh));
                if svp >= 1
                    sigma1 = sigma1(1:svp)-thresh;
                else
                    svp = 1;
                    sigma1 = 0;
                end
                V{i} = U1(:,1:svp)*diag(sigma1)*V1(:,1:svp)'; 
            elseif strcmp(normStyle1,'1') % ell_1 norm 
                V{i} = max(0,temp - thresh) + min(0,temp + thresh); 
            else % invalid norm 
                error('Unknown norm option for the attribute 1 classes. Please use options.normStyle1 = ''*'' for nuclear norm and options.normStyle1 = ''1'' for ell_1 norm.\n');
            end             
            VtranspV{i} = V{i}'*V{i};            
            N = N - U{i}*Vi_old*X*D{i} + U{i}*V{i}*X*D{i};         
        end  
        % V{i}s for attribute 2 (e.g., Expression Classes)
        for i=1:numClasses2
            temp = {VtranspV{setdiff(1:numClasses,i+numClasses1)}}';                     
            summand = sum(cat(3,temp{:}),3);
            Vi_old = V{i+numClasses1};
            grad = (-U{i+numClasses1}')*(X - N - E + Y/mu)*(X*D{i+numClasses1})' +  2*(eta/mu)*V{i+numClasses1}*summand; 
            lipsch = 1.02*max(eig(X(:,indices{i+numClasses1})*X(:,indices{i+numClasses1})'+ 2*(eta/mu)*summand));              
            temp = V{i+numClasses1} - grad/lipsch;  
            thresh = lambdas(i+numClasses1)/(mu*lipsch); 
            if strcmp(normStyle2,'*') % nuclear norm
                [U1,sigma1,V1] = svd(temp,'econ');
                sigma1 = diag(sigma1);    
                svp = length(find( sigma1 > thresh ));
                if svp >= 1
                   sigma1 = sigma1(1:svp)- thresh;
                else
                   svp = 1;
                   sigma1 = 0;
                end
                V{i+numClasses1} = U1(:,1:svp)*diag(sigma1)*V1(:,1:svp)'; 
            elseif strcmp(normStyle2,'1') % ell_1 norm (default)
                V{i+numClasses1} = max(0,temp - thresh) + min(0,temp + thresh);    
		    else % invalid norm
                error('Unknown norm option for the attribute 2 classes. Please use options.normStyle2 = ''1'' for ell_1 norm and options.normStyle2 = ''*'' for nuclear norm.\n');
            end         
            VtranspV{i+numClasses1} = V{i+numClasses1}'*V{i+numClasses1};            
            N = N - U{i+numClasses1}*Vi_old*X*D{i+numClasses1} + U{i+numClasses1}*V{i+numClasses1}*X*D{i+numClasses1};         
        end     
        % U{i}s 
        for i=1:numClasses
            Ui_old = U{i};
            Nrest = N - U{i}*V{i}*X*D{i};
            temp = (X - Nrest - E + (Y/mu) )*(V{i}*X*D{i})'; 
            [U1,~,V1] = svd(temp,'econ');
            U{i} = U1*V1';              
            N = N - Ui_old*V{i}*X*D{i} + U{i}*V{i}*X*D{i};  
        end
        % Outliers
        temp = X - N + Y/mu;
        E = max(0,temp - lambda0/mu) + min(0,temp + lambda0/mu);                         
        % Lagrangian and Fitting Error
        f = X - N - E;    
        relatError =  norm(f,'fro')/normXfro; 
        stopC = relatError < tol;             

        if displayFlag && (iter == 1 || mod(iter,25)== 0 || stopC)
            fprintf('iteration %d , f=%f..........\n',iter,norm(f,'fro')/normXfro);
            ranks = zeros(numClasses,1);
            for i=1:numClasses
                ranks(i) = rank(V{i});
            end
            ranks = ranks'; 
            densities = zeros(numClasses,1);
            for i=1:numClasses
                densities(i) = nnz(V{i})/numel(V{i});
            end
            densities = densities';                           
            disp(['ranks of V_i:' num2str(ranks)]);  
            disp(['densities of V_i:' num2str(densities)]);
            disp(['density of E:' num2str(nnz(E)/numel(E))]);
        end
        
        if stopC 
            break;
        else
		  % update Lagrangian and mu
            Y = Y + mu*f; 
            mu = min(mu_bar,mu*rho);            
        end

end
exectime = toc;
fprintf('Converged. Iter: %d, Execution time: %.1f.\n',iter,exectime);
% Outputs (comment out to discard fields you do not need). 
Y = struct;
Y.options = options;
Y.iter = iter;
Y.exectime = exectime;
Y.relatError = relatError;
Y.indices = indices;
Y.U = U;
Y.V = V;
Y.uniqueLabels1 = uniqueLabels1;
Y.uniqueLabels2 = uniqueLabels2;
Y.D = D;
Y.Rec = N; % Reconstruction by the DICA
Y.E = E; % Error accounting for outliers
% Dictionary for attribute 1 classes (e.g., identity)
Y.Dictionary1 = cell(numClasses1,1);
Y.Images1 = cell(numClasses1,1);
Y.Labels1 = cell(numClasses1,1);
for i=1:numClasses1
    Y.Dictionary1{i} =  U{i}*V{i}*X(:,indices{i});     
    Y.Images1{i} =  X(:,indices{i});
    Y.Labels1{i} = Y.uniqueLabels1(i)*ones(numel(indices{i}),1);
end
% Dictionary for attribute 2 classes (e.g., expression)
Y.Dictionary2 = cell(numClasses2,1);
Y.Images2 = cell(numClasses2,1);
Y.Labels2 = cell(numClasses2,1);
for i=1:numClasses2
    Y.Dictionary2{i} =  U{i+numClasses1}*V{i+numClasses1}*X(:,indices{i+numClasses1});      
    Y.Images2{i} =  X(:,indices{i+numClasses1}); 
    Y.Labels2{i} = Y.uniqueLabels2(i)*ones(numel(indices{i+numClasses1}),1);
end


end


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
