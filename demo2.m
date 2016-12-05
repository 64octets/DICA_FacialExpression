%% Setup fastRPCA
cd stephenbeckr-fastRPCA-ffa256a
setup_fastRPCA;
cd ..
%%Initialization, Input File
% Output: "X" is the training matrix X which contains in its columns the vectorized training face images
%         "cellX" is X as cell
%         "expressionLabel" is the labels for each column in X

X = [];
expressionLabel = [];
filespath = 'C:\train'; %training folder
if ~isdir(filespath)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', filespath);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(filespath, '*.tiff');
tiffFiles = dir(filePattern);
reshapedimension = 10000;
nc = 7;
an = 1;
man = [];
di = 1;
mdi = [];
fe = 1;
mfe = [];
ha = 1;
mha = [];
ne = 1;
mne = [];
sa = 1;
msa = [];
su = 1;
msu = [];
for k = 1:length(tiffFiles)
  baseFileName = tiffFiles(k).name;
  fullFileName = fullfile(filespath, baseFileName);
  tmp = strsplit(baseFileName,'.');
  tmp{2} = regexprep(tmp{2}, '\d', '');
  %fprintf(1, 'Now reading %s\n', fullFileName);65536
  switch tmp{2}
      case 'AN'
          %fprintf(1,'Expression: Angry\n');
          man = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) man];
          an = an + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 1];
      case 'DI'
          %fprintf(1,'Expression: Disgust\n');
          mdi = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) mdi];
          di = di + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 2];
      case 'FE'
          %fprintf(1,'Expression: Fear\n');
          mfe = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) mfe];
          fe = fe + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 3];
      case 'HA'
          %fprintf(1,'Expression: Happy\n');
          mha = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) mha];
          ha = ha + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 4];
      case 'NE'
          %fprintf(1,'Expression: Neutral\n');
          mne = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) mne];
          ne = ne + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 5];
      case 'SA'
          %fprintf(1,'Expression: Sad\n');
          msa = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) msa];
          sa = sa + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 6];
      case 'SU'
          %fprintf(1,'Expression: Surprise\n');
          msu = [double(reshape(imresize(imread(fullFileName),60/256),3600,1)) msu];
          su = su + 1;
          X = [X double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
          expressionLabel = [expressionLabel 7];
  end

end

cellX = [];
cellX{1} = man;
cellX{2} = mdi;
cellX{3} = mfe;
cellX{4} = mha;
cellX{5} = mne;
cellX{6} = msa;
cellX{7} = msu;

%%Normalize each column of X to unit l2-norm .
% This function was created by author
% Output: Normalized X
X = normaliseColumns(X);

%%Compute low-rank matrices A
lowrankA = [];
% read lowrankA files from the last training
% for i=1:nc
%     lowrankA{i} = dlmread(strcat(strcat('lowrankA-',int2str(i)),'.txt'));
% end
%original
for i=1:nc
    nFrames     = size(cellX{i},2);
    lambda      = 2e-2;
    L0          = repmat( median(cellX{i},2), 1, nFrames );
    S0          = cellX{i} - L0;
    epsilon     = 5e-3*norm(cellX{i},'fro'); % tolerance for fidelity to data
    opts        = struct('sum',false,'L0',L0,'S0',S0,'max',true,...
        'tau0',3e5,'SPGL1_tol',1e-1,'tol',1e-3);
    [Lrpca,Srpca] = solver_RPCA_SPGL1(cellX{i},lambda,epsilon,[],opts);
    lowrankA{i} = Lrpca;
    %Write down low-rank matrix and sprase matrix to rpcaresult.txt
    dlmwrite(strcat(strcat('lowrankA-',int2str(i)),'.txt'),lowrankA{i});
end

%%Initialize U V , skinny SVD
% Output : U,V
M = [];
S = [];
N = [];
U = [];
V = [];
for i=1:nc
    [M{i},S,N{i}] = svd(lowrankA{i},0);
    U{i} = M{i};
    V{i} = M{i}';
end

%% Run the DICA to count V{i}
% Output: Struct S includes Dictionary1 for Identity , Dictionary2 for
%         Expression
[d,N] = size(X);
% params
options = struct;
% necessary fields
options.Labels{1} = ones(1,N); % Class Labels w.r.t. attribute 1 
options.Labels{2} = expressionLabel; % Class Labels w.r.t. attribute 2
% optional fields
options.eta = 0.1; % mutual incoherence param
options.rank1 = 1; % dimension of subspace corresponding to attribute 1
options.rank2 = 7; % dimension of subspace corresponding to attribute 2
options.normStyle1 = '*'; % nuclear norm ---> low-rank components for attribute 1
options.normStyle2 = '1'; % ell_1 norm   ---> sparse components for attribute 2
options.lambda2 = 0.001; % lambda for the sparse component (need to experiment with this to achieve good results)
% execute
S = DICA(X,U,V,options);

%% Normalize Dictionary
% Normalizing dictionary follows the downloaded function SRC

% Dictionary = [];
% 
% sizeofdictionary = size(S.Dictionary1);
% for i=1:sizeofdictionary(1)
%     Dictionary = [Dictionary S.Dictionary1{i}];
% end
% 
% Dictionary = normaliseColumns(Dictionary);
% Dictionary = Dictionary';
% Label = Label';
% 
% dlmwrite('Dictionary1.txt',Dictionary);

sizeofdictionary = size(S.Dictionary2);
for i=1:sizeofdictionary(1)
    Dictionary = [Dictionary S.Dictionary2{i}];
end

Dictionary = normaliseColumns(Dictionary);
Dictionary = Dictionary';
expressionLabel = expressionLabel';

dlmwrite('Dictionary2.txt',Dictionary);

%% Using SRC
% Output : "Evaluate" is (right-predicted labels/all labels)*100
%          "predictions" : predicted labels

queryimages = [];
filespath = 'C:\train';
if ~isdir(filespath)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', filespath);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(filespath, '*.tiff');
tiffFiles = dir(filePattern);
inputlabels = [];
for k = 1:length(tiffFiles)
  baseFileName = tiffFiles(k).name;
  fullFileName = fullfile(filespath, baseFileName);
  queryimages = [queryimages double(reshape(imresize(imread(fullFileName),60/256),3600,1))];
  tmp = strsplit(baseFileName,'.');
  tmp{2} = regexprep(tmp{2}, '\d', '');
  %fprintf(1, 'Now reading %s\n', fullFileName);65536
  switch tmp{2}
      case 'AN'
          %fprintf(1,'Expression: Angry\n');
          inputlabels = [inputlabels 1];
      case 'DI'
          %fprintf(1,'Expression: Disgust\n');
          inputlabels = [inputlabels 2];
      case 'FE'
          %fprintf(1,'Expression: Fear\n');
          inputlabels = [inputlabels 3];
      case 'HA'
          %fprintf(1,'Expression: Happy\n');
          inputlabels = [inputlabels 4];
      case 'NE'
          %fprintf(1,'Expression: Neutral\n');
          inputlabels = [inputlabels 5];
      case 'SA'
          %fprintf(1,'Expression: Sad\n');
          inputlabels = [inputlabels 6];
      case 'SU'
          %fprintf(1,'Expression: Surprise\n');
          inputlabels = [inputlabels 7];
  end

end

queryimages = queryimages';

[predictions,src_scores] = src(Dictionary,expressionLabel,queryimages,0.3);

% Evaluate
fail = 0;
niceokgood = 0;
numoftestpoints = size(predictions);
predictions = predictions';
for i=1:numoftestpoints(1)
    if predictions(i) == inputlabels(i)
        niceokgood = niceokgood + 1;
    else
        fail = fail + 1;
    end
end

evaluate = niceokgood*100/numoftestpoints(1);

