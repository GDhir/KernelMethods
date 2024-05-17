%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');
N = 11;
ndims = 10;
mu = zeros( N, ndims );
rng('default')  % For reproducibility

for dim = 1:ndims
    firstDim = -1 + (1 + 1)*rand(N,1);
    mu( :, dim ) = firstDim;
end

mu = repmat(mu, 100, 1);
sigmaval = 0.8;
Sigma = diag( ones( ndims, 1 )*sigmaval );
data = mvnrnd(mu,Sigma);

figure;
plot( data(:, 1), data(:, 2), 'ro' )
xlabel('x');
ylabel('y');

%% standard PCA
d = min( N, ndims );
disp('Performing standard PCA...');
[Y1 , pcaEigVector, pcaEigValue] = PCA( data, d );
figure;
plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
title('standard PCA');


for k = 2:d
    truncVals = getAllReconstructions( pcaEigVector, data', k )';
    % figure;
    % plot( truncVals( 1:end, 1 ), truncVals( 1:end, 2 ), 'b*');
    % title('standard PCA');
    pcaError = computeCenterError( truncVals, mu );
    disp(pcaError)
end


%% polynomial kernel PCA
para=5;
disp('Performing polynomial kernel PCA...');
[Y2]=kPCA(data,d,'poly',para);
figure;hold on;
plot(Y2(1:end,1),Y2(1:end,2),'b*');
title('polynomial kernel PCA');
drawnow;

%% Gaussian kernel PCA
d = 200;
DIST=distanceMatrix(data);
DIST(DIST==0)=inf;
DIST=min(DIST);
para = 2*mean(DIST);
disp('Performing Gaussian kernel PCA...');
[Y3, eigVector]=kPCA(data,d,'gaussian',para);
figure;hold on;
plot(Y3(1:end,3),Y3(1:end,24),'b*');
title('Gaussian kernel PCA');
drawnow;

%% pre-image reconstruction for Gaussian kernel PCA
disp('Performing kPCA pre-image reconstruction...');

for k = 1:d
    PI=zeros(size(data)); % pre-image
    eigVectorForReconstruct = eigVector( :, 1:k );
    Y3ForReconstruct = Y3( :, 1:k );
    for i = 1:size(data,1)
        PI(i,:)=kPCA_PreImage( Y3ForReconstruct', eigVectorForReconstruct, data, para, k)';
    end
    kpcaError = computeCenterError( PI, mu );
    disp( kpcaError )
end

%% pre-image reconstruction using custom optimization
disp('Performing kPCA pre-image reconstruction using custom objective...');

d = 5;
Nval = size( data, 1 );
gamma = zeros( 1, Nval );


% options.InitialPopulationRange = [-10;90];
z = mean(data); % initialization
nvars = size( z, 2 );
PI2=zeros(size(data));

for i = 1:size(data,1)

    for j = 1:Nval
        gamma(j) = eigVector( j, 1:d )*Y3( i, 1:d )';
    end
    
    [PI2(i,:), fval] = ga( @(z)projectionObjective( z, gamma, data, para ), nvars, [],[],[],[],[],[],[],[] );
end

kpcaError = computeCenterError( PI2, mu );
disp( kpcaError )
% figure;
% plot(PI(1:end,1),PI(1:end,2), 'b*');
% hold on;
% title('Reconstructed pre-images of Gaussian kPCA');

