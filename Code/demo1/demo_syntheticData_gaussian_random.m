%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');

dataDistributionPrefix = "data_gaussian";

N = 2;
ndims = 2;
mu = zeros( N, ndims );
rng('default')  % For reproducibility

for dim = 1:ndims
    firstDim = rand(N,1);
    mu( :, dim ) = firstDim;
end

repval = 1000;
mu = repmat(mu, repval, 1);
sigmaval = 0.03;
Sigma = diag( ones( ndims, 1 )*sigmaval );
data = mvnrnd(mu,Sigma);

figure;

for distVal = 1:N
    colorval = [ 1, rand(), rand() ];

    plot( data( distVal: N : end, 1), data( distVal: N : end, 2),...
        'o', "Color", colorval )
    hold on;
end

% plot( data(2:2:end, 1), data(2:2:end, 2), 'o', "Color","b" )
xlabel('x');
ylabel('y');

Nval = N * repval;

%% standard PCA
d = min( Nval, ndims );
disp('Performing standard PCA...');
[Y1 , pcaEigVector, pcaEigValue] = PCA( data, d );

[X, Y] = ndgrid( linspace(min( data(:, 1) ), max( data( :, 1 ) ), 150),...
    linspace( min( data(:, 2) ), max( data(:, 2) ), 150) );
Z = griddata( data( :, 1 ), data( :, 2 ), Y1( :, 1 ), X, Y, 'cubic');

nlevels = 10;
nPts = 200;
nEigVecs = 2;
withScatter = true;

filenamePrefixPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PCAContours";
% generateProjectionContours( data, Y1, nPts, nEigVecs, nlevels,...
%     filenamePrefixPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "PCAProjections";
plotProjections( Y1, filenameProjections, N )
% figure;
% % plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
% scatter( data(:, 1), data(:, 2), [], 80*( Y1(:, 1) + 2*abs( min( Y1(:, 1) ) ) ) )
% title('standard PCA');


%% polynomial kernel PCA
para = 5;
d = 2;
disp('Performing polynomial kernel PCA...');
[ Y2 ] = kPCA( data, d, 'poly', para );

nlevels = 20;
nPts = Nval;
nEigVecs = d;
withScatter = false;

filenamePrefixPolyPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAContours_deg" + string( para );
% generateProjectionContours( data, Y2, nPts, nEigVecs, nlevels,...
%     filenamePrefixPolyPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAProjections_deg" + string( para );
plotProjections( Y2, filenameProjections, N )
% figure;hold on;
% plot(Y2(1:end,1),Y2(1:end,2),'b*');
% title('polynomial kernel PCA');
% drawnow;

%% Gaussian kernel PCA
d = 2;
DIST = distanceMatrix(data);
DIST( DIST == 0 ) = inf;
DIST = min( DIST );
para = 5 * mean( DIST );
% para = 100000*sigmaval;
disp( 'Performing Gaussian kernel PCA...' );
[Y3, eigVector] = kPCA( data, d, 'gaussian', para );
% figure;hold on;
% plot(Y3(1:end,3),Y3(1:end,24),'b*');
% title('Gaussian kernel PCA');
% drawnow;

nlevels = 20;
nPts = Nval;
nEigVecs = 4;
withScatter = false;

filenamePrefixGaussianPCAContours = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAContours_sigma" + string( para );
% generateProjectionContours( data, Y3, nPts, nEigVecs, nlevels,...
%     filenamePrefixGaussianPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAProjections_sigma" + string( para );
plotProjections( Y3, filenameProjections, N, 1, 2 )