%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');

dataDistributionPrefix = "data_polynomial_order2_noise_gaussian";

N = 1000;
ndims = 2;
rng('default')  % For reproducibility

data = zeros(N, ndims);

data( :, 1 ) = linspace( -1, 1, N );

noiseAmpl = 0.2;

for dim = 2:ndims
    data( :, dim ) = data( :, 1 ).^2 + noiseAmpl * randn( N, 1 );
end

data = sign( data ).* ( abs( data ).^(1/5) );

figure;
plot( data(:, 1), data(:, 2), 'ro' )
xlabel('x');
ylabel('y');

Nval = N;
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
generateProjectionContours( data, Y1, nPts, nEigVecs, nlevels,...
    filenamePrefixPCAContours, withScatter )

% figure;
% % plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
% scatter( data(:, 1), data(:, 2), [], 80*( Y1(:, 1) + 2*abs( min( Y1(:, 1) ) ) ) )
% title('standard PCA');


%% polynomial kernel PCA
para = 5;
d = 2;
disp('Performing polynomial kernel PCA...');
[ Y2 ] = kPCA( data, d, 'poly', para );

nlevels = 50;
nPts = Nval;
nEigVecs = d;
withScatter = true;

filenamePrefixPolyPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAContours_deg" + string( para );
generateProjectionContours( data, Y2, nPts, nEigVecs, nlevels,...
    filenamePrefixPolyPCAContours, withScatter )
% figure;hold on;
% plot(Y2(1:end,1),Y2(1:end,2),'b*');
% title('polynomial kernel PCA');
% drawnow;

%% Gaussian kernel PCA
d = 20;
DIST = distanceMatrix(data);
DIST( DIST == 0 ) = inf;
DIST = min( DIST );
para = 2*mean( DIST );
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
generateProjectionContours( data, Y3, nPts, nEigVecs, nlevels,...
    filenamePrefixGaussianPCAContours, withScatter )