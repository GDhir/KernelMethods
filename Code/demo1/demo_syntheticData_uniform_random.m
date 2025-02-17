%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');
N = 1000;
ndims = 2;
rng('default')  % For reproducibility

data = zeros(N, ndims);

for dim = 1:ndims
    firstDim = rand( N, 1 );
    data( :, dim ) = firstDim;
end

figure;
plot( data(:, 1), data(:, 2), 'ro' )
xlabel('x');
ylabel('y');

%% standard PCA
d = min( N, ndims );
disp('Performing standard PCA...');
[Y1 , pcaEigVector, pcaEigValue] = PCA( data, d );

[X, Y] = ndgrid( linspace(min( data(:, 1) ), max( data( :, 1 ) ), 150),...
    linspace( min( data(:, 2) ), max( data(:, 2) ), 150) );
Z = griddata( data( :, 1 ), data( :, 2 ), Y1( :, 1 ), X, Y, 'cubic');

filenamePCAContours = "../Plots/PCAContoursEig1.eps";
plotContours( X, Y, Z, data, Y1, 1, filenamePCAContours )

[X, Y] = ndgrid( linspace(min( data(:, 1) ), max( data( :, 1 ) ), 150),...
    linspace( min( data(:, 2) ), max( data(:, 2) ), 150) );
Z = griddata( data( :, 1 ), data( :, 2 ), Y1( :, 2 ), X, Y, 'cubic');

filenamePCAContours = "../Plots/PCAContoursEig2.eps";
plotContours( X, Y, Z, data, Y1, 2, filenamePCAContours )

% figure;
% % plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
% scatter( data(:, 1), data(:, 2), [], 80*( Y1(:, 1) + 2*abs( min( Y1(:, 1) ) ) ) )
% title('standard PCA');


%% polynomial kernel PCA
para = 3;
d = 3;
disp('Performing polynomial kernel PCA...');
[ Y2 ] = kPCA( data, d, 'poly', para );

nlevels = 50;
nPts = 200;
nEigVecs = 3;
withScatter = false;

filenamePrefixPolyPCAContours = "../Plots/PolyPCAContours_deg" + string( para );
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
nPts = N;
nEigVecs = 4;
withScatter = false;

filenamePrefixGaussianPCAContours = "../Plots/GaussianPCAContours";
generateProjectionContours( data, Y3, nPts, nEigVecs, nlevels,...
    filenamePrefixGaussianPCAContours, withScatter )