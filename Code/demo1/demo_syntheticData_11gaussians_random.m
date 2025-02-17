%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');

dataDistributionPrefix = "data_3gaussians";

N = 11;
ndims = 10;
centerVals = zeros( N, ndims );
rng('default')  % For reproducibility

for dim = 1:ndims
    firstDim = -1 + 2 * rand(N,1);
    centerVals( :, dim ) = firstDim;
end

% centerVals = [ -0.5, -0.1; 0, 0.7; 0.5, 0.1 ];

repval = 100;
mu = repmat( centerVals, repval, 1);

nsigma = 1;
sigmaval = 0.0001;
Sigma = diag( ones( ndims, 1 )*sigmaval );
data = mvnrnd(mu,Sigma);

figure;

for distVal = 1:N
    colorval = [ 1, rand(), rand() ];

    plot( data( distVal: N : end, 1), data( distVal: N : end, 2),...
        '.', "Color", colorval )
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

pcaErrorTrain = zeros( nsigma, d );

for k = 1:d
    truncVals = getAllReconstructions( pcaEigVector, data', k )';
    
    pcaErrorTrain( 1, k ) = computeCenterError( truncVals, mu );
    disp( pcaErrorTrain( 1, k ) )
end
% filenameProjections = "../Plots/" + dataDistributionPrefix +...
%     "PCAProjections";
% plotProjections( Y1, filenameProjections, N )
% figure;
% % plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
% scatter( data(:, 1), data(:, 2), [], 80*( Y1(:, 1) + 2*abs( min( Y1(:, 1) ) ) ) )
% title('standard PCA');

%% Gaussian kernel PCA
d = 20;
DIST = distanceMatrix(data);
DIST( DIST == 0 ) = inf;
DIST = min( DIST );
para = 5 * mean( DIST );
% para = 2*sigmaval;
disp( 'Performing Gaussian kernel PCA...' );
[Y3, eigVector] = kPCA( data, d, 'gaussian', para );

%% Create Test Vectors

repval = 30;
muTest = repmat( centerVals, repval, 1);

Sigma = diag( ones( ndims, 1 )*sigmaval );
dataTest = mvnrnd( muTest, Sigma );

Nval = N * repval;

%% pre-image reconstruction for Standard PCA (Test Data)
disp('Performing PCA Test pre-image reconstruction...');
d = min( Nval, ndims );
pcaErrorTest = zeros( nsigma, d );

for k = 1:d
    truncVals = getAllReconstructions( pcaEigVector, dataTest', k )';
    
    pcaErrorTest( 1, k ) = computeCenterError( truncVals, muTest );
    disp( pcaErrorTest( 1, k ) )
end

%% pre-image reconstruction for Gaussian kernel PCA (Training Data)
disp('Performing kPCA pre-image reconstruction...');
d = min( Nval, ndims );

kpcaErrorTrainFixedPoint = zeros( nsigma, d );

for k = 1:d
    PI=zeros( size(data) ); % pre-image
    eigVectorForReconstruct = eigVector( :, 1:k );
    Y3ForReconstruct = Y3( :, 1:k );
    for i = 1:size(data, 1)
        PI(i,:) = kPCA_PreImage( Y3ForReconstruct(i, :)', eigVectorForReconstruct, data, para, k)';
    end

    kpcaErrorTrainFixedPoint( 1, k ) = computeCenterError( PI, mu );
    disp( kpcaErrorTrainFixedPoint( 1, k ) )
end

%% pre-image reconstruction for Gaussian kernel PCA (Test Data)
disp('Performing kPCA pre-image reconstruction...');
d = min( Nval, ndims );

outputvalsIterations = zeros( size( dataTest, 1 ), 1 );
kpcaErrorTestFixedPoint = zeros( nsigma, d );

for k = 2:d
    PI=zeros( size(dataTest) ); % pre-image
    eigVectorForReconstruct = eigVector( :, 1:k );
    for i = 1:size( dataTest, 1 )

        Y3Test = kPCA_TestProjections( dataTest( i, : ), data, eigVector, 'gaussian', para );
        Y3ForReconstruct = Y3Test( :, 1:k );

        [PI(i,:), niter] = kPCA_PreImage( Y3ForReconstruct', eigVectorForReconstruct, data, para, k);
        outputvalsIterations( i, 1 ) = niter;
    
    end

    kpcaErrorTestFixedPoint( 1, k ) = computeCenterError( PI, muTest );
    disp( kpcaErrorTestFixedPoint(1, k) )
end

%% pre-image reconstruction for Gaussian kernel PCA (Test Data, Custom Objective)
% Uses gradient based optimization algorithms for solution

d = min( Nval, ndims );
kpcaErrorTestCustom = zeros( nsigma, d );

for k = 2:d
    meanval = mean( data );
    optimInitPoint = meanval;
    eigVectorForReconstruct = eigVector( :, 1:k );
    
    reconVals = zeros( size(dataTest) );
    
    rng(14,'twister')
    
    problem.solver = 'fmincon';
    problem.x0 = optimInitPoint;
    
    algOptions = ["sqp", "interior-point"];
    nAlgOptions = size( algOptions, 2 );
    
    outputvalsFuncEvals = zeros( nAlgOptions + 1, size( dataTest, 1 ));
    
    for i = 1:size( dataTest, 1 )
    
        Y3Test = kPCA_TestProjections( dataTest( i, : ), data, eigVector, 'gaussian', para );
        Y3ForReconstruct = Y3Test( :, 1:k );
    
        problem.objective = @(z)projectionObjective( z, Y3ForReconstruct', eigVectorForReconstruct, data, k, para );
        
        j = 1;
        for algOption = algOptions
    
            options = optimoptions( "fmincon", "Algorithm", algOption );
            problem.options = options;
    
            [reconVals(i, :), ~, ~, output] = fmincon( problem );
            outputvalsFuncEvals( j, i ) = output.funcCount;
    
            j = j + 1;
        end
    end
    
    kpcaErrorTestCustom( 1, k ) = computeCenterError( reconVals, muTest );
    disp( kpcaErrorTestCustom(1, k) )
end
% outputvalsFuncEvals( end, : ) = outputvalsIterations';

% filenameperf = "../Plots/" + dataDistributionPrefix + "FuncEvalsOptim.eps";
% plotPerfData( outputvalsFuncEvals, filenameperf );
