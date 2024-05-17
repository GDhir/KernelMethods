%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');

%% Generate Data
dataDistributionPrefix = "data_halfCircle";

N = 1000;
ndims = 2;
rng('default')  % For reproducibility

data = zeros(N, ndims);
originalDenoisedData = zeros(N, ndims);

noiseAmpl = 0.1;

th = linspace( pi, 0, N);
R = 1;  %or whatever radius you want
x = R*cos(th);
y = R*sin(th);

data( :, 1 ) = x' + noiseAmpl * randn( N, 1 );
data( :, 2 ) = y' + noiseAmpl * randn( N, 1 );

originalDenoisedData( :, 1 ) = x';
originalDenoisedData( :, 2 ) = y';

% figure;
filename = "../Plots/" + dataDistributionPrefix + "InitialData.eps";
plotNoiseData( data, filename )
Nval = N;

%% standard PCA
d = min( Nval, ndims );
disp('Performing standard PCA...');
[Y1 , pcaEigVector, pcaEigValue] = PCA( data, d );

nlevels = 10;
nPts = 200;
nEigVecs = 2;
withScatter = true;

filenamePrefixPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PCAContours";
generateProjectionContours( data, Y1, nPts, nEigVecs, nlevels,...
    filenamePrefixPCAContours, withScatter )

k = 1;
reconValsPCA = mean( data )' + getAllReconstructions( pcaEigVector, data', k );

filename = "../Plots/" + dataDistributionPrefix + "ReconstructedDataPCA_k=" +...
    string(k) + ".eps";

plotNoiseReconstructionData( data, reconValsPCA',originalDenoisedData, true, filename )

pcaError = computeCenterError( reconValsPCA', originalDenoisedData );
disp( pcaError )

%% polynomial kernel PCA
paraPoly = 2;
d = 2;
disp('Performing polynomial kernel PCA...');
[ Y2, eigVectorPoly, ~ ] = kPCA( data, d, 'poly', paraPoly );

nlevels = 20;
nPts = Nval;
nEigVecs = d;
withScatter = true;

filenamePrefixPolyPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAContours_deg" + string( paraPoly );
generateProjectionContours( data, Y2, nPts, nEigVecs, nlevels,...
    filenamePrefixPolyPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAProjections_deg" + string( paraPoly );
% plotProjections( Y2, filenameProjections, N )
% figure;hold on;
% plot(Y2(1:end,1),Y2(1:end,2),'b*');
% title('polynomial kernel PCA');
% drawnow;

%% Plot reconstruction objective for different training vectors (Polynomial Kernel)
% k = 1;
% 
% for i = 671
% 
%     y = Y2( i, 1:k )';
%     % valsRange = mean( data );
%     valsRange = [ -3, 3 ];
% 
%     filenameObjective = "../Plots/" + dataDistributionPrefix + "PolynomialKernel" +...
%         "objectivePlot_i=" + string(i) + ".eps";
% 
%     plotReconstructionObjective( y, data, eigVectorPoly, k, paraPoly, valsRange,...
%         filenameObjective, 'poly' );
% 
% end

%% pre-image reconstruction for Gaussian kernel PCA (Training Data)
disp('Performing kPCA pre-image reconstruction...');
d = 2;

for k = 1
    reconstructedDataGaussianPCA = zeros( size(data) ); % pre-image
    eigVectorForReconstruct = eigVectorGaussian( :, 1:k );
    Y3ForReconstruct = Y3( :, 1:k );
    for i = 1:size(data, 1)
        reconstructedDataGaussianPCA(i,:) = kPCA_PreImage( Y3ForReconstruct(i, :)',...
            eigVectorForReconstruct, data, paraGaussian, k)';
    end

    filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAReconstructions.eps";
    plotNoiseReconstructionData( data, reconstructedDataGaussianPCA, originalDenoisedData, true, filenameReconstructions )

    kpcaError = computeCenterError( reconstructedDataGaussianPCA, originalDenoisedData );
    disp( kpcaError )
end

%% pre-image reconstruction for Polynomial kernel PCA (Training Data, Custom Objective)
% Uses gradient based optimization algorithms for solution

k = 2;
meanval = mean( data );
optimInitPoint = meanval;
eigVectorForReconstruct = eigVectorPoly( :, 1:k );
ktype = 'poly';

reconValsPolynomialPCA_MATLABSolver = zeros( size(data) );

npts = 64;
startPts = zeros( npts, ndims );
rng(14,'twister')

for i = 1:npts
    startPts(i, :) = meanval + rand( 1, ndims );
end
tpoints = CustomStartPointSet(startPts);

problem.solver = 'fmincon';
problem.x0 = optimInitPoint;


% algOptions = ["sqp", "interior-point"];
algOptions = ["interior-point"];
nAlgOptions = size( algOptions, 2 );

outputvalsFuncEvals = zeros( nAlgOptions + 1, size( data, 1 ));

for i = 1:size( data, 1 )

    disp(i)
    Y2ForReconstruct = Y2( i, 1:k );
    
    problem.objective = @(z)projectionObjective( z, Y2ForReconstruct',...
        eigVectorForReconstruct, data, k, paraPoly, ktype );
    
    j = 1;
    for algOption = algOptions

        options = optimoptions( "fmincon", "Algorithm", algOption );
        problem.options = options;

        [reconValsPolyPCA_MATLABSolver(i, :), ~, ~, output] = fmincon( problem );
        outputvalsFuncEvals( j, i ) = output.funcCount;

        j = j + 1;
    end
end

filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAReconstructions_MATLABSQPGlobal_TrainData.eps";
plotNoiseReconstructionData( data, reconValsPolynomialPCA_MATLABSolver,...
    originalDenoisedData, true, filenameReconstructions )

kpcaError = computeCenterError( reconValsPolyPCA_MATLABSolver, originalDenoisedData );
disp( kpcaError )

% outputvalsFuncEvals( end, : ) = outputvalsIterations';

% filenameperf = "../Plots/" + dataDistributionPrefix + "FuncEvalsOptim.eps";
% plotPerfData( outputvalsFuncEvals, filenameperf );

%% Gaussian kernel PCA
d = 2;
DIST = distanceMatrix(data);
DIST( DIST == 0 ) = inf;
DIST = min( DIST );

% paraGaussianVals = [ 5 * mean( DIST ), 10 * mean( DIST ), 15 * mean( DIST ),...
%     20 * mean( DIST ), 30 * mean( DIST ), 40 * mean( DIST ), 50 * mean( DIST ),...
%     70 * mean( DIST ), 100 * mean( DIST )];

paraGaussianVals = linspace(5, 100, 20 ) * mean ( DIST );
% para = 100000*sigmaval;

kpcaErrorVals = zeros( size( paraGaussianVals ) );
idxIter = 1;

for paraGaussian = paraGaussianVals
    disp( 'Performing Gaussian kernel PCA...' );
    [Y3, eigVectorGaussian] = kPCA( data, d, 'gaussian', paraGaussian );
    % figure;hold on;
    % plot(Y3(1:end,3),Y3(1:end,24),'b*');
    % title('Gaussian kernel PCA');
    % drawnow;
    
    nlevels = 40;
    nPts = Nval;
    nEigVecs = 2;
    withScatter = true;
    
    filenamePrefixGaussianPCAContours = "../Plots/" + dataDistributionPrefix + "_paraGaussian=" +...
        string(paraGaussian) + "d=" + string(d) + "GaussianPCAContours";
    generateProjectionContours( data, Y3, nPts, nEigVecs, nlevels,...
        filenamePrefixGaussianPCAContours, withScatter )
    
    % Plot reconstruction objective for different training vectors (Gaussian Kernel)
    % k = 2;
    % 
    % for i = 1:100:900
    % 
    %     y = Y3( i, 1:k )';
    %     % valsRange = mean( data );
    %     valsRange = [ -3, 3 ];
    % 
    %     filenameObjective = "../Plots/" + dataDistributionPrefix + "GaussianKernel" +...
    %         "objectivePlot_i=" + string(i) + ".eps";
    % 
    %     plotReconstructionObjective( y, data, eigVectorGaussian, k, paraGaussian, valsRange,...
    %         filenameObjective, 'gaussian' );
    % 
    % end
    
    % pre-image reconstruction for Gaussian kernel PCA (Training Data, Custom Objective)
    % Uses gradient based optimization algorithms for solution
    
    k = 2;
    meanval = mean( data );
    optimInitPoint = meanval;
    eigVectorForReconstruct = eigVectorGaussian( :, 1:k );
    ktype = "gaussian";
    
    reconValsGaussianPCA_MATLABSolver = zeros( size(data) );
    
    % For multiple starting points with optimization
    % npts = 64;
    % startPts = zeros( npts, ndims );
    % rng(14,'twister')
    % 
    % for i = 1:npts
    %     startPts(i, :) = meanval + rand( 1, ndims );
    % end
    % tpoints = CustomStartPointSet(startPts);
    
    problem.solver = 'fmincon';
    problem.x0 = optimInitPoint;
    
    % algOptions = ["sqp", "interior-point"];
    algOptions = ["interior-point"];
    nAlgOptions = size( algOptions, 2 );
    
    outputvalsFuncEvals = zeros( nAlgOptions + 1, size( data, 1 ));
    
    for i = 1:size( data, 1 )
    
        % disp(i)
        Y3ForReconstruct = Y3( i, 1:k );
        
        problem.objective = @(z)projectionObjective( z, Y3ForReconstruct',...
            eigVectorForReconstruct, data, k, paraGaussian, ktype );
        
        j = 1;
        for algOption = algOptions
    
            options = optimoptions( "fmincon", "Algorithm", algOption );
            problem.options = options;
            problem.ub = [3, 1];
            problem.lb = [-3, 0];
    
            [reconValsGaussianPCA_MATLABSolver(i, :), ~, ~, output] = fmincon( problem );
            outputvalsFuncEvals( j, i ) = output.funcCount;
    
            j = j + 1;
        end
    end
    
    filenameReconstructions = "../Plots/" + dataDistributionPrefix + "_paraGaussian=" +...
        string(paraGaussian) + "k=" + string(k) +...
        "GaussianPCAReconstructions_MATLABSQPGlobal_TrainData.eps";
    
    plotNoiseReconstructionData( data, reconValsGaussianPCA_MATLABSolver,...
        originalDenoisedData, true, filenameReconstructions )
    
    kpcaErrorVals( idxIter ) = computeCenterError( reconValsGaussianPCA_MATLABSolver, originalDenoisedData );
    idxIter = idxIter + 1;
        
    % outputvalsFuncEvals( end, : ) = outputvalsIterations';
    
    % filenameperf = "../Plots/" + dataDistributionPrefix + "FuncEvalsOptim.eps";
    % plotPerfData( outputvalsFuncEvals, filenameperf );

end

filename = "../Plots/" + dataDistributionPrefix + "_paraGaussian=" +...
        string(paraGaussian) + "k=" + string(k) +...
        "GaussianPCAParameterErrorVariation_MATLABSQPGlobal_TrainData.eps";
plotKPCAParaError( paraGaussianVals(1:9), kpcaErrorVals(1:9), filename )