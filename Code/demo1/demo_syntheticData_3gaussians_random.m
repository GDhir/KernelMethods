%%  This is a demo showing how to use this toolbox
%   This experiment is performed on synthetic data

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

clear;clc;close all;

addpath('../code');

dataDistributionPrefix = "data_3gaussians";

N = 3;
ndims = 2;
mu = zeros( N, ndims );
rng('default')  % For reproducibility

% for dim = 1:ndims
%     firstDim = rand(N,1);
%     mu( :, dim ) = firstDim;
% end

centerVals = [ -0.5, -0.1; 0, 0.7; 0.5, 0.1 ];

repval = 300;
mu = repmat( centerVals, repval, 1);
sigmaval = 0.01;
Sigma = diag( ones( ndims, 1 )*sigmaval );
data = mvnrnd(mu,Sigma);

% figure;
filename = "../Plots/" + dataDistributionPrefix + "InitialData.eps";
plotDataDistribution( data, N, filename )

% for distVal = 1:N
%     colorval = [ 1, rand(), rand() ];
% 
%     plot( data( distVal: N : end, 1), data( distVal: N : end, 2),...
%         '.', "Color", colorval )
%     hold on;
% end

% plot( data(2:2:end, 1), data(2:2:end, 2), 'o', "Color","b" )
% xlabel('x');
% ylabel('y');

Nval = N * repval;

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

% filenameProjections = "../Plots/" + dataDistributionPrefix +...
%     "PCAProjections";
% plotProjections( Y1, filenameProjections, N )
% figure;
% % plot( Y1( 1:end, 1 ), Y1( 1:end, 2 ), 'b*');
% scatter( data(:, 1), data(:, 2), [], 80*( Y1(:, 1) + 2*abs( min( Y1(:, 1) ) ) ) )
% title('standard PCA');

%% polynomial kernel PCA
para = 3;
d = 2;
disp('Performing polynomial kernel PCA...');
[ Y2 ] = kPCA( data, d, 'poly', para );

nlevels = 20;
nPts = Nval;
nEigVecs = d;
withScatter = true;

filenamePrefixPolyPCAContours = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAContours_deg" + string( para );
generateProjectionContours( data, Y2, nPts, nEigVecs, nlevels,...
    filenamePrefixPolyPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "PolyPCAProjections_deg" + string( para );
% plotProjections( Y2, filenameProjections, N )
% figure;hold on;
% plot(Y2(1:end,1),Y2(1:end,2),'b*');
% title('polynomial kernel PCA');
% drawnow;

%% Gaussian kernel PCA
d = 20;
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

nlevels = 40;
nPts = Nval;
nEigVecs = 4;
withScatter = true;

filenamePrefixGaussianPCAContours = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAContours";
generateProjectionContours( data, Y3, nPts, nEigVecs, nlevels,...
    filenamePrefixGaussianPCAContours, withScatter )

filenameProjections = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAProjections_sigma" + string( para );
% plotProjections( Y3, filenameProjections, N, 1, 2 )

%% Create Test Vectors

repval = 30;
muTest = repmat( centerVals, repval, 1);
sigmaval = 0.01;
Sigma = diag( ones( ndims, 1 )*sigmaval );
dataTest = mvnrnd( muTest, Sigma );

% figure;

for distVal = 1:N
    colorval = [ 1, rand(), rand() ];

    % plot( dataTest( distVal: N : end, 1), dataTest( distVal: N : end, 2),...
    %     '.', "Color", colorval )
    % hold on;
end

filename = "../Plots/" + dataDistributionPrefix + "InitialTestData.eps";
plotDataDistribution( dataTest, N, filename )
% plot( data(2:2:end, 1), data(2:2:end, 2), 'o', "Color","b" )
% xlabel('x');
% ylabel('y');

Nval = N * repval;

%% pre-image reconstruction for Standard PCA (Test Data)
disp('Performing PCA Test pre-image reconstruction...');
d = min( Nval, ndims );
pcaErrorTest = zeros( 1, d );

for k = 1:1
    truncVals = getAllReconstructions( pcaEigVector, dataTest', k )';
    
    pcaErrorTest( 1, k ) = computeCenterError( truncVals, muTest );
    disp( pcaErrorTest( 1, k ) )
end

filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "PCAReconstructionsTestData.eps";

plotReconstructions( truncVals, mu, filenameReconstructions, N, true, true )

%% Plot reconstruction objective for different training vectors
k = 5;

for i = 201

    y = Y3( i, 1:k )';
    % valsRange = mean( data );
    valsRange = [ -1, 1 ];
    
    filenameObjective = "../Plots/" + dataDistributionPrefix +...
        "objectivePlot_i=" + string(i) + ".eps";

    plotReconstructionObjective( y, data, eigVector, k, para, valsRange,...
        filenameObjective );

end

%% Plot reconstruction objective for different test vectors
k = 2;

for i = 1:5:size(dataTest, 1)

    Y3Test = kPCA_TestProjections( dataTest( i, : ), data, eigVector, 'gaussian', para );
    
    y = Y3Test(1, 1:k)';
    % valsRange = mean( data );
    valsRange = [ -1, 1 ];
    
    filenameObjective = "../Plots/" + dataDistributionPrefix +...
        "objectiveTestDataPlot_i=" + string(i) + ".eps";

    plotReconstructionObjective( y, data, eigVector, k, para, valsRange,...
        filenameObjective );

end

%% pre-image reconstruction for Gaussian kernel PCA (Training Data)
disp('Performing kPCA pre-image reconstruction...');
d = 5;

for k = 2
    PI=zeros( size(data) ); % pre-image
    eigVectorForReconstruct = eigVector( :, 1:k );
    Y3ForReconstruct = Y3( :, 1:k );
    for i = 1:size(data, 1)
        PI(i,:) = kPCA_PreImage( Y3ForReconstruct(i, :)', eigVectorForReconstruct, data, para, k)';
    end

    filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAReconstructions.eps";
    plotReconstructions( PI, mu, filenameReconstructions, N, true, true )

    kpcaError = computeCenterError( PI, mu );
    disp( kpcaError )
end

%% pre-image reconstruction for Gaussian kernel PCA (Test Data)
disp('Performing kPCA pre-image reconstruction...');
d = 5;

outputvalsIterations = zeros( size( dataTest, 1 ), 1 );

for k = 1
    PI=zeros( size(dataTest) ); % pre-image
    eigVectorForReconstruct = eigVector( :, 1:k );
    for i = 1:size( dataTest, 1 )

        Y3Test = kPCA_TestProjections( dataTest( i, : ), data, eigVector, 'gaussian', para );
        Y3ForReconstruct = Y3Test( :, 1:k );

        [PI(i,:), niter] = kPCA_PreImage( Y3ForReconstruct', eigVectorForReconstruct, data, para, k);
        outputvalsIterations( i, 1 ) = niter;
    
    end

    filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAReconstructionsTestData.eps";
    plotReconstructions( PI, muTest, filenameReconstructions, N, true, true )

    kpcaError = computeCenterError( PI, muTest );
    disp( kpcaError )
end

%% pre-image reconstruction for Gaussian kernel PCA (Test Data, Custom Objective)
% Uses global search with multiple start points 
% start points can be custom start points or default set by the solver

k = 2;
meanval = mean( data );
optimInitPoint = meanval + rand( 1, ndims );
eigVectorForReconstruct = eigVector( :, 1:k );

reconVals = zeros( size(dataTest) );

npts = 64;
startPts = zeros( npts, ndims );
rng(14,'twister')

for i = 1:npts
    startPts(i, :) = meanval + rand( 1, ndims );
end
tpoints = CustomStartPointSet(startPts);

for i = 1:10:size( dataTest, 1 )

    Y3Test = kPCA_TestProjections( dataTest( i, : ), data, eigVector, 'gaussian', para );
    Y3ForReconstruct = Y3Test( :, 1:k );
    ms = MultiStart;
    

    problem = createOptimProblem('fmincon',...
        'objective',@(z)projectionObjective( z, Y3ForReconstruct', eigVectorForReconstruct, data, k, para ),...
        'x0', optimInitPoint,...
        'lb',[-3,-3],'ub',[3,3]);
    
    [reconVals(i, :), ~] = fmincon(problem);

    gs = GlobalSearch('Display','off','StartPointsToRun','bounds', 'PlotFcn','gsplotbestf',...
        'NumTrialPoints', 10000);

    [reconVals(i, :), fval, flag, outpt, allmins] = run(gs, problem);
end

filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAReconstructions_MATLABSQPGlobal_TestData.eps";
    plotReconstructions( reconVals, muTest, filenameReconstructions, N, true, true )

kpcaError = computeCenterError( reconVals, muTest );
disp( kpcaError )


%% pre-image reconstruction for Gaussian kernel PCA (Test Data, Custom Objective)
% Uses gradient based optimization algorithms for solution

k = 2;
meanval = mean( data );
optimInitPoint = meanval;
eigVectorForReconstruct = eigVector( :, 1:k );

reconVals = zeros( size(dataTest) );

npts = 64;
startPts = zeros( npts, ndims );
rng(14,'twister')

for i = 1:npts
    startPts(i, :) = meanval + rand( 1, ndims );
end
tpoints = CustomStartPointSet(startPts);

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

filenameReconstructions = "../Plots/" + dataDistributionPrefix +...
    "GaussianPCAReconstructions_MATLABSQPGlobal_TestData.eps";
    plotReconstructions( reconVals, muTest, filenameReconstructions, N, true, true )

kpcaError = computeCenterError( reconVals, muTest );
disp( kpcaError )

outputvalsFuncEvals( end, : ) = outputvalsIterations';

filenameperf = "../Plots/" + dataDistributionPrefix + "FuncEvalsOptim.eps";
plotPerfData( outputvalsFuncEvals, filenameperf );
