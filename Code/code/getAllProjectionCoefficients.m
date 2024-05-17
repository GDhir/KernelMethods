function allProjVals = getAllProjectionCoefficients( U, samples )
%% Compute projection coefficient matrix of the sample images using eigenmodes

    % Projection coefficient Matrix corresponding to the complete eigenspace

    nSamples = size(samples, 2);
    eigShape = size( U, 2 );

    allProjVals = zeros( eigShape, nSamples );

    for idx = 1:nSamples
        allProjVals( :, idx ) = getSampleProjectionCoefficients( U, samples(:, idx) );
    end

end