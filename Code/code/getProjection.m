function truncVals = getProjection( U, sample, k )
%% Compute reconstruction of the sample image using k eigenmodes
    szSample = size( sample, 1 );

    % Projection coefficients corresponding to the complete eigenspace
    projVals = transpose( U )*sample;

    truncVals = zeros( szSample );

    % Reconstruction using truncated eigenspace
    for idx = 1:k

        truncVals = truncVals + projVals(idx)*U( :, idx );

    end

end