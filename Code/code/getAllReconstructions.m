function truncVals = getAllReconstructions( U, samples, k )
%% Compute reconstruction of the sample image using k eigenmodes
    szSample = size( samples, 1 );
    nSamples = size( samples, 2 );

    truncVals = zeros( szSample, nSamples );
    % Projection coefficients corresponding to the complete eigenspace
    
    for idx = 1:nSamples
        projVals = transpose( U )*samples( :, idx );
        
        % Reconstruction using truncated eigenspace
        for mode = 1:k
            truncVals( :, idx ) = truncVals( :, idx ) + ...
                projVals( mode )*U( :, mode );
        end
    end
end