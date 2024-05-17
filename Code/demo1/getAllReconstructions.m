function truncVals = getAllReconstructions( U, samples, k )
%% Compute reconstruction of all samples using k eigenmodes

    % Projection coefficients corresponding to the complete eigenspace
    projVals = transpose( U ) * samples;

    % Reconstruction using truncated eigenspace
    truncVals = U(:, 1 : k) * projVals( 1 : k, : );


end