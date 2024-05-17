function error = computeCenterError( samples, meanvals )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    error = sum( (samples - meanvals).^2, 2 );
    error = sqrt( mean( error ) );

end