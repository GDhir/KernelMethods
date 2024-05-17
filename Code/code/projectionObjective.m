function objVal = projectionObjective( z, y, eigVector, X, d, para, ktype )

    Nsamples = size(X, 1);
    gamma = zeros( Nsamples, 1 );

    for i = 1:Nsamples
        gamma(i) = eigVector(i, 1:d) * y;
    end

    objVal = kernelFunc( z, z, ktype, para );

    for i = 1:Nsamples
        objVal = objVal - 2*gamma(i)*kernelFunc( z, X( i, : ), ktype, para );
    end
end