function y = projectionObjective2D( z1, z2, gamma, X, para, ktype )

    z = [z1, z2];

    y = -kernelFunc( z, z, ktype, para );
    N = size(X, 1);
    % y = 0;

    for i = 1:N
        y = y + 2 * gamma(i)*kernelFunc( z, X( i, : ), ktype, para );
    end
end