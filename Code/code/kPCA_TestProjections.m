function projVals = kPCA_TestProjections( y, data, eigVector, type, para )

Nsamples = size( data, 1 );
ky = zeros( Nsamples, 1 );

oneMatrix = ones( Nsamples, Nsamples );
oneVector = ones( Nsamples, 1 );

K = kernel( data, type, para );

for i = 1:Nsamples
    ky( i, 1 ) = kernelFunc( y, data( i, : ), type, para );
end

kytilde = ky - oneMatrix * ky / Nsamples - K * oneVector / Nsamples...
    + oneMatrix * K * oneVector / Nsamples / Nsamples;

projVals = ( eigVector' ) * kytilde;

projVals = projVals';

end