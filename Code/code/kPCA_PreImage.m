%   y: dimensionanlity-reduced data
%	eigVector: eigen-vector obtained in kPCA
%   X: data matrix
%   para: parameter of Gaussian kernel
%	z: pre-image of y

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its 
%   Applications in Face Recognition and Active Shape Models. 
%   arXiv:1207.3538 [cs.CV], 2012. 

function [z, niter] = kPCA_PreImage( y, eigVector, X, para, d )

Nsamples = size(X, 1);
dims = size(X, 2);
% d=max(size(y));

ncheckptIter = 10000;

gamma=zeros(1, Nsamples);
for i = 1:Nsamples
    gamma(i) = eigVector(i, 1:d) * y;
end

z = mean(X)' + abs( rand( dims, 1 ) );
doubleval = 1;

iter = 0;

while 1
    pre_z = z;
    xx = bsxfun(@minus,X',z);
    xx = xx.^2;
    xx = -sum(xx) / (2*para.^2);
    xx = exp(xx);
    xx = xx.*gamma;
    
    sumval = sum(xx);

    if (abs(sumval) > 1e-5)
        z = xx * X / sumval;
        z = z';
    else
        z = abs( mean(X)' ) + abs( rand( dims, 1 ) );
        doubleval = doubleval * 2;
    end

    error = norm(pre_z - z) / norm(z);
    % disp(error)    
    if error < 1e-6
        break;
    end
    iter = iter + 1;

    if( rem( iter, ncheckptIter ) == 0 )
        disp(error)   
        % z = abs( mean(X)' ) + doubleval * abs( rand( dims, 1 ) );
        % doubleval = doubleval * 2;
        % z = z + rand( dims, 1 ); % initialization
    end

end

z = z';

niter = iter;
