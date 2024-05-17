function generateProjectionContours( data, Yvals, nPts, nEigVecs, nlevels,...
    filenamePrefix, withScatter )

for idxIter = 1:nEigVecs

    [X, Y] = ndgrid( linspace(min( data(:, 1) ), max( data( :, 1 ) ), nPts),...
        linspace( min( data(:, 2) ), max( data(:, 2) ), 150) );
    Z = griddata( data( :, 1 ), data( :, 2 ), Yvals( :, idxIter ), X, Y, 'cubic');
    
    filenameContours = filenamePrefix + "Eig" + string( idxIter ) + ".eps";
    plotContours( X, Y, Z, data, Yvals, idxIter, filenameContours,...
        withScatter, nlevels )

end
 
end