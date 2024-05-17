function kvalue = kernelFunc( x, y, type, para)

if strcmp(type,'simple')
    kvalue = x*y';
end

if strcmp(type,'poly')
    kvalue = x*y' + 1;
    kvalue = kvalue^para;
end

if strcmp(type,'gaussian')
    kvalue = sum( (x - y).^2 );
    kvalue = exp( -kvalue / ( 2*para^2 ) );
end

end
