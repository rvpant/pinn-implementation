function [f, df, ddf, dddf] = activation_rowdy(x, k)

n = 0.1;
if k == 1
    [f, df, ddf, dddf] = activation_base(x);
else
    [f, df, ddf, dddf] = activation_base(x);
    for j=2:k
       f = f + n*sin((j-1)*n*x); 
       df = df + (j-1)*n^2*cos((j-1)*n*x);
       ddf = ddf - (j-1)^2*n^3*sin((j-1)*n*x);
       dddf = dddf - (j-1)^3*n^4*cos((j-1)*n*x);
    end
end
end