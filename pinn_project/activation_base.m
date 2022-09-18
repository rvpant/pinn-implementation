function [f, df, ddf, dddf] = activation_base(x)

f = tanh(x);
df = 1-tanh(x).^2;
ddf = -2*tanh(x).^3;
dddf = 6*tanh(x).^4 - 6*tanh(x).^2;

end