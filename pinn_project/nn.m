function [NN, DNN, d_NN, d_DNN] = nn(x,t,theta, k)
width = (length(theta)-1)/4;
a = theta(1:width);
w = theta((width+1):(2*width));
b = theta((2*width+1):(3*width));
c = theta((3*width+1):(4*width));
ther = theta(end);

% [y, dy, ddy, dddy] = activation_base(a*x + w*t +b)
[y, dy, ddy, dddy] = activation_rowdy(a*x + w*t +b, k); %can change to rowdy

NN = sum(c.*y);
DNN = sum(w.*c.*dy - ther.*(a.^2).*c.*ddy);
d_NN = [x.*c.*dy; t.*c.*dy; c.*dy; y; 0];
d_DNN = [x.*w.*c.*ddy - ther*2.*a.*c.*ddy + x.*(a.^2).*c.*dddy; ...
c.*dy + t.*w.*c.*ddy - ther*t.*(a.^2).*c.*dddy; ...
w.*c.*ddy - ther*(a.^2).*c.*dddy; ...
dy.*w - ther*(a.^2).*ddy; ...
-sum(c.*(a.^2).*ddy)];
end