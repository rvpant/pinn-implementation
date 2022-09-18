function [L, dL] = PINN_loss(theta, lam1, lam2)

num_para = length(theta);

global R_xt_train
num_Rdata = length(R_xt_train);
Dg = zeros(num_Rdata,1);
Jacobian_Dg = zeros(num_Rdata,num_para);

for i = 1 : num_Rdata
    x = R_xt_train(i,1);
    t = R_xt_train(i,2);
    k = randi(7);
    [NN, DNN, d_NN, d_DNN] = nn(x,t,theta, k);
    Dg(i) = sqrt(lam1/num_Rdata).*DNN;
    Jacobian_Dg(i,:) = sqrt(lam1/num_Rdata)*d_DNN;
end

global U_xt_train
global U_y_train
num_Udata = length(U_y_train);
g = zeros(num_Udata,1);
Jacobian_g = zeros(num_Udata,num_para);
for i = 1 : num_Udata
    x = U_xt_train(i,1);
    t = U_xt_train(i,2);
    y = U_y_train(i);
    k = randi(7);
    [NN, DNN, d_NN, d_DNN] = nn(x,t,theta, k);
    g(i) = sqrt(lam2/num_Udata)*(NN-y);
    Jacobian_g(i,:) = sqrt(lam2/num_Udata)*d_NN;
end
L = sum(g.^2) + sum(Dg.^2);
dL = 2*Jacobian_g'*g + 2*Jacobian_Dg'*Dg;
end