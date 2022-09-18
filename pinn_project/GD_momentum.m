% Perform Nesterov's Accelerated Gradient Descent with constant step size
%
% INTPUT: handler, return L(Theta;lam1,lam2) and its derivatives
% theta0: the starting point
% stepSize: the stepsize
% niter: the maximum number of iterations
% RETURN: Loss_traj: the loss trajectory w.r.t. iterations
% Thetak : the approximation obtained after niter-th iterations
function [losses, theta_k] = GD_momentum(obj,theta_0,lam1,lam2,s,maxiter)
    theta_k = theta_0;
    yk = theta_0;
    losses = zeros(maxiter+1,0);
    [f,~] = feval(obj,yk,lam1,lam2);
    losses(1) = f; % the distance to the optimum
    lamk_old = 0; %stores old value to be used for momentum step
    %this loop computes the update steps with momentum
    for k = 1 : maxiter
        xk_old = theta_k;
        [~,g] = feval(obj,yk,lam1,lam2); % evals f and its grad
        theta_k = yk - s * g;
        lamk = (1+sqrt(4*lamk_old^2+1))/2;
        gammak = (1-lamk_old)/lamk;
        yk = theta_k - gammak * (theta_k - xk_old);
        lamk_old = lamk;
        [f,~] = feval(obj,theta_k,lam1,lam2);
        losses(k+1) = f; % the distance to the optimum
    end
end