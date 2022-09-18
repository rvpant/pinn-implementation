% Implements GD with a constant step size.
%
% INTPUT: handler, return L(Theta;lam1,lam2) and its derivatives
% theta0: the starting point
% stepSize: the stepsize
% niter: the maximum number of iterations
% RETURN: Loss_traj: the loss trajectory w.r.t. iterations
% Thetak : the approximation obtained after niter-th iterations
function [losses, theta_k] = GD(obj,theta_0,lam1,lam2,s,maxiter)
    theta_k = theta_0;
    losses = zeros(maxiter+1,0);
    for k = 1 : maxiter
        [f,g] = feval(obj,theta_k,lam1,lam2); % evals f and its grad
        losses(k) = f; % the distance to the optimum
        theta_k = theta_k - s * g; %update step
    end
    [f,~] = feval(obj,theta_k,lam1,lam2);
    losses(maxiter+1) = f; % final distance to the optimum
end