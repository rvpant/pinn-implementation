% Implements the ADAM optimizer with a constant step size.
% INPUT: function handler to eval, starting param vector Theta_0, 
% constant stepsize s, maximum iterations maxiter.
% OUTPUT: vector of losses over time, theta_k approx trained params

function [losses, theta_k] = ADAM(obj, theta_0, lam1, lam2, s, maxiter)
    theta_k = theta_0;
    mk = 0;
    vk = 0;
    beta_1 = 0.9;
    beta_2 = 0.999;
    eps_adam = 1e-8;
    
    losses = zeros(maxiter+1, 0);
    for i = 1:maxiter
       [f, g] = feval(obj, theta_k, lam1, lam2); %evals func and its grad
       losses(i) = f;
       
       mk = beta_1*mk + (1-beta_1)*g;
       vk = beta_2*vk + (1-beta_2)*(g.^2);
       theta_k = theta_k - s * (mk/(1-beta_1^i))./(sqrt(vk/(1-beta_2^i)) + eps_adam);
    end
    [f, ~] = feval(obj, theta_k, lam1, lam2);
    losses(maxiter+1) = f; %stores final distance to optimum
end