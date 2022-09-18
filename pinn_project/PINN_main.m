clear all
close all
clc
%% DO NOT CHANGE THIS PART ------------------------------------------------
rng(1234, 'twister')
%--------------------------------------------------------------------------
%% P3 (g) Setup [CHOOSE APPROPRIATLY TO GET THE BEST RESULTS] -------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
width  = 5; % width
niter  = 1000; % the maximum number of iterations
Theta0 = rand(4*width+1, 1); % the initial guess, size of (4*width+1,1)
lam1   = 10;
lam2   = 100; 
stepSize_GD  = 0.001; % To complete ; % the stepsize for GD
stepSize_NAG = 0.001; % To complete ; % the stepsize for NAG
stepSize_ADAM= 0.001; % To complete ; % the stepsize for ADAM
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% Loading DATA -----------------------------------------------------------
global R_xt_train
global U_xt_train
global U_y_train
load('FINAL_PINN_DATA.mat')
%--------------------------------------------------------------------------
%% The four first order optimization methods-------------------------------
tic
display('Quadratic: GD with constant stepsize')
[GD_dist2opt, Theta_GD] = GD('PINN_loss',Theta0,lam1,lam2,stepSize_GD,niter);
toc
%--------------------------------------------------------------------------
% tic
% display('Quadratic: Polyak Heavy Ball with constant stepsize')
% [PHB_dist2opt, Theta_PHB] = Opt_PHB('PINN_loss',Theta0,lam1,lam2,mu,stepSize_PHB,niter);
% toc
%--------------------------------------------------------------------------
tic
display('Quadratic: Nesterov Accelerated GD with constant stepsize')
[NAG_dist2opt, Theta_NAG] = GD_momentum('PINN_loss',Theta0,lam1,lam2,stepSize_NAG,niter);
toc
%--------------------------------------------------------------------------
tic
display('Quadratic: ADAM with constant stepsize')
[ADAM_dist2opt, Theta_ADAM] = ADAM('PINN_loss',Theta0,lam1,lam2,stepSize_ADAM,niter);
toc
%--------------------------------------------------------------------------
%% P3 (g) Report the figure generated according to the below --------------
GD_str   = strcat('lr=',num2str(stepSize_GD,'%1.2e'),', GD');
%PHB_str  = strcat('lr=',num2str(stepSize_PHB,'%1.2e'),', PHB');
NAG_str  = strcat('lr=',num2str(stepSize_NAG,'%1.2e'),', NAG');
ADAM_str = strcat('lr=',num2str(stepSize_ADAM,'%1.2e'),', ADAM');
figure
semilogy(0:niter,  GD_dist2opt,'b-')
hold on
grid on
%semilogy(0:niter, PHB_dist2opt,'r-')
semilogy(0:niter, NAG_dist2opt,'g-')
semilogy(0:niter,ADAM_dist2opt,'k-')
legend(GD_str,NAG_str,ADAM_str)
xlabel('the number of iterations, k')
ylabel('$L(\theta^{(k)})$','Interpreter','latex')
title(strcat('PINNs, Width=',num2str(width)))
%--------------------------------------------------------------------------
fprintf('ThM_GD= %1.4f, ThM_NAG= %1.4f, ThM_ADAM= %1.4f\n', Theta_GD(end), Theta_NAG(end), Theta_ADAM(end))
%--------------------------------------------------------------------------
%% P3 (h) SAVE your Theta -------------------------------------------------
Theta_best = Theta_GD;   % Uncomment if you want to report Theta trained by GD
% % Theta_best = Theta_PHB;  % Uncomment if you want to report Theta trained by PHB
% % Theta_best = Theta_NAG;  % Uncomment if you want to report Theta trained by NAG
% % Theta_best = Theta_ADAM; % Uncomment if you want to report Theta trained by ADAM
save('trainedTheta.mat','Theta_best')