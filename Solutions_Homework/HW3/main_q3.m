clear all ; close all ; clc ;
tic ;
%% Setup 

% create structure with structural parameters
call_parameters;
% create structure with numerical parameters
numerical_parameters ;
% create structure with grids and initial guesses for v
grid = create_grid(param,num) ;

param.r = (1/param.beta) - 1 ; % This is now the rental cost of capital
param.A = 0.5;
param.alpha= 0.5;

%%  Run Value Function Iteration with Implicit Method
v_old = grid.v0 ;
tic ;

M = optimal_choice(param, num, grid);

v_old = grid.v0 ; % + param.y 
tic ;
vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
dist = 1 ;
counter = 0;
while dist > num.tol 
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
    dist = max(abs((v_new - v_old)));
    % disp("Current error: " + dist) 
    v_old = v_new ;
    counter = counter + 1;
end
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
v_implicit = v_new;
c_implicit = c;

%% Compare results 
figure()
hold on 
plot(grid.a, v_implicit, Color = 'blue', LineStyle='--', LineWidth=0.5)
hold off

figure()
hold on 
plot(grid.a, c_implicit, Color = 'blue', LineStyle='--', LineWidth=0.5)
hold off


% 
% [sim] = simulate(c,param,num,grid) ;
% 
% figure()
% plot(sim.a)
% figure()
% plot(sim.beta_1)

