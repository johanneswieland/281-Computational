clear all ; close all ; clc ;
tic ;
%% Setup 

% create structure with structural parameters
call_parameters;
% create structure with numerical parameters
numerical_parameters ;
% create structure with grids and initial guesses for v
param.r = (1/param.beta) - 1 ; % This is now the rental cost of capital
param.A = 0.5;
param.A_p = 0.8;    
param.alpha= 0.5;
param.kappa = 0.0;
param.la = 1;

grid = create_grid(param,num) ;

%%  Run Value Function Iteration with Implicit Method
v_old = grid.v0 ;
tic ;

M = optimal_choice_q4(param, num, grid);

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
v_implicit_k0 = v_new;
c_implicit_k0 = c;

%%

param.kappa = 0.1;

v_old = grid.v0 ;
tic ;

M = optimal_choice_q4(param, num, grid);

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
v_implicit_k1 = v_new;
c_implicit_k1 = c;

%%

param.kappa = 0.4;

v_old = grid.v0 ;
tic ;

M = optimal_choice_q4(param, num, grid);

v_old = grid.v0 ; 
tic ;
vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
dist = 1 ;
counter = 0;
while dist > num.tol 
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
    dist = max(abs((v_new - v_old)));
    v_old = v_new ;
    counter = counter + 1;
end
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
v_implicit_k2 = v_new;
c_implicit_k2= c;

%%

param.kappa = 4;

v_old = grid.v0 ;
tic ;

M = optimal_choice_q4(param, num, grid);

v_old = grid.v0 ; 
tic ;
vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
dist = 1 ;
counter = 0;
while dist > num.tol 
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
    dist = max(abs((v_new - v_old)));

    v_old = v_new ;
    counter = counter + 1;
end
    [v_new,c] = vfi_iteration_implicit_q3(v_old,param,num,grid,M) ;
toc ;
v_implicit_k3 = v_new;
c_implicit_k3= c;

%% Compare results 
figure()
hold on 
plot(grid.a, v_implicit_k0, Color = 'blue', LineStyle='--', LineWidth=0.5)
plot(grid.a, v_implicit_k1, Color = 'red', LineStyle='--', LineWidth=0.5)
plot(grid.a, v_implicit_k2, Color = 'green', LineStyle='--', LineWidth=0.5)
plot(grid.a, v_implicit_k3, Color = 'yellow', LineStyle='--', LineWidth=0.5)
hold off

figure()
hold on 
plot(grid.a, c_implicit_k0, Color = 'blue', LineStyle='--', LineWidth=0.5)
plot(grid.a, c_implicit_k1, Color = 'red', LineStyle='--', LineWidth=0.5)
plot(grid.a, c_implicit_k1, Color = 'green', LineStyle='--', LineWidth=0.5)
plot(grid.a, c_implicit_k3, Color = 'yellow', LineStyle='--', LineWidth=0.5)
hold off


