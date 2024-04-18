clear all ; close all ; clc ;
tic ;
% create structure with structural parameters
call_parameters;
% create structure with numerical parameters
numerical_parameters ;
% create structure with grids and initial guesses for v
grid = create_grid(param,num) ;

% create a function vfi_iteration that takes as input the structures
% previously created and produces a guess for v and c.


v_old = grid.v0 ;

tic ;
vfi_iteration(v_old,param,num,grid) ;
toc ;


dist = 1 ;
while dist > num.tol 
    [v_new,~] = vfi_iteration(v_old,param,num,grid) ;
    dist = max(abs((v_new - v_old))) ;
    v_old = v_new ;
end
    [v_new,c] = vfi_iteration(v_old,param,num,grid) ;
toc ;


[sim] = simulate(c,param,num,grid) ;

figure()
plot(sim.a)
figure()
plot(sim.beta_1)

