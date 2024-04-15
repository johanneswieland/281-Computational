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
[vp_upwind] = vp_upwind(v_old,param,num,grid) ;

toc ;
