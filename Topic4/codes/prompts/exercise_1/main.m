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

dist = 1 ;
while dist > num.tol 
    [v_new,~,~] = vfi_iteration(v_old,param,num,grid) ;
    dist = max(abs((v_new(:) - v_old(:)))) ;
    v_old = v_new ;
   % disp(dist)
end
[v_new,c,A] = vfi_iteration(v_new,param,num,grid) ;


[gg] = kf_equation(A,grid,num) ;
g = [gg(1:num.a_n),gg(num.a_n+1:2*num.a_n)];


toc ;

figure()
plot(grid.a,c,'o')


figure()
plot(grid.a,g)
xlim([num.a_min 1])
