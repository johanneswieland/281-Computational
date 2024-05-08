%clear all ; close all ; clc ;
tic ;
global par grid num
% create structure with structural parameters
call_parameters;
% create structure with numerical parameters
numerical_parameters ;
% create structure with grids and initial guesses for v
grid = create_grid(par,num) ;

if counter_loop ==1
    r0 = par.rho - 0.001 ;
else
    r0 = r_vec(end) ;
end
X = [r0] ;   
f = @(x) mkt_clearing(x);
options = optimset('Display','iter','TolFun',1e-08);
[X,err,exitflag] = fminsearch(f, X ,options) ;

r = X ;
w = (1-par.alpha)*(par.alpha/(r + par.delta))^(par.alpha/(1-par.alpha)) ;
[~,~,A] = hh_vfi(par,num,grid,w,r) ;
gg = kf_equation(A,grid,num) ;
g = [gg(1:num.a_n),gg(num.a_n+1:2*num.a_n)];

k_supply = sum(grid.a(:).*g(:).*grid.da) ;
l_supply =  sum(sum(par.e.*g.*grid.da)) ;

[k_demand] = firm_problem(par,num,grid,r,l_supply) ;

resid = (k_demand/k_supply - 1)^2 ;

r_vec = [r_vec;r] ;
k_vec = [k_vec;k_supply] ;
