function [L_s] = hh_problem(par,w,C)
% Assume mkt clearing and get C from budget constraint.
% Use C and labor supply to get L_s
L_s = (w*C^(-par.sigma)).^(1/par.psi) ;
end