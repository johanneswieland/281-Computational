function [v_new,c] = vfi_iteration_lab(v0,param,num,grid)
% implement the algorithm in the slides.


% Perform the upwind scheme.

[vp_upwind] = vp_upwind(v_old,param,num,grid) ;


% Infer consumption from Va_Upwind
%c = 
u = utility(c) ;

% Compute the change between rho*V and the new iteration u + v'*s
%Vchange =  ;


% update the value function with a step parameter Delta
v_new = v_old + num.Delta*Vchange;


end