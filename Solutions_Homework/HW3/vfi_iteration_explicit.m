function [v_new,c] = vfi_iteration_explicit(v0,param,num,grid)
% implement the algorithm in the slides.

vp = vp_upwind(v0, param, num, grid);

% Infer consumption from Va_Upwind
c = vp.Va_Upwind.^(-1); 
u = utility(c) ;

% Compute the change between rho*V and the new iteration u + v'*s
Vchange = u + vp.Va_Upwind.*(param.y + param.r.*grid.a - c) - param.rho.*v0 ;


% update the value function with a step parameter Delta
v_new = v0 + num.Delta*Vchange;


end