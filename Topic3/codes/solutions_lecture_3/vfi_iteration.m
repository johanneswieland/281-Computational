function [v_new,c] = vfi_iteration(v0,param,num,grid)
% implement the algorithm in the slides.


[Va_Upwind] = vp_upwind(v0,param,num,grid) ;

c = Va_Upwind.^(-1);
u = utility(c);
Vchange = u + Va_Upwind.*(param.y + param.r*grid.a - c) - param.rho*v0 ;

v_new = v0 + num.Delta*Vchange;


end