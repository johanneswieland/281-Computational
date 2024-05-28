function [resid] = mkt_clearing(vec)
global sdfss
w = vec(1) ;
q = vec(2) ;
global par num grid
% capital producing firm
x_supply = kproducer(q,par) ;
% firm problem
[~,inv,profits,l,A] = firm_vfi(par,num,grid,w,q) ;
gg = kf_equation(A,grid,num) ;
g = [gg(1:num.k_n),gg(num.k_n+1:2*num.k_n)];
L_d = sum(l(:).*g(:).*grid.dk) ;
% Get consumption from the budget constraint
inv_supply = x_supply ;
inv_demand = sum(inv(:).*g(:).*grid.dk) ;
adj_costs = par.varphi/2.*sum((inv(:)./grid.k(:)).^2.*grid.k(:).*g(:).*grid.dk) ; 
Y = sum(sum(l.^par.nu.*grid.k.^par.theta.*repmat(exp(par.e),[num.k_n 1]).*g.*grid.dk)) ;
C = Y - q*inv_demand - adj_costs ;
[L_s] = hh_problem(par,w,C) ;

%C = w*L_d + sum(profits(:).*g(:).*grid.dk)/sdfss ;
sdfss = C^(-par.sigma) ;
resid = (L_d/L_s - 1)^2 + (inv_demand/inv_supply - 1)^2 ; 
plot(grid.k,g)
drawnow
end