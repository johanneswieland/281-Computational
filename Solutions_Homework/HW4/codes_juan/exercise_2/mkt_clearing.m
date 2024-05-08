function [resid] = mkt_clearing(r)
global par num grid
if par.pe == 0
    w = (1-par.alpha)*par.Z*(par.alpha*par.Z/(r + par.delta))^(par.alpha/(1-par.alpha)) ;
else
    w = 1.4617 ;
end
[~,~,A] = hh_vfi(par,num,grid,w,r) ;
gg = kf_equation(A,grid,num) ;
g = [gg(1:num.a_n),gg(num.a_n+1:2*num.a_n)];

k_supply = sum(grid.a(:).*g(:).*grid.da) ;
l_supply =  sum(sum(par.e.*g.*grid.da)) ;

[k_demand] = firm_problem(par,num,grid,r,l_supply) ;

resid = (k_demand/k_supply - 1)^2 ;
plot(grid.a,g)
drawnow;
end