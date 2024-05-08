function [k_demand] = firm_problem(par,num,grid,r,l_supply)

k_demand = (par.alpha*par.Z/(r + par.delta))^(1/(1-par.alpha))*l_supply ;

end