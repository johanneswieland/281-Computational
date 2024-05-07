function [capital_demand,labor_demand] = firm_problem(labor_supply, r, w, param, num, grid)
% obtain capital demand and labor demand
    capital_demand = (param.alpha/(r + param.delta))^(1/(1 - param.alpha)) * sum(labor_supply);
    labor_demand = ((1-param.alpha)/w)^(1/param.alpha) * capital_demand;
end