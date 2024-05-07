function [sum_squared,ex_lab_demand, ex_cap_demand] = ex_demand(r, w, param, num, grid)
% Given prices, obtain excess_labor_demand and excess_capital_demand
    
    % solve households problem to obtain labor supply and capital supply
    [~, ~, ~, labor_supply, capital_supply] = hh_problem(r, w, param, num, grid);
    
    % solve firms problem to obtain labor demand and capital demand
    [capital_demand, labor_demand] = firm_problem(labor_supply, r, w, param, num, grid);

    ex_lab_demand = labor_demand - sum(labor_supply);
    ex_cap_demand = capital_demand - sum(capital_supply);

    sum_squared = ex_lab_demand^2 + ex_cap_demand^2;

end