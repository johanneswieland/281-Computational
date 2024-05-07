function checkMarketClearance(capital_demand, capital_supply, labor_demand, labor_supply,num)
    if abs(capital_demand - sum(capital_supply)) < num.mc_tol
        disp('Capital market clears. The difference is: ');
        disp(abs(capital_demand - sum(capital_supply)));
    else
        disp('Capital market does not clear. The difference is: ');
        disp(abs(capital_demand - sum(capital_supply)));
    end

    if abs(labor_demand - sum(labor_supply)) < num.mc_tol
        disp('Labor market clears. The difference is: ');
        disp(abs(labor_demand - sum(labor_supply)));
    else
        disp('Labor market does not clear. The difference is: ');
        disp(abs(labor_demand - sum(labor_supply)));
    end
end