function [r, w] = find_prices(param, num, grid)
    % Find r and w that clear the markets
    
    % Define the function to minimize
    funToMinimize = @(p) ex_demand(p(1),p(2), param, num, grid);
    
    % Initial guess for (r, w)
    initialGuess = [param.rho, param.rho+1];
    
    % Use fminsearch to find the minimum
    optimalParameters = fminsearch(funToMinimize, initialGuess);
    r = optimalParameters(1);
    w = optimalParameters(2);

end