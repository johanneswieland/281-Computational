function [grid] = create_grid(param,num)
% Uses structures of structural parameters and numerical parameters to
% create a grid and initial guesses for v.
grid.k = repmat(linspace(num.k_min,num.k_max,num.k_n)',[1 2]) ;
grid.v0 = (grid.k.*exp(param.e)) ;
grid.dk = grid.k(2) - grid.k(1) ;

end