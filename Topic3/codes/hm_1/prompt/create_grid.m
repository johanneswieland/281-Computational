function [grid] = create_grid(param,num)
% Uses structures of structural parameters and numerical parameters to
% create a grid and initial guesses for v.
grid.a = linspace(num.a_min,num.a_max,num.a_n)' ;
grid.v0 = utility(0.1*grid.a) ;
grid.da = grid.a(2) - grid.a(1) ;

end