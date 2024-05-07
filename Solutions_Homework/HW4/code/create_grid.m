function [grid] = create_grid(param,num)
% Uses structures of structural parameters and numerical parameters to
% create a grid and initial guesses for v.
grid.a = repmat(linspace(num.a_min,num.a_max,num.a_n)',[1 2]) ;
grid.v0 = utility(0.01*grid.a + param.e*1.5) ;
grid.da = grid.a(2) - grid.a(1) ;

end