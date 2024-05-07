function [g, c, aprime, labor_supply, capital_supply] = hh_problem(r, w, param, num, grid)
%   Solve
v_old = grid.v0 ;

dist = 1 ;
while dist > num.tol 
    [v_new,~,~] = vfi_iteration(r, w, v_old, param, num, grid) ;
    dist = max(abs((v_new(:) - v_old(:)))) ;
    v_old = v_new ;
   % disp(dist)
end
[~,c,A] = vfi_iteration(r, w, v_old, param, num, grid) ;

[gg] = kf_equation(A, grid, num) ;
g = [gg(1:num.a_n),gg(num.a_n+1:2*num.a_n)];
aprime = r * grid.a + w * repmat(param.e, num.a_n, 1) - c;

labor_supply = sum(param.e .* g * grid.da, 1);
capital_supply = sum(grid.a .* g * grid.da, 1);

end