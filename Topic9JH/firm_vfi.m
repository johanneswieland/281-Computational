function [v,inv,profits,l,A] = firm_vfi(par,num,grid,w,q)
v_old = grid.v0 ;
dist = 1 ;
while dist > num.tol 
    [v_new,~,~] = vfi_iteration(v_old,par,num,grid,w,q) ;
    dist = max(abs((v_new(:) - v_old(:)))) ;
    v_old = v_new ;
   % disp(dist)
end
[v,inv,profits,l,A] = vfi_iteration(v_old,par,num,grid,w,q) ;

end