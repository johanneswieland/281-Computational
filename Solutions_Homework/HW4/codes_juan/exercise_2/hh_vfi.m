function [v,c,A] = hh_vfi(par,num,grid,w,r)

v_old = grid.v0 ;

dist = 1 ;
while dist > num.tol 
    [v_new,~,~] = vfi_iteration(v_old,par,num,grid,w,r) ;
    dist = max(abs((v_new(:) - v_old(:)))) ;
    v_old = v_new ;
   % disp(dist)
end
[v,c,A] = vfi_iteration(v_old,par,num,grid,w,r) ;

end