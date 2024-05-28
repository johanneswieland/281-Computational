function [v_new,inv,profits,l,A] = vfi_iteration(v0,par,num,grid,w,q)
global Aswitch sdfss
% implement the algorithm in the slides.
I = num.k_n ;


[Vk_Upwind,ssb,ssf] = vp_upwind(v0,par,num,grid,w,q) ;

inv = (Vk_Upwind/sdfss - q*(1-par.tau)).*grid.k/par.varphi ;
l = (par.nu.*exp(par.e).*grid.k.^par.theta/w).^(1/(1-par.nu)) ;
profits = sdfss*(exp(par.e).*grid.k.^par.theta.*l.^par.nu - w*l - q*(1-par.tau)*inv - par.varphi/2*(inv./grid.k).^2.*grid.k) ;

X = - min(ssb,0)/grid.dk;
Y = - max(ssf,0)/grid.dk + min(ssb,0)/grid.dk ;
Z = max(ssf,0)/grid.dk ;
    
A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);

A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;


B = (1/num.Delta + par.rho)*speye(2*I) - A;

u_stacked = [profits(:,1);profits(:,2)];
V_stacked = [v0(:,1);v0(:,2)];

b = u_stacked + V_stacked/num.Delta;
V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

v_new = [V_stacked(1:I),V_stacked(I+1:2*I)];


end