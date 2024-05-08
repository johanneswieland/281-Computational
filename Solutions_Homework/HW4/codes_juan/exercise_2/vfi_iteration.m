function [v_new,c,A] = vfi_iteration(v0,param,num,grid,w,r)
% implement the algorithm in the slides.
I = num.a_n ;

Aswitch = [-speye(I)*param.lambda(1),speye(I)*param.lambda(1);speye(I)*param.lambda(2),-speye(I)*param.lambda(2)];

[Va_Upwind,ssb,ssf] = vp_upwind(v0,param,num,grid,w,r) ;

c = Va_Upwind.^(-1);
u = utility(c);
X = - min(ssb,0)/grid.da;
Y = - max(ssf,0)/grid.da + min(ssb,0)/grid.da;
Z = max(ssf,0)/grid.da;
    
A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);

A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;


B = (1/num.Delta + param.rho)*speye(2*I) - A;

u_stacked = [u(:,1);u(:,2)];
V_stacked = [v0(:,1);v0(:,2)];

b = u_stacked + V_stacked/num.Delta;
V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

v_new = [V_stacked(1:I),V_stacked(I+1:2*I)];


end