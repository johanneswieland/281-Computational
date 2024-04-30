function [v_new,c] = vfi_iteration_implicit_q3(v0,param,num,grid,M)
%tic;

    v_old = v0;
    
    Va_Upwind = vp_upwind_q3(v0, param, num, grid,M);

    c = Va_Upwind.Va_Upwind.^(-1);
    u = utility(c);

    %CONSTRUCT MATRIX A
    X = - min(Va_Upwind.sb,0)./grid.da;
    Y = - max(Va_Upwind.sf,0)./grid.da + min(Va_Upwind.sb,0)./grid.da;
    Z = max(Va_Upwind.sf,0)./grid.da;
    
    updiag=0; %This is needed because of the peculiarity of spdiags.

    updiag=[updiag;Z(1:num.N-1);0];
   
    
    centdiag_B=reshape(Y,num.N,1);
    
    lowdiag=X(2:num.N);
    lowdiag=[lowdiag;0];
    
    B=spdiags(centdiag_B,0,num.N,num.N)+spdiags([updiag;0],1,num.N,num.N)+spdiags([lowdiag;0],-1,num.N,num.N);
    
    A = B;
 
    AA = (1/num.delta + param.rho)*speye(num.N) - A;

    %u_stacked = reshape(u,num.N,1);
    %V_stacked = reshape(V,I*J*K,1);
    
    b = u + v_old*(1/num.delta);

    v_new = AA\b; %SOLVE SYSTEM OF EQUATIONS

    %V = reshape(V_stacked,I,J,K);
    
    %Vchange = V - v;
    %v = V;

end    
    

    