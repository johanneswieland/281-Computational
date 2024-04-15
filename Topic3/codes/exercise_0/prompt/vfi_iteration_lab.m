function [v_new,c] = vfi_iteration_lab(v0,param,num,grid)
% implement the algorithm in the slides.


% unpack the initial guess for v0 and call it V




%initialize forward and backwards differences with zeros Vaf Vab



% use V to compute backward and forward differences



% impose the following boundary conditions
Vaf(end) = 0; 
Vab(1) = (param.r*grid.a(1) + param.y).^(-1); %state constraint boundary condition    


%consumption and savings with forward difference
cf = max(Vaf,1e-08).^(-1);
%sf =  ;


%consumption and savings with backward difference
cb = max(Vab,1e-08).^(-1);
%sb =  ;
%consumption and derivative of value function at steady state
%c0 =  ;
Va0 = c0.^(-1);


% compute indicator functions that capture the upwind scheme.



% Compute the upwind scheme
Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term


% Infer consumption from Va_Upwind
%c = 
u = utility(c) ;

% Compute the change between rho*V and the new iteration u + v'*s
%Vchange =  ;


% update the value function with a step parameter Delta
v_new = v_old + num.Delta*Vchange;


end