function [Va_Upwind] = vp_upwind(v0,param,num,grid)


% unpack the initial guess for v0 and call it V



%initialize forward and backwards differences with zeros: Vaf Vab

Vaf = zeros(num.a_n,1) ;
Vab = zeros(num.a_n,1) ;


% use V to compute backward and forward differences

%Vaf = 
%Vab = 

% impose the following boundary conditions. We will talk about them.
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
%If = 
%Ib = 
%I0 =


% Compute the upwind scheme
%Va_Upwind = ;





end