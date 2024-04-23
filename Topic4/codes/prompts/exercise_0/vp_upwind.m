function [Va_Upwind] = vp_upwind(v0,param,num,grid)


V = v0 ;



%initialize forward and backwards differences with zeros: Vaf Vab

Vaf = zeros(num.a_n,2) ;
Vab = zeros(num.a_n,2) ;



% use v0 to compute backward and forward differences
Vaf(1:end-1,:) = (V(2:end,:) - V(1:end-1,:))/grid.da ;
Vaf(end,:) = 0; 

Vab(2:end,:) = (V(2:end,:)-V(1:end-1,:))./grid.da;
Vab(1,:) = (param.r*grid.a(1,:) + param.y).^(-1); %state constraint boundary condition    


%consumption and savings with forward difference
cf = max(Vaf,1e-08).^(-1);
sf = param.y + param.r.*grid.a - cf ;
%consumption and savings with backward difference
cb = max(Vab,1e-08).^(-1);
sb = param.y + param.r.*grid.a - cb ;
%consumption and derivative of value function at steady state
c0 = param.y + param.r.*grid.a ;
Va0 = c0.^(-1);

If = sf > 0; %positive drift --> forward difference
Ib = sb < 0; %negative drift --> backward difference
I0 = (1-If-Ib); %at steady state
Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term


end