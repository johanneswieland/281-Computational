function [Va_Upwind,sb,sf] = vp_upwind(r, w, v0, param, num, grid)

V = v0 ;

%initialize forward and backwards differences with zeros: Vaf Vab

Vaf = zeros(num.a_n,2) ;
Vab = zeros(num.a_n,2) ;


% use v0 to compute backward and forward differences
Vaf(1:end-1,:) = (V(2:end,:) - V(1:end-1,:))/grid.da ;
Vaf(end,:) = 0; 

Vab(2:end,:) = (V(2:end,:)-V(1:end-1,:))./grid.da;
Vab(1,:) = (r * param.r_factors .* grid.a(1,:) + w * param.e).^(-1); %state constraint boundary condition    


% consumption and savings with forward difference
cf = max(Vaf,1e-08).^(-1);
sf = r * param.r_factors .* grid.a + w * param.e - cf ;
% consumption and savings with backward difference
cb = max(Vab,1e-08).^(-1);
sb = r * param.r_factors .* grid.a + w * param.e - cb ;
% consumption and derivative of value function at steady state
c0 = r * param.r_factors .* grid.a + w * param.e;
Va0 = c0.^(-1);

If = sf > 0; % positive drift --> forward difference
Ib = sb < 0; % negative drift --> backward difference
I0 = (1-If-Ib); % at steady state
Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term


end