function [Vk_Upwind,sb,sf,If,Ib,I0] = vp_upwind(v0,par,num,grid,w,q)

global sdfss
V = v0 ;

r = par.rho ;

%initialize forward and backwards differences with zeros: Vaf Vab

Vkf = zeros(num.k_n,2) ;
Vkb = zeros(num.k_n,2) ;



% use v0 to compute backward and forward differences
Vkf(1:end-1,:) = (V(2:end,:) - V(1:end-1,:))/grid.dk ;
Vkf(end,:) = 0; 

Vkb(2:end,:) = (V(2:end,:)-V(1:end-1,:))./grid.dk ;
Vkb(1,:) = sdfss*(par.varphi*(par.delta) + q*(1-par.tau)) ; %state constraint boundary condition    


%consumption and savings with forward difference
invf = (Vkf/sdfss - q*(1-par.tau)).*grid.k/par.varphi;
sf = -par.delta*grid.k + invf ;
%consumption and savings with backward difference
invb = (Vkb/sdfss - q*(1-par.tau)).*grid.k/par.varphi;
sb = -par.delta*grid.k + invb ;
%consumption and derivative of value function at steady state
inv0= par.delta*grid.k ;
Vk0 = sdfss*(par.varphi*(inv0./grid.k) + q*(1-par.tau)) ;

If = sf > 0; %positive drift --> forward difference
Ib = sb < 0; %negative drift --> backward difference
I0 = (1-If-Ib); %at steady state
Vk_Upwind = Vkf.*If + Vkb.*Ib + Vk0.*I0; %important to include third term


end