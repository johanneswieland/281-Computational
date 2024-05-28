% Inputs:  (1) vars: vector which contains the variables in the system, the time derivatives of 
%			         those variables, the expectational errors, and the shocks
%
% Outputs: (1) vResduals: residuals of equilibrium conditions, evaluated at vars

function vResidual = equilibrium_conditions(vars)

%----------------------------------------------------------------
% Housekeeping
%----------------------------------------------------------------

% Declare global variables
global par num grid Aswitch IfSS IbSS I0SS varsSS nVars nEErrors

I = num.k_n ;


% Unpack vars
V=vars(1:2*I) + varsSS(1:2*I);
g=vars(2*I+1:4*I-1) + varsSS(2*I+1:4*I-1);	% vector of distribution, removing last point		
g_end=1/grid.dk-sum(g);		% ensures that distribution integrates to 1
tau = vars(4*I) ;
w = vars(4*I+1) + varsSS(4*I+1);
q = vars(4*I+2) + varsSS(4*I+2);
sdf = vars(4*I+3) + varsSS(4*I+3);
aggI = vars(4*I+4) + varsSS(4*I+4);
aggY = vars(4*I+5) + varsSS(4*I+5);

V = reshape(V,I,2);

VDot = vars(nVars+1:nVars+2*I);
gDot = vars(nVars+2*I+1:nVars+4*I-1);
tauDot = vars(nVars+4*I);

VEErrors = vars(2*nVars+1:2*nVars+2*I);

tauShock = vars(2*nVars+nEErrors+1);

% Initialize other variables, using vars to ensure everything is a dual number
Vkf = V;
Vkb = V;

%----------------------------------------------------------------
% Compute one iteration of HJB Equation
%----------------------------------------------------------------

Vkf(1:end-1,:) = (V(2:end,:) - V(1:end-1,:))/grid.dk ;
Vkf(end,:) = 0; 

Vkb(2:end,:) = (V(2:end,:)-V(1:end-1,:))./grid.dk ;
Vkb(1,:) = repmat(sdf*(par.varphi*(par.delta) + q*(1-tau)),[1 2]) ; %state constraint boundary condition    


%consumption and savings with forward difference
invf = (Vkf/sdf - q*(1-tau)).*grid.k/par.varphi;
sf = -par.delta*grid.k + invf ;
%consumption and savings with backward difference
invb = (Vkb/sdf - q*(1-tau)).*grid.k/par.varphi;
sb = -par.delta*grid.k + invb ;
%consumption and derivative of value function at steady state
inv0= par.delta*grid.k ;
Vk0 = sdf*(par.varphi*(inv0./grid.k) + q*(1-tau)) ;

If = sf > 0; %positive drift --> forward difference
Ib = sb < 0; %negative drift --> backward difference
I0 = (1-If-Ib); %at steady state
Vk_Upwind = Vkf.*If + Vkb.*Ib + Vk0.*I0; %important to include third term

inv = (Vk_Upwind/sdf - q*(1-tau)).*grid.k/par.varphi ;
l = (par.nu.*exp(par.e).*grid.k.^par.theta/w).^(1/(1-par.nu)) ;
profits = sdf.*(exp(par.e).*grid.k.^par.theta.*l.^par.nu - w*l - q*(1-tau)*inv - par.varphi/2*(inv./grid.k).^2.*grid.k) ;
X = -sb.*IbSS/grid.dk;
Y = -sf.*IfSS/grid.dk + sb.*IbSS/grid.dk;
Z = sf.*IfSS/grid.dk;

X(1,:)=0;
lowdiag=reshape(X,2*I,1);
Z(I,:)=0;

A = spdiags(reshape(Y,2*I,1),0,2*I,2*I)...
    +spdiags(lowdiag(2:2*I),-1,2*I,2*I)...
    +spdiags([0,reshape(Z,1,2*I)]',1,2*I,2*I)...
    +Aswitch;


%----------------------------------------------------------------
% Compute equilibrium conditions
%----------------------------------------------------------------

% HJB Equation
hjbResidual = reshape(profits,2*I,1) + A * reshape(V,2*I,1) + VDot + VEErrors - par.rho * reshape(V,2*I,1);

% KFE 
gIntermediate = A' * [g;g_end];
gResidual = gDot - gIntermediate(1:2*I-1,1);


% temporary variables

temp_inv = sum(inv(:).* [g;g_end] * grid.dk ) ;
firm_output = (exp(par.e).*grid.k.^par.theta.*l.^par.nu) ;
temp_output = sum(firm_output(:).* [g;g_end] * grid.dk ) ;
q_hat = temp_inv.^(1/par.xi) ;
Taxes = q*tau*temp_inv ;
Profits = sum(profits(:).* [g;g_end] * grid.dk )/sdf ;

C = w*sum(l(:).* [g;g_end] * grid.dk ) + Profits - Taxes;
% Use C and labor supply to get L_s
sdf_hat = C^(-par.sigma) ;
w_hat = sum(l(:).* [g;g_end] * grid.dk )^(par.psi)/sdf ;


% Aggregates

qResidual = q_hat - q;
wResidual = w_hat - w;
sdfResidual = sdf_hat - sdf ;
aggIResidual = temp_inv - aggI ;
aggYResidual = temp_output - aggY ;
% Law of motion for aggregate shocks
tauResidual = tauDot + (1 - par.rhotau) * tau - par.sigmatau * tauShock;

vResidual = [hjbResidual;gResidual;tauResidual;wResidual;qResidual;sdfResidual;aggIResidual;aggYResidual];












