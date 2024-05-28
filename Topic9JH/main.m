clear all ; close all ; clc ;


addpath('/Users/juanherreno/Documents/MATLABAutoDiff');
addpath('/Users/juanherreno/Documents/phact');


tic ;
global par grid num Aswitch sdfss
% create structure with structural parameters
call_parameters;
% create structure with numerical parameters
numerical_parameters ;
% create structure with grids and initial guesses for v
grid = create_grid(par,num) ;
I = num.k_n ;
Aswitch = [-speye(I)*par.lambda(1),speye(I)*par.lambda(1);speye(I)*par.lambda(2),-speye(I)*par.lambda(2)];

sdfss = 0.56 ;
w0 = 1.1 ;
q0 = 0.55 ;
X = [w0,q0] ;   
f = @(x) mkt_clearing(x);
options = optimset('Display','iter','TolFun',1e-08);
[X,err,exitflag] = fminsearch(f, X ,options) ;
toc;
w = X(1) ;
q = X(2) ;
% capital producer firm
x_supply = kproducer(q,par) ;
% firm problem
[v,inv,profits,l,A] = firm_vfi(par,num,grid,w,q) ;

gg = kf_equation(A,grid,num) ;
g = [gg(1:num.k_n),gg(num.k_n+1:2*num.k_n)];
L_d = sum(l(:).*g(:).*grid.dk) ;
% Get consumption from the budget constraint
inv_supply = x_supply ;
inv_demand = sum(inv(:).*g(:).*grid.dk) ;
adj_costs = par.varphi/2.*sum((inv(:)./grid.k(:)).^2.*grid.k(:).*g(:).*grid.dk) ; 
Y = sum(sum(l.^par.nu.*grid.k.^par.theta.*repmat(exp(par.e),[num.k_n 1]).*g.*grid.dk)) ;
C = Y - q*inv_demand - adj_costs ;
C1 = w*L_d + sum(profits(:).*g(:).*grid.dk)/sdfss ;
[L_s] = hh_problem(par,w,C) ;
resid = (L_d/L_s - 1)^2 + (inv_demand/inv_supply - 1)^2 ; 


global sbSS sfSS IfSS IbSS I0SS
[~,sbSS,sfSS,IfSS,IbSS,I0SS] = vp_upwind(v,par,num,grid,w,q) ;

%% Step 2: Linearize Model Equations



% Number of variables in the system 
global nVars nEErrors n_v n_g n_p varsSS
n_v = 2 * I ; 
n_g = 2 * I-1 + 1; % 2*I - 1 for g, 1 for tau
n_p = 5; % one for w, one for q, one for sdf, one for I, one for Y
nVars = n_v + n_g + n_p;
nEErrors = 2 * I; % same as n_v.

varsSS = zeros(nVars,1);
varsSS(1:2*I,1) = reshape(v,2*I,1);
varsSS(2*I+1:4*I-1,1) = gg(1:2*I-1);
varsSS(4*I,1) = 0; %tau
varsSS(4*I+1,1) = w; % w
varsSS(4*I+2,1) = q; % q
varsSS(4*I+3,1) = sdfss; % sdf
varsSS(4*I+4,1) = inv_demand; % I
varsSS(4*I+5,1) = Y; % Y



fprintf('Taking derivatives of equilibrium conditions...\n')
t0 = tic;

% Prepare automatic differentiation
vars = zeros(2*nVars+nEErrors+1,1);
vars = myAD(vars);

% Evaluate derivatives
derivativesIntermediate = equilibrium_conditions(vars);

% Extract out derivative values
derivs = getderivs(derivativesIntermediate);

% Unpackage derivatives
mVarsDerivs = derivs(:,1:nVars);
mVarsDotDerivs = derivs(:,nVars+1:2*nVars);
mEErrorsDerivs = derivs(:,2*nVars+1:2*nVars+nEErrors);
mShocksDerivs = derivs(:,2*nVars+nEErrors+1);

% rename derivatives to match notation in paper
g0 = mVarsDotDerivs;
g1 = -mVarsDerivs;
c = sparse(nVars,1);
psi = -mShocksDerivs;
pi = -mEErrorsDerivs;


[state_red,inv_state_red,g0,g1,c,pi,psi] = clean_G0_sparse(g0,g1,c,pi,psi);
n_g_red = n_g;


%% Step 4: Solve Linear System
t0 = tic;
fprintf('Solving reduced linear system...\n')

[G1,~,impact,eu,F] = schur_solver(g0,g1,c,psi,pi,1,1,1);

fprintf('...Done!\n')
fprintf('Existence and uniqueness? %2.0f and %2.0f\n',eu);
fprintf('Time to solve linear system: %2.4f seconds\n\n\n',toc(t0))

%% Step 5: Simulate Impulse Response Functions
fprintf('Simulating Model...\n')
t0 = tic;

trans_mat = inv_state_red;
[simulated,vTime] = simulate(G1,impact,num.T,num.N,num.vAggregateShock,'implicit',trans_mat);

fprintf('...Done!\n')
fprintf('Time to simulate model: %2.4f seconds\n\n\n',toc(t0))

% Add state-states back in to get values in levels
%varsSS_small = varsSS(4*I:4*I+6,1);
vAggregatetau = simulated(400,:) + varsSS(400);
vAggregatew = simulated(401,:) + varsSS(401);
vAggregateq = simulated(402,:) + varsSS(402);
vAggregatesdf = simulated(403,:) + varsSS(403);
vAggregateI = simulated(404,:) + varsSS(404);
vAggregateC = vAggregatesdf.^(-1/par.sigma) ;
vAggregateY = simulated(405,:) + varsSS(405);

% Compute Adjustment Costs

vAggregateAdj = vAggregateY - vAggregateC - vAggregateq.*vAggregateI ; 
vAggregateI_mis = vAggregateI + vAggregateAdj./vAggregateq ;

% Compute log differences for plotting
vAggregatetau_reduced = vAggregatetau;
vAggregatew_reduced = log(vAggregatew) - log(varsSS(401));
vAggregateq_reduced = log(vAggregateq) - log(varsSS(402));
vAggregatesdf_reduced = log(vAggregatesdf) - log(varsSS(403));
vAggregateI_reduced = log(vAggregateI) - log(varsSS(404));
vAggregateC_reduced = log(vAggregateC) - log(varsSS(403)^(-1/par.sigma));
vAggregateY_reduced = log(vAggregateY) - log(varsSS(405));
vAggregateAdj_reduced = log(vAggregateAdj) - log(vAggregateAdj(1));
vAggregateI_reduced_mis = log(vAggregateI_mis) - log(vAggregateI_mis(1));

time = linspace(0,num.T,num.N) ;
subplot(2,3,1)
plot(time,vAggregatetau_reduced)
title('investment subsidy')
subplot(2,3,2)
plot(time,vAggregatew_reduced)
title('wage rate')
subplot(2,3,3)
plot(time,vAggregateq_reduced)
title('price of capital')
subplot(2,3,4)
plot(time,vAggregateC_reduced)
title('Agg Consumption')
subplot(2,3,5)
plot(time,vAggregateI_reduced)
title('Agg Investment')
subplot(2,3,6)
plot(time,vAggregateY_reduced)
title('Agg Output')

disp('Sum of Investment Responses / Sum of Tax Changes')
disp(sum(vAggregateI_reduced)/sum(vAggregatetau_reduced))
disp('Equivalent of an IV (I/tau)/(q/tau)')
disp((sum(vAggregateI_reduced)/sum(vAggregatetau_reduced))/(sum(vAggregateq_reduced)/sum(vAggregatetau_reduced)))
iv = ((sum(vAggregateI_reduced)/sum(vAggregatetau_reduced))/(sum(vAggregateq_reduced)/sum(vAggregatetau_reduced))) ;

disp('Compare to a supply elasticity of')
disp(par.xi)


disp('Measurement Error Case')


disp('Sum of Investment Responses / Sum of Tax Changes')
disp(sum(vAggregateI_reduced_mis)/sum(vAggregatetau_reduced))
disp('Equivalent of an IV (I/tau)/(q/tau)')
disp((sum(vAggregateI_reduced_mis)/sum(vAggregatetau_reduced))/(sum(vAggregateq_reduced)/sum(vAggregatetau_reduced)))
iv_mis = ((sum(vAggregateI_reduced_mis)/sum(vAggregatetau_reduced))/(sum(vAggregateq_reduced)/sum(vAggregatetau_reduced))) ;

disp('Compare to a supply elasticity of')
disp(par.xi)

disp('Bias on Supply Elasticity (%)')
disp(100*(iv_mis/iv - 1))

disp('Effect of \tau on q. Is it close to 1?')

disp(sum(vAggregateq_reduced)/sum(vAggregatetau_reduced))

