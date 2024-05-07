%% Solves the Krusell and Smith (1998)


tstart = tic;

%% Setup the toolbox
% Just need to include folders containing the files in the path.

addpath('/Users/juanherreno/Documents/MATLABAutoDiff');
addpath('/Users/juanherreno/Documents/phact');


% initialize shocks for simulation
T = 200;
N = 2000;
vAggregateShock = zeros(1,N);
vAggregateShock(1,1) = 1;

%% Step 0: Set Parameters
% The script sets up parameters relevant for the model
set_parameters;

%% Step 1: Solve for Steady State
tStart = tic;

fprintf('Computing steady state...\n')
global IfSS IbSS I0SS varsSS A

[rSS,wSS,KSS,ASS,uSS,cSS,VSS,gSS,dVUSS,dVfSS,dVbSS,IfSS,IbSS,I0SS] = ...
    compute_steady_state();

fprintf('Time to compute steady state: %.3g seconds\n\n\n',toc(tStart));

% Store steady state values
varsSS = zeros(nVars,1);
varsSS(1:2*I,1) = reshape(VSS,2*I,1);
ggSS = reshape(gSS,2*I,1);
varsSS(2*I+1:4*I-1,1) = ggSS(1:2*I-1);
varsSS(4*I,1) = 0;
varsSS(4*I+1,1) = KSS;
varsSS(4*I+2,1) = rSS;
varsSS(4*I+3,1) = wSS;
varsSS(4*I+4,1) = (KSS ^ aalpha) * (zAvg ^ (1 - aalpha));
CSS = sum(cSS(:) .* gSS(:) * da);
varsSS(4*I+5,1) = CSS;
varsSS(4*I+6,1) = ddelta * KSS;

%% Step 2: Linearize Model Equations
% For computing derivatives, the codes written for solving for the
%    steady-state can be used almost verbatim using automatic
%    differentiation toolbox as long as only the functions supported by
%    automatic differentation are used. For list of supported functions and
%    documentation of relevant syntax check <<https://github.com/sehyoun/MATLABAutoDiff>>
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
[simulated,vTime] = simulate(G1,impact,T,N,vAggregateShock,'implicit',trans_mat);

fprintf('...Done!\n')
fprintf('Time to simulate model: %2.4f seconds\n\n\n',toc(t0))

% Add state-states back in to get values in levels
%varsSS_small = varsSS(4*I:4*I+6,1);
vAggregateTFP = simulated(400,:) + varsSS(400);
vAggregateOutput = simulated(404,:) + varsSS(404);
vAggregateConsumption = simulated(405,:) + varsSS(405);
vAggregateInvestment = simulated(406,:) + varsSS(406);

% Compute log differences for plotting
vAggregateTFP_reduced = vAggregateTFP;
vAggregateOutput_reduced = log(vAggregateOutput) - log(varsSS(404));
vAggregateConsumption_reduced = log(vAggregateConsumption) - log(varsSS(405));
vAggregateInvestment_reduced = log(vAggregateInvestment) - log(varsSS(406));


%% (optional) Step 7: Plot relevant values
% Plot impulse response functions
figure
subplot(2,2,1)
hold on
plot(vTime,100 * vAggregateTFP_reduced,'linewidth',1.5)
set(gcf,'color','w')
xlim([vTime(1) vTime(end)])
title('TFP','interpreter','latex','fontsize',14)
ylabel('$\%$ deviation from s.s.','interpreter','latex')
hold off

subplot(2,2,2)
hold on
plot(vTime,100 * vAggregateOutput_reduced,'linewidth',1.5)
set(gcf,'color','w')
xlim([vTime(1) vTime(end)])
title('Output','interpreter','latex','fontsize',14)
hold off

subplot(2,2,3)
hold on
plot(vTime,100 * vAggregateConsumption_reduced,'linewidth',1.5)
set(gcf,'color','w')
xlim([vTime(1) vTime(end)])
title('Consumption','interpreter','latex','fontsize',14)
ylabel('$\%$ deviation from s.s.','interpreter','latex')
xlabel('Quarters','interpreter','latex')
hold off

subplot(2,2,4)
hold on
plot(vTime,100 * vAggregateInvestment_reduced,'linewidth',1.5)
set(gcf,'color','w')
xlim([vTime(1) vTime(end)])
title('Investment','interpreter','latex','fontsize',14)
xlabel('Quarters','interpreter','latex')
hold off

toc(tstart)


