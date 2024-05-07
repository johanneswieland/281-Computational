% Outputs:  (1) r: steady state interest rate
%           (2) w: steady state wage
%           (3) K: capital stock
%           (4) A: matrix for computing value function
%           (5) u: utility over grid
%           (5) c: consumption over grid
%           (6) V: value function over grid
%			(7) g: distribution
%           (8) dV_Upwind: derivative of value function by upwind scheme
%           (9) dVf: derivative of value function by forward difference
%           (10) dVb: derivative of value function by backward difference
%           (11) If: indicator for forward drift in savings
%           (12) Ib: indicator for backward drift in savings
%           (13) I0: indicator for no drift in savings

function [r,w,K,A,u,c,V,g,dV_Upwind,dVf,dVb,If,Ib,I0] = compute_steady_state()

%----------------------------------------------------------------
% Housekeeping
%----------------------------------------------------------------

% Declare global variables
global ggamma rrho ddelta aalpha ssigmaTFP rrhoTFP z lla mmu ttau I amin amax a da aa ...
	zz Aswitch rmin rmax r0 maxit crit Delta Ir crit_S zAvg A
	
% Initialze variables for iteration
dVf = zeros(I,2);
dvB = zeros(I,2);
c = zeros(I,2);
KS = zeros(Ir,1);
r = r0;
KD = (((aalpha) / (r + ddelta)) ^ (1 / (1 - aalpha))) * zAvg;
w = (1 - aalpha) * (KD ^ aalpha) * ((zAvg) ^ (-aalpha));
v0(:,1) = (w*mmu*(1-z(1)) + r.*a).^(1-ggamma)/(1-ggamma)/rrho;
v0(:,2) = (w*(1-ttau)*z(2) + r.*a).^(1-ggamma)/(1-ggamma)/rrho;

%----------------------------------------------------------------
% Iterate to find steady state interest rate
%----------------------------------------------------------------

for ir=1:Ir

    r_r(ir)=r;
    rmin_r(ir)=rmin;
    rmax_r(ir)=rmax;
    
    KD(ir,1) = (((aalpha) / (r + ddelta)) ^ (1 / (1 - aalpha))) * zAvg;
	w = (1 - aalpha) * (KD(ir) ^ aalpha) * ((zAvg) ^ (-aalpha));
        
    if ir>1
    
        v0 = V_r(:,:,ir-1);
        
    end
    
    v = v0;
    
    %%%%
    % Solve for value function given r
    %%%%
    
    for n=1:maxit
    
        V = v;
        V_n(:,:,n)=V;
        
        % Compute forward difference
        dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVf(I,:) = (w*((1 - ttau) * z + mmu * (1 - z)) + r.*amax).^(-ggamma); %will never be used, but impose state constraint a<=amax just in case
        
        % Compute backward difference
        dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVb(1,:) = (w*((1 - ttau) * z + mmu * (1 - z)) + r.*amin).^(-ggamma); %state constraint boundary condition

        % Compute consumption and savings with forward difference
        cf = dVf.^(-1/ggamma);
        ssf = w*((1 - ttau) * zz + mmu * (1 - zz)) + r.*aa - cf;
        
        % Compute consumption and savings with backward difference
        cb = dVb.^(-1/ggamma);
        ssb = w*((1 - ttau) * zz + mmu * (1 - zz)) + r.*aa - cb;
        
        % Compute consumption and derivative of value function for no drift
        c0 = w*((1 - ttau) * zz + mmu * (1 - zz)) + r.*aa;
        dV0 = c0.^(-ggamma);
        
        % Compute upwind differences    
        If = ssf > 0;       %positive drift --> forward difference
        Ib = ssb < 0;       %negative drift --> backward difference
        I0 = (1-If-Ib);     %no drift
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
        c = dV_Upwind.^(-1/ggamma);
        u = c.^(1-ggamma)/(1-ggamma);
        savingsSS = w*((1 - ttau) * zz + mmu * (1 - zz)) + r.*aa - c;
        
        % Construct matrix for updating implicit scheme
        X = -min(ssb,0)/da;
        Y = -max(ssf,0)/da + min(ssb,0)/da;
        Z = max(ssf,0)/da;
 
        A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
        A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
        A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
        
        B = (1/Delta + rrho)*speye(2*I) - A;

        u_stacked = [u(:,1);u(:,2)];
        V_stacked = [V(:,1);V(:,2)];
        b = u_stacked + V_stacked/Delta;
        
        % Solve system of equations for updating implicit scheme
        V_stacked = B\b;
        
        V = [V_stacked(1:I),V_stacked(I+1:2*I)];
        
        % Update value function and check convergence
        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        if dist(n)<crit
        
            %disp('Value Function Converged, Iteration = ')
            %disp(n)
            break
            
        end
        
    end
    
    %%%%
    % Solve for stationary distribution
    %%%%
    
    % Preallocate matrices for solving linear system
    AT = A';
    b = zeros(2*I,1);

    % Normalization so pdf integrates to 1
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,2*I-i_fix)];
    AT(i_fix,:) = row;
    
    %Solve linear system for distribution
    gg = AT\b;
    g_sum = gg'*ones(2*I,1)*da;
    gg = gg./g_sum;
    g = [gg(1:I),gg(I+1:2*I)];

    % Compute objects from this iteration
    g_r(:,:,ir) = g;
    adot(:,:,ir) = w * ((1 - ttau) * zz + mmu * (1 - zz))+ r.*aa - c;
    V_r(:,:,ir) = V;
    
    KS(ir,1) = g(:,1)'*a*da + g(:,2)'*a*da;
    S(ir,1) = KS(ir,1) - KD(ir,1);
    
    % Update interest rate
    if S(ir)>crit_S
    
        %disp('Excess Supply')
        rmax = r;
        r = 0.5*(r+rmin);
        
    elseif S(ir)<-crit_S;
    
        %disp('Excess Demand')
        rmin = r;
        r = 0.5*(r+rmax);
        
    elseif abs(S(ir))<crit_S;
    
        display('Steady State Found, Interest rate =')
        disp(r)
        break
        
    end
    
end

% Save steady state aggregate capital stock
K = KS(ir,1);
% investment
savings = w*((1 - ttau) * zz + mmu * (1 - zz)) + r.*aa - c;
invest = sum((reshape(savings,I*2,1) .* gg)'*da);