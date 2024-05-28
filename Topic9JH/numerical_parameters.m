num.Delta = 100 ;
num.k_n = 100 ;
num.k_min = 0.01 ;
num.k_max = 30 ;
num.tol = 1e-7 ;


% initialize shocks for simulation
num.T = 20;
num.N = 2000;
num.vAggregateShock = zeros(1,num.N);
num.vAggregateShock(1,50) = 1;