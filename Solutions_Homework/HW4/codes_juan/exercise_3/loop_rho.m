clear all ; close all ; clc ;

rho_vec = linspace(0.02,0.1,10) ;
r_vec = [] ;
k_vec = [] ;
for counter_loop = 1:numel(rho_vec)
    main ;
end

save('rho_loop','k_vec','r_vec')