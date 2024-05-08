clear all ; close all ; clc ;

Z_vec = [1:-0.1:0.01] ;
r_vec = [] ;
k_vec = [] ;
for counter_loop = 1:numel(Z_vec)
    main ;
end

save('z_loop','k_vec','r_vec')