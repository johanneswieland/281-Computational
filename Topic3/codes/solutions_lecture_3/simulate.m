function [sim] = simulate(c,param,num,grid)



sim.a_fine = linspace(num.a_min,num.a_max,num.a_n_fine) ; %define a finer grid.
sim.c_interp = interp1(grid.a,c,sim.a_fine) ; %interpolation

sim.a = nan(num.T,num.N) ; 
sim.a_index = nan(num.T,num.N) ; % T should be 5*100, N = 100 
sim.a_index(1,:) = randi(num.a_n_fine,[1 num.N]) ; % initial position (index)
sim.a(1,:) = sim.a_fine(sim.a_index(1,:)) ; % initial asset holdings.


for tt = 2:num.T
    sim.a(tt,:) = sim.a_fine(sim.a_index(tt-1,:)) + num.delta*(param.r*sim.a_fine(sim.a_index(tt-1,:)) + param.y - sim.c_interp(sim.a_index(tt-1,:))) ;
    sim.a_index(tt,:) = knnsearch(sim.a_fine',sim.a(tt,:)') ;
    sim.c(tt-1,:) = sim.c_interp(sim.a_index(tt-1,:)) ;
    temp = regress(sim.c_interp(sim.a_index(tt-1,:))',[ones(num.N,1) sim.a_fine(sim.a_index(tt-1,:))']) ;
    sim.beta_0(tt) = temp(1) ;
    sim.beta_1(tt) = temp(2) ;
    
end

sim.beta_0(1) = nan ;
sim.beta_1(1) = nan ;

   

end