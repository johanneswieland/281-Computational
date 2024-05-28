function [gg] = kf_equation(A,grid,num)

AT = A';
b = zeros(2*num.k_n,1);

%need to fix one value, otherwise matrix is singular
i_fix = 1;
b(i_fix)=.1;
row = [zeros(1,i_fix-1),1,zeros(1,2*num.k_n-i_fix)];
AT(i_fix,:) = row;

%Solve linear system
gg = AT\b;
g_sum = gg'*ones(2*num.k_n,1)*grid.dk;
gg = gg./g_sum;

end