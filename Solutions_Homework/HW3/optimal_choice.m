function [profit] = optimal_choice(param,num,grid)

    kuTrad = (((param.A)*param.alpha)/param.r).^(1/param.alpha);

    PiTrad = (param.A.*kuTrad).^param.alpha - param.r.*kuTrad;

    %kuProd = (((param.A_p)*param.alpha)/param.r).*(1/param.alpha) + kappa;
    %kProd = max(min(la*grid.a, kuProd),0);
    
    %PiProd = (param.A_p.*max(kProd-kappa,0)).^param.alpha - param.r.*kuTrad;
    
    profit = PiTrad;
    %kModern = max(min(la*aaa,kuModern),0);

end