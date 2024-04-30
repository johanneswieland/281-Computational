function [profit] = optimal_choice_q4(param,num,grid)

    kuTrad = (((param.A)*param.alpha)/param.r).^(1/param.alpha);
    kTrad = max(min(param.la*grid.a, kuTrad),0);
    PiTrad = (param.A.*kTrad).^param.alpha - param.r.*kTrad;
    PiTrad = PiTrad.*ones(num.N,1);
    kuProd = (((param.A_p)*param.alpha)/param.r).^(1/param.alpha) + param.kappa;
    kProd = max(min(param.la*grid.a, kuProd),0);
    
    PiProd = (param.A_p.*max(kProd-param.kappa,0)).^param.alpha - param.r.*kProd;
    
    profit = max(PiTrad,PiProd);
    %kModern = max(min(la*aaa,kuModern),0);

end