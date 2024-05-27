#%%
# import time
import numpy as np
from sequence_jacobian import simple, solved, create_model
from basicfixedcost_upperenv import household_v, make_grids, labor_income, cash_on_hand, net_assets

household = household_v
household = household.add_hetinputs([make_grids, labor_income, cash_on_hand])
household = household.add_hetoutputs([net_assets])

@simple
def firm(Y, W_ss):
    W = W_ss 
    L = Y
    return W, L

@simple
def monetary_policy(r_ss, rshock):
    r = r_ss + rshock
    return r  

@simple
def durableprice(p, pbar, pshock, X, X_ss, psid):
    dur_supply = p - pbar * (X / X_ss) ** psid + pshock
    return dur_supply   

@simple
def firm_ss(Y, pbar):
    L = 1
    W = Y
    p = pbar
    return W, L, p

# @simple
# def fiscal(G_ss, B_ss, r, Y, tau_ss, tshock, phib):
#     G = G_ss
#     Tr = tshock
#     B = B_ss 
#     tau =  ( G - B  + (1 + r) * B(-1) + Tr ) / Y
#     govtbudget = -G + B - (1 + r) * B(-1) - Tr + tau * Y
#     return B, G, Tr, tau, govtbudget

@solved(unknowns={'B': (0.01, 2)}, targets=["govtbudget"])
def fiscal(G_ss, B_ss, r, Y, B, Y_ss, tau_ss, tshock, phib):
    G = G_ss
    Tr = tshock
    tau = tau_ss + phib * (B(-12) - B_ss)  
    govtbudget = -G + B - (1 + r) * B(-1) - Tr + tau * Y

    return govtbudget, G, Tr, tau  

@simple
def fiscal_ss(G_ss, B_ss, r, Y):
    G = G_ss
    Tr = 0
    B = B_ss 
    tau = (G + r * B) / Y
    govtbudget = -G + B - (1 + r) * B(-1) - Tr + tau * Y

    return govtbudget, B, G, Tr, tau 

# @simple
# def usercost(p, r, deltad, op_cost, pbar, spread):
#     user_cost = p + pbar * op_cost - p(+1) * (1 - deltad) / (1 + r(+1))
#     user_cost_borrow = p + pbar * op_cost - p(+1) * (1 - deltad) / (1 + r(+1) + spread)
#     p_p = p(+1)
#     return user_cost, user_cost_borrow, p_p


@simple
def mkt_clearing(B, ANET, BORROWCOST, OPERATINGCOST, Y, C, DUREXP, G): #, D, p
    asset_mkt = ANET - B 
    goods_mkt = Y - G - C - DUREXP - OPERATINGCOST - BORROWCOST
    
    return asset_mkt, goods_mkt



# '''Part 3: DAG'''

def dag():
    # Combine blocks
    fc_model = create_model([household, firm, mkt_clearing, fiscal, durableprice, monetary_policy], name="FC")
    fc_model_ss = create_model([household, firm_ss, mkt_clearing, fiscal_ss], name="FC SS")

    rhoYq = 0.966
    rhoYm = 0.966**(1/3)
    sigmaYq = 0.5
    sigmaYm = sigmaYq * 3 / np.sqrt(1 + (1 + rhoYq) ** 2 + (1 + rhoYq + rhoYq**2) ** 2 + rhoYq ** 2 * (1 + rhoYq) ** 2  + rhoYq ** 2)

    calibration = {'taste_shock': 1E-3 * 100 , 'r': 0.02/12,
            'eis': 1, 'xi': 1, 
            # 'psi': 0.11, 'deltad': 0.015, 'fc': 0.2,
            'psi': 0.06, 'deltad': 0.015, 'fc': 0.05,
            'collateral': 0, 'spread': 0,
            # 'maint_cost': 0.466, 'op_cost': 0.018, 
            'maint_cost': 0, 'op_cost': 0, 
            'psid': 0.2, 
            'Tr': 0, 'tau': 0,
            # 'beta': 0.9948110014496739,
            'beta': 0.995,
            'G_ss': 0, # 0.2 * 1/12 'B_ss': 0.1,
            'rho_z': rhoYm, 'sd_z': sigmaYm, 'n_z': 2, 
            'min_a': 0.0, 'max_a': 80/12, 'n_a': 100,
            'min_m': 0.1/12, 'max_m': 80/12, 'n_m': 200,
            'min_d': 0.01, 'max_d': 12/12, 'n_d': 50,
            'n_x': 200,
            'min_c': 10 ** - 6,
            'Y': 1/12, 'p': 1, 'pbar': 1, 'phib': 0.1,
            'tshock': 0, 'rshock': 0, 'pshock': 0}  

    unknowns_ss = {'B_ss': (0.2, 0.6)}
    targets_ss = {'asset_mkt': 0.}
    ss = fc_model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='brentq')


    # Transitional dynamics
    exogenous = ['tshock','rshock','pshock']
    unknowns = ['Y','p']
    targets = ['asset_mkt','dur_supply']
    ss['X_ss'] = ss['X']
    ss['r_ss'] = ss['r']
    ss['W_ss'] = ss['W']
    ss['Y_ss'] = ss['Y']
    ss['tau_ss'] = ss['tau']

    return fc_model_ss, ss, fc_model, unknowns, targets, exogenous

if __name__=='__main__':
    # t0 = time.time()

    fc_model_ss, ss, fc_model, unknowns, targets, exogenous = dag()

    assert np.abs(ss['goods_mkt']) < 10 ** -8

    jac = household.jacobian(ss, inputs=['Y','Tr','p','r'], T=30)
    print(jac['TOTEXP']['Tr'][0:3,0].sum(), jac['NONDUREXP']['Tr'][0:3,0].sum(), jac['DUREXP']['Tr'][0:3,0].sum())
    print(jac['DUREXP']['p'][0:6,6:30].sum() / (ss['DUREXP'] * 6) )


    T = 300
    G = fc_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

    # t1 = time.time()
    # print(t1-t0)

#     # rhos = np.array([0.2, 0.4])
#     dZ = np.zeros([T,])
#     dZ[1] = 0.01
#     # dZ = 0.01*ss['Z']*rhos**(np.arange(T)[:, np.newaxis]) # get T*5 matrix of dZ
#     dr = G['Y']['rshock'] @ dZ

#     plt.plot(dr[:50, ])
#     plt.title(r'$Y$ response to 1% $r$ shock$')
#     plt.ylabel(r'basis points deviation from ss')
#     plt.xlabel(r'quarters')
#     plt.show()

#     dZ = np.zeros([T,])
#     dZ[0] = 0.01
#     # dZ = 0.01*ss['Z']*rhos**(np.arange(T)[:, np.newaxis]) # get T*5 matrix of dZ
#     dr = G['Y']['pshock'] @ dZ

#     plt.plot(dr[:50, ])
#     plt.title(r'$Y$ response to 1% $p$ shock$')
#     plt.ylabel(r'basis points deviation from ss')
#     plt.xlabel(r'quarters')
#     plt.show()

#     dZ = np.zeros([T,])
#     dZ[0] = 0.01
#     # dZ = 0.01*ss['Z']*rhos**(np.arange(T)[:, np.newaxis]) # get T*5 matrix of dZ
#     dr = G['Y']['tshock'] @ dZ

#     plt.plot(dr[:50, ])
#     plt.title(r'$Y$ response to 1% $T$ shock$')
#     plt.ylabel(r'basis points deviation from ss')
#     plt.xlabel(r'quarters')
#     plt.show()

#     # tests

#     for mkt in ['goods_mkt', 'asset_mkt']:
#         assert ss[mkt]<10**-6, mkt + ' does not clear in steady state'
#         assert abs(G[mkt]['rshock']).max()<10**-6, mkt + ' does not clear dynamically'


# %%
