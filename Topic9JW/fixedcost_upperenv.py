#%%
import copy
import numpy as np
# from numba import njit
import scipy.sparse as sp
import scipy.optimize as opt
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import time

from sequence_jacobian.grids import agrid, markov_rouwenhorst
import sequence_jacobian.utilities as utils
from sequence_jacobian import simple, solved, het, create_model, estimation, interpolate, misc
from sequence_jacobian.utilities.interpolate import interpolate_y
from sequence_jacobian.utilities.misc import logit_choice

from func_upper_envelope import upperenv, upperenv_durable
from func_utility import util, margutil_c, margutil_d, margutilinv_c


from scipy.interpolate import LinearNDInterpolator, interpn


# TODO


def hh_init(coh, a_grid, d1_grid, eis, psi, xi, collateral_fc):
    V = util(0.2 * (coh[1,...] + collateral_fc * d1_grid[np.newaxis,:,np.newaxis]), d1_grid[np.newaxis,:,np.newaxis], psi=psi, eis=eis, xi=xi) / 0.01    
    Va = np.empty_like(V)
    
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    Vd = np.zeros_like(V)

    assert Va.min() > 0
    assert np.isnan(V).sum()==0
    assert np.isnan(Va).sum()==0
    
    return V, Va, Vd

def make_grids(rho_z, sd_z, n_z, min_a, max_a, n_a, min_d, max_d, n_d, min_m, max_m, n_m, n_x):
    e_grid, pi_e, Pi = markov_rouwenhorst(rho_z, sd_z, n_z)
    a_grid = agrid(max_a, n_a, min_a)
    m_grid = agrid(max_m, n_m, min_m)
    d1_grid = agrid(max_d, n_d, min_d)
    x_grid = agrid(max_d + max_a, n_x, min_a + min_d)

    return e_grid, pi_e, Pi, a_grid, m_grid, d1_grid, x_grid

def labor_income(e_grid, Y, Tr, tau):
    y_grid = e_grid * (1 - tau) *  Y + Tr

    return y_grid


def cash_on_hand(a_grid, y_grid, d1_grid, r, p, deltad, pbar, fc, maint_cost, op_cost, collateral, spread):

    collateral_fc = collateral * (1 - deltad) * (1 - fc) * pbar

    # effective interest rate based on whether household was borrowing
    atilde_on_grid = a_grid[np.newaxis, np.newaxis, :] - collateral_fc * d1_grid[np.newaxis, :, np.newaxis]
    borrow = (atilde_on_grid < 0).astype(int)
    if  spread>0:
        interestrate = r  + spread * (atilde_on_grid < 0)
    else:
        interestrate = r
    
    # durable values
    deltad_maint = deltad * (1 - maint_cost)
    maint_cost_ratio = p * maint_cost * deltad / (1 - deltad_maint)

    dur_maint = pbar * op_cost + maint_cost_ratio
    effective_dur_price = p + pbar * op_cost  - collateral_fc

    # cash on hand after durable upkeep and durable sales
    adj_dur_value = p * (1 - deltad) * (1 - fc)  - (1 + interestrate) * collateral_fc
    noadj_dur_value = - dur_maint * (1 - deltad_maint) - collateral_fc * (interestrate + deltad_maint)

    # cash on hand after durable upkeep and durable sales
    dur_value = np.zeros([2,1,len(d1_grid),len(a_grid)])
    dur_value[1, ...] = adj_dur_value
    dur_value[0, ...] = noadj_dur_value

    coh = ( ((1 + interestrate) * a_grid[np.newaxis, np.newaxis, :])[np.newaxis, :, :, :]
             + dur_value * d1_grid[np.newaxis, np.newaxis, :, np.newaxis] 
             + y_grid[np.newaxis, :, np.newaxis, np.newaxis] )

    return coh, dur_value, deltad_maint, effective_dur_price, interestrate, borrow, collateral_fc


@het(exogenous='Pi', policy=['d1','a'], backward=['V','Vd','Va'], backward_init=hh_init)
def household_v(Va_p, Vd_p, V_p, coh, a_grid, d1_grid, x_grid, m_grid, y_grid, interestrate, dur_value, effective_dur_price, beta, eis, psi, xi, deltad_maint, taste_shock, min_c):

    util_params = {'psi': psi, 'eis': eis, 'xi': xi}
    
    # ===================
    # === EGM PROBLEM ===
    # ===================
    # === STEP 1: Find a for given d' using EGM ===
    W = beta * V_p                    # end-of-stage vfun
    Wd = beta * Vd_p                  # end-of-stage vdfun
    uc_endo = beta * Va_p             # envelope condition

    d1egm = d1_grid[np.newaxis,:,np.newaxis] + np.zeros_like(W)

    c_egm_agrid = margutilinv_c(uc_endo, d1egm, **util_params)          # Euler equation
    m_egm_agrid = (   c_egm_agrid                                               # budget constraint
                    + a_grid[np.newaxis, np.newaxis, :]) 

    max_cegm = (   m_grid[np.newaxis, np.newaxis, :]
                -  0 * d1_grid[np.newaxis, :, np.newaxis] 
                +  0 * y_grid[:, np.newaxis, np.newaxis] ) 

    max_cegm = np.maximum(max_cegm, min_c)           # impose very low non-negative consumption when upkeep too expensive         
    
    # interpolate with upper envelope, enforce borrowing limit
    Vegm, Vmegm, Vdegm, cegm, aegm = upperenv(W, Wd, m_egm_agrid, max_cegm, m_grid, a_grid, d1egm, psi, eis, xi)

    # ==================================
    # === DURABLE ADJUSTMENT PROBLEM ===
    # ==================================
    # === STEP 2: Find adjustment solution conditional on cash on hand ===
    # no adjustment sets d'=(1-delta)d and has cash on hand x=a+y
    eval_FOC_swap = (-Vdegm + effective_dur_price * Vmegm).swapaxes(-2,-1)
    Vegm_swap = Vegm.swapaxes(-2,-1)
    Vdegm_swap = Vdegm.swapaxes(-2,-1)
    
    coh_adj_m_swap =   (  m_grid[np.newaxis, np.newaxis, :] 
                        + effective_dur_price * d1_grid[np.newaxis, :, np.newaxis] 
                        + np.zeros_like(Vegm)).swapaxes(-2,-1)

    assert np.isnan(eval_FOC_swap).sum() == 0
    assert np.isnan(coh_adj_m_swap).sum() == 0
    assert np.isnan(Vegm_swap).sum() == 0
    assert np.isnan(Vdegm_swap).sum() == 0
    
    dadjx, Vadjx, Vdadjx = upperenv_durable(Vegm_swap, Vdegm_swap, eval_FOC_swap, coh_adj_m_swap, d1_grid, x_grid)
    
    # interpolate onto COH grid
    dadj = interpolate_y(x_grid[np.newaxis, np.newaxis, :], coh[1, ...], dadjx[:, np.newaxis, :])
    dadj = np.maximum(dadj, d1_grid[0])
    
    # ========================
    # === COMBINED PROBLEM ===
    # ========================
    # === STEP 3: put no adjustment in row = 0, adjustment in row = 1 and solve 
    # ===         for endogenous outcomes conditional on either choice
    d1 = np.zeros_like(coh)
    d1[1, ...] = dadj
    d1[0, ...] = (1 - deltad_maint) * d1_grid[np.newaxis, :, np.newaxis] + np.zeros_like(dadj)

    d_purchase = np.zeros_like(coh)
    d_purchase[1,...] = dadj

    m_post_adj = coh - effective_dur_price * d_purchase

    # find optimal a, c from EGM solutions given coh and d1
    n_all = np.prod(coh.shape)
    income = y_grid[np.newaxis, :, np.newaxis, np.newaxis] + np.zeros_like(coh)
    
    adjust_points_m = np.concatenate((income.reshape([n_all,1]), d1.reshape([n_all,1]), m_post_adj.reshape([n_all,1])), axis=1)
    
    a = interpn((y_grid, d1_grid, m_grid), aegm, adjust_points_m, method='linear', bounds_error=False, fill_value=None).reshape(coh.shape)
    a = np.maximum(a, a_grid[0])

    c = m_post_adj - a
    c = np.maximum(c, min_c)

    # check budget constraint
    assert np.max(np.abs(a + effective_dur_price * d_purchase + c - coh) * (c > 10 ** - 6).astype(int))<10**-8


    # =========================
    # === UPDATE VALUE FUNC ===
    # =========================
    # === STEP 4: Update value functions ===
    # interpolation points for value functions
    all_points_a = np.concatenate((income.reshape([n_all,1]), d1.reshape([n_all,1]), a.reshape([n_all,1])), axis=1)

    # marginal value of liquid assets
    muc = margutil_c(c, d1, **util_params)
    Va = (1 + interestrate) * muc 

    Wdinterp = interpn((y_grid, d1_grid, a_grid), Wd, all_points_a, method='linear', bounds_error=False, fill_value=None).reshape(coh.shape)

    # marginal value of durables goods
    Vd = np.zeros_like(Va)
    Vd[1, ...] = dur_value[1, ...] * muc[1, ...]

    # Vd[1, ...] = (1 - deltad) * (1 - fc) * ( margutil_d(c[1, ...], d1[1, ...], **util_para).squeeze() 
                                # + Wdinterp_a[1, ...] ).squeeze()

    

    # Wdinterp_noadj_a = interpn((y_grid, d1_grid, a_grid), Wd, noadjust_points_a, method='linear', bounds_error=False, fill_value=None).reshape(coh.shape[1:])
    mud = margutil_d(c, d1, **util_params)
    Vd[0, ...] = ((1 - deltad_maint) * (   mud[0, ...] + Wdinterp[0, ...] )
                                        + dur_value[0, ...] * muc[0, ...] )
                                         
    # value function
    Winterp = interpn((y_grid, d1_grid, a_grid), W, all_points_a, method='linear', bounds_error=False, fill_value=None).reshape(coh.shape)
    V = util(c, d1, **util_params) + Winterp

    assert c.min()>0, 'negative c'
    assert d1.min()>0, 'negative d'
    assert np.isnan(V).sum()==0, 'undefined V'

    # ============================
    # ===   LOGIT ADJUSTMENT   ===
    # ============================  
    
    # === STEP 9: Logit adjustment function  ===
    P, EV = logit_choice(V, taste_shock)
    adjust = P[1, ...]


    # === STEP 10: Update solutions and value function ===
    a = (P * a).sum(axis=0)
    c = (P * c).sum(axis=0)
    d1 = (P * d1).sum(axis=0)
    Va = (P * Va).sum(axis=0)
    Vd = (P * Vd).sum(axis=0)
    V = EV

    # print(c_adj.min(), c_noadj.min())
    # print(d1_adj.min(), d1_noadj.min())    
    assert np.isnan(V).sum()==0


    return Va, Vd, V, a, c, d1, adjust
    


def net_assets(a, d1, c, adjust, collateral_fc, borrow, deltad, d1_grid, y_grid, r, p, pbar, fc, op_cost, spread):
    # net asset position
    anet = a - collateral_fc * d1 
    aborrow = - anet * borrow

    # durable expenditure
    x = d1 - (1 - deltad) * (1 - fc * adjust) * d1_grid[np.newaxis, :, np.newaxis]

    # budget constraint
    income = y_grid[:, np.newaxis, np.newaxis] + np.zeros_like(c)
    durexp = p * x
    borrowcost = spread * aborrow 
    operatingcost = pbar * op_cost * d1 
    nondurexp = c + operatingcost + borrowcost
    totexp = durexp + nondurexp
    bdgt_constr = income + r * anet - totexp

    return anet, aborrow, x, bdgt_constr, income, nondurexp, durexp, totexp, operatingcost, borrowcost


if __name__=='__main__':
    hh = household_v
    hh = hh.add_hetinputs([make_grids, labor_income, cash_on_hand])
    hh = hh.add_hetoutputs([net_assets])


    cali = dict()

    rhoYq = 0.966
    rhoYm = 0.966**(1/3)
    sigmaYq = 0.5
    sigmaYm = sigmaYq * 3 / np.sqrt(1 + (1 + rhoYq) ** 2 + (1 + rhoYq + rhoYq**2) ** 2 + rhoYq ** 2 * (1 + rhoYq) ** 2  + rhoYq ** 2)

    cali = {'taste_shock': 1E-3 * 30000 , 'r': 0.02/12,
            'eis': 0.25, 'xi': 1, 
            # 'psi': 0.11, 'deltad': 0.015, 'fc': 0.2,
            'psi': 150, 'deltad': 0.015, 'fc': 0.05,
            'collateral': 0.7, 'spread': 0.04/12,
            # 'maint_cost': 0.466, 'op_cost': 0.018, 
            'maint_cost': 0.466, 'op_cost': 0.018, 
            'Tr': 0, 'tau': 0,
            # 'beta': 0.9948110014496739,
            'beta': 0.992,
            # 'B_ss': 0.1,
            'rho_z': rhoYm, 'sd_z': sigmaYm, 'n_z': 5, 
            'min_a': 0.0, 'max_a': 180/12, 'n_a': 50,
            'min_m': 0.0, 'max_m': 180/12, 'n_m': 100,
            'min_d': 0.0015, 'max_d': 24/12, 'n_d': 40,
            'n_x': 100,
            'min_c': 10 ** - 6,
            'Y': 1/12, 'p': 1, 'pbar': 1}  

    ss = hh.steady_state(cali)
    print(f"Aggregate durables: {ss['D1']/ss['NONDUREXP']/12:0.3f}")
    print(f"Aggregate durable exp share: {ss['DUREXP']/ss['TOTEXP']:0.3f}")
    print(f"Aggregate assets: {ss['ANET']:0.3f}")
    print(f"Aggregate adjustment: {ss['ADJUST']:0.3f}")
    print(f"Aggregate BC: {ss['BDGT_CONSTR']:0.4f}")

    jac = hh.jacobian(ss, inputs=['Tr','p','r'], outputs=['TOTEXP','NONDUREXP','DUREXP'], T=30,  h=1E-6)

    print(jac['TOTEXP']['Tr'][0:3,0].sum(), jac['NONDUREXP']['Tr'][0:3,0].sum(), jac['DUREXP']['Tr'][0:3,0].sum())
    print(jac['DUREXP']['p'][0:6,6:30].sum() / (ss['DUREXP'] * 6) )
    print(jac['NONDUREXP']['Tr'][:12,0].sum())


    distr = ss.internals['household_v']['D']
    adjust = ss.internals['household_v']['adjust']
    d1_grid = ss.internals['household_v']['d1_grid']
    a_grid = ss.internals['household_v']['a_grid']

    figsize=0.6
    fig, axes = plt.subplots(1, 3, figsize=(24*figsize, 8*figsize))
    ax = axes.flatten()

    ax[0].plot(a_grid, distr.sum(axis=0).sum(axis=0), linewidth=2)
    ax[0].set_title('Assets')
    ax[1].plot(d1_grid, distr.sum(axis=0).sum(axis=1), linewidth=2)
    ax[1].set_title('Durables')
    ax[2].plot(d1_grid, (distr * adjust).sum(axis=0).sum(axis=1) / distr.sum(axis=0).sum(axis=1), linewidth=2)
    ax[2].set_title('Durables')


# %%
