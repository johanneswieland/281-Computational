import numpy as np
from numba import njit

from sequence_jacobian.interpolate import interpolate_point
from func_utility import util, margutil_c, margutil_d

@njit
def upperenv_vec(W, Wd, m_endo, coh, m_grid, a_grid, d, *args):


    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_m = coh.shape
    n_b2, n_a = W.shape

    assert coh.min()>0

    assert np.abs(n_b-n_b2)<10**-8

    a = np.zeros_like(coh)
    c = np.zeros_like(coh)
    V = -np.inf * np.ones_like(coh)
    Vd = -np.inf * np.ones_like(coh)
    Vm = -np.inf * np.ones_like(coh)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        d0 = d[ib, 0]

        # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_a - 1):
            m_low, m_high = m_endo[ib, ja], m_endo[ib, ja + 1]
            W_low, W_high = W[ib, ja], W[ib, ja + 1]
            Wd_low, Wd_high = Wd[ib, ja], Wd[ib, ja + 1]
            ap_low, ap_high = a_grid[ja], a_grid[ja + 1]
            
           # loop over exogenous asset grid (increasing) 
            for im in range(n_m):  
                mcur = m_grid[im]
                coh_cur = coh[ib, im]
                
                interp = (m_low <= mcur <= m_high) 
                extrap = (ja == n_a - 2) and (mcur > m_endo[ib, n_a - 1])

                # exploit that a_grid is increasing
                if (m_high < mcur < m_endo[ib, n_m - 1]):
                    break

                if interp or extrap:
                    W0 = interpolate_point(mcur, m_low, m_high, W_low, W_high)
                    Wd0 = interpolate_point(mcur, m_low, m_high, Wd_low, Wd_high)
                    a0 = interpolate_point(mcur, m_low, m_high, ap_low, ap_high)
                    c0 = coh_cur - a0
                    V0 = util(c0, d0, *args) + W0
                    Vm0 = margutil_c(c0, d0, *args)
                    Vd0 = margutil_d(c0, d0, *args) + Wd0

                    # upper envelope, update if new is better
                    if V0 > V[ib, im]:
                        a[ib, im] = a0 
                        c[ib, im] = c0
                        V[ib, im] = V0
                        Vm[ib, im] = Vm0
                        Vd[ib, im] = Vd0

        # Enforce borrowing constraint
        im = 0
        while im < n_m and m_grid[im] <= m_endo[ib, 0]:
            a[ib, im] = a_grid[0]
            c[ib, im] = coh[ib, im]
            V[ib, im] = util(c[ib, im], d0, *args) + W[ib, 0]
            Vm[ib, im] = margutil_c(c[ib, im], d0, *args)
            Vd[ib, im] = margutil_d(c[ib, im], d0, *args) + Wd[ib, 0]
            im += 1
    
    assert c.min()>0
    assert a.min()>a_grid[0] - 10 ** - 10

    return V, Vm, Vd, c, a                                               


def upperenv(W, Wd, m_endo, coh, m_grid, a_grid, d, *args):
    # collapse (adj, z, d, a) into (b, a)
    shapea = W.shape
    W = W.reshape((-1, shapea[-1]))
    Wd = Wd.reshape((-1, shapea[-1]))
    m_endo = m_endo.reshape((-1, shapea[-1]))
    d = d.reshape((-1, shapea[-1]))

    shapem = coh.shape
    coh = coh.reshape((-1, shapem[-1]))
    
    
    V, Vm, Vd, c, a = upperenv_vec(W, Wd, m_endo, coh, m_grid, a_grid, d, *args)
    
    # report on (adj, z, d, a)
    return V.reshape(shapem), Vm.reshape(shapem), Vd.reshape(shapem), c.reshape(shapem), a.reshape(shapem)      

@njit
def upperenv_durable_vec(V_FOC, Vd_FOC, eval_FOC, coh, d1_grid, x_grid, d):

    """Interpolate durable expenditure to exogenous grid."""
    n_b, n_m, n_d = eval_FOC.shape
    n_x = x_grid.shape[0]
    n_d = d1_grid.shape[0]

    
    # assert V_FOC.shape == eval_FOC.shape
    # assert coh.shape == eval_FOC.shape
    assert coh.min()>0

    
    # d = np.zeros([n_b,n_x])
    # V = -np.inf * np.ones([n_b,n_x])
    V = -np.inf * np.ones_like(d)
    Vd = np.zeros_like(d)
    
    # loop over other states, collapsed into single axis
    for ib in range(n_b):

        xlist = []
        dlist = []
        Vlist = []
        Vdlist = []

        for id in range(n_m):
            # m0 = m_grid[id]

            # loop over segments of FOC (not necessarily increasing)
            for jd in range(n_d - 1):
                foc_low, foc_high = eval_FOC[ib, id, jd], eval_FOC[ib, id, jd + 1]
                
                if np.sign(foc_low) != np.sign(foc_high):
                    x_low, x_high = coh[ib, id, jd], coh[ib, id, jd + 1]
                    d_low, d_high = d1_grid[jd], d1_grid[jd + 1]
                    V_FOC_low, V_FOC_high = V_FOC[ib, id, jd], V_FOC[ib, id, jd + 1]
                    Vd_FOC_low, Vd_FOC_high = Vd_FOC[ib, id, jd], Vd_FOC[ib, id, jd + 1]

                    x0 = interpolate_point(0, foc_low, foc_high, x_low, x_high)
                    d0 = interpolate_point(0, foc_low, foc_high, d_low, d_high)
                    V0 = interpolate_point(0, foc_low, foc_high, V_FOC_low, V_FOC_high)
                    Vd0 = interpolate_point(0, foc_low, foc_high, Vd_FOC_low, Vd_FOC_high)

                    dlist.append(d0)
                    xlist.append(x0)
                    Vlist.append(V0)
                    Vdlist.append(Vd0)
        
        # now have combos given coh0, choice m0, d0 satisfies FOC
        # with unique solution, will just have one d0 for each x0 and increasing in x0
        n_jm = len(xlist)

        # if dlist == []:
        #     xlist.append(coh[ib, :, :].min())
        #     xlist.append(coh[ib, :, :].max())
        #     dlist.append(d[ib, 0])
        #     dlist.append(d[ib, 0])
        #     Vlist.append(V_FOC[ib, 0, 0])
        #     Vlist.append(V_FOC[ib, 0, 0])
        #     Vdlist.append(Vd_FOC[ib, 0, 0])
        #     Vdlist.append(Vd_FOC[ib, 0, 0])

            # print(eval_FOC[ib, :])
            # print(coh[ib, :])

        # assert np.asarray(dlist).min()>0, 'negative d in envelope'
        # assert np.asarray(xlist).min()>0, 'negative x in envelope'
        
        # loop over adjecent (x,d) solutions (not necessarily increasing)
        for jm in range(n_jm - 1):
            x_low, x_high = xlist[jm], xlist[jm + 1]
            d_low, d_high = dlist[jm], dlist[jm + 1]
            V_low, V_high = Vlist[jm], Vlist[jm + 1]
            Vd_low, Vd_high = Vdlist[jm], Vdlist[jm + 1]

            # loop over exogenous cash on hand grid (increasing) 
            for ix in range(n_x):  
                xcur = x_grid[ix]
                
                interp = (x_low <= xcur <= x_high)
                extrap = (jm == n_jm - 2) and (xcur > xlist[n_jm - 1])

                # exploit that x_grid is increasing
                if (x_high < xcur < xlist[n_jm - 1]):
                    break

                if interp or extrap:
                    if abs(x_low-x_high)<10 ** -8:
                        d1 = 0.5 * (d_low + d_high)
                        V1 = 0.5 * (V_low + V_high)
                        Vd1 = 0.5 * (Vd_low + Vd_high)
                    else:
                        d1 = interpolate_point(xcur, x_low, x_high, d_low, d_high)
                        V1 = interpolate_point(xcur, x_low, x_high, V_low, V_high)
                        Vd1 = interpolate_point(xcur, x_low, x_high, Vd_low, Vd_high)

                    # upper envelope, update if new is better
                    if V1 > V[ib, ix]:
                        d[ib, ix] = d1
                        V[ib, ix] = V1
                        Vd[ib, ix] = Vd1
        
            # When nodes of the coh grid are not covered then that means
            # d* = dmin
            ix = 0
            xmin = xlist[0]
            while ix < n_x and x_grid[ix] <= xmin:
                d[ib, ix] = d1_grid[0]
                ix += 1 

            # Extrapolate at boundary
            if ix >= 1:
                d[ib, ix-1] = interpolate_point(x_grid[ix-1], xlist[0], xlist[1], dlist[0], dlist[1])      
    
    return d, V, Vd

def upperenv_durable(V_FOC, Vd_FOC, eval_FOC, coh, d1_grid, x_grid):
    # collapse (adj, z, m, d) into (b, m, d)
    shapem = V_FOC.shape
    V_FOC = V_FOC.reshape((-1, shapem[-2], shapem[-1]))
    Vd_FOC = Vd_FOC.reshape((-1, shapem[-2], shapem[-1]))
    eval_FOC = eval_FOC.reshape((-1, shapem[-2], shapem[-1]))
    coh = coh.reshape((-1, shapem[-2], shapem[-1]))

    shapex = (np.prod(shapem[:-2]), x_grid.shape[0])
    shapeout = shapem[:-2] + (x_grid.shape[0],)

    assert shapem[-1] == d1_grid.shape[0]
    assert V_FOC.shape[0] == np.prod(shapem[:-2])

    d = np.zeros(shapex)
    
    d, V, Vd = upperenv_durable_vec(V_FOC, Vd_FOC, eval_FOC, coh, d1_grid, x_grid, d)
    
    # report on (adj, z, x) 
    return d.reshape(shapeout), V.reshape(shapeout), Vd.reshape(shapeout)   