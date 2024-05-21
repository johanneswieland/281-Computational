import numpy as np
import scipy as sp

def firm(calibration, ss, T):

    alpha = calibration['alpha']

    I = sp.sparse.eye(T)
    Ip1 = sp.sparse.diags([np.ones(T-1)], [1], (T, T))
    Im1 = sp.sparse.diags([np.ones(T-1)], [-1], (T, T))
    Z = sp.sparse.csr_matrix((T, T))

    J = {}

    Y = ss['Z'] * ss['K'] ** alpha * ss['L'] ** (1 - alpha)
    r = alpha * Y / ss['K']
    w = (1 - alpha) * Y / ss['L']

    # firm block matrices: output
    J['Y'] = {'Z': Y / ss['Z'] * I, 
              'K': alpha * Y / ss['K'] * Im1}
    
    # firm block matrices: real rate
    J['r'] = {'Z': r / ss['Z'] * I, 
              'K': (alpha - 1) * r / ss['K'] * Im1}
    
    # firm block matrices: real rate
    J['w'] = {'Z': w / ss['Z'] * I, 
              'K': alpha * w / ss['K'] * Im1}
    
    return J