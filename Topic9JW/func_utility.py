import numpy as np
from numba import njit

@njit
def util(c, d, psi, eis, xi):
    if eis==1:
        uc = np.log(c)
    else:
        uc = (c ** (1 - 1/eis) - 1) / (1 - 1/eis)

    if xi==1:
        ud = psi * np.log(d)
    else:
        ud = psi * (d ** (1 - 1/xi) - 1) / (1 - 1/xi)

    u = uc + ud

    return u

@njit
def margutil_c(c, d, psi, eis, xi):
    muc = c ** (-1/eis)
    return muc  

@njit
def margutilinv_c(muc, d, psi, eis, xi):
    c = muc ** (-eis)
    return c       

@njit
def margutil_d(c, d, psi, eis, xi):
    mud = psi * d ** (-1/xi)
    return mud   

# @njit
# def util(c, d, psi=0, eis=1, xi=1):
#     if eis==1:
#         uc = psi * np.log(c)
#     else:
#         uc = psi * (c ** (1 - 1/eis) - 1) / (1 - 1/eis)

#     if xi==1:
#         ud = (1 - psi) * np.log(d)
#     else:
#         ud = (1 - psi) * (d ** (1 - 1/xi) - 1) / (1 - 1/xi)

#     u = uc + ud

#     return u

# @njit
# def margutil_c(c, d, psi=0, eis=1, xi=1):
#     muc = psi * c ** (-1/eis)
#     return muc  

# @njit
# def margutilinv_c(muc, d, psi=0, eis=1, xi=1):
#     c = psi * muc ** (-eis)
#     return c       

# @njit
# def margutil_d(c, d, psi=0, eis=1, xi=1):
#     mud = (1 - psi) * d ** (-1/xi)
#     return mud        

# @njit
# def util(c, d, psi=0.5, eis=1, xi=1):
    
#     if eis==1 and xi==1:
#         u =  psi * np.log(c) + (1 - psi) * np.log(d)
#     elif eis!=1:
#         u =  (c ** (psi * (1 - 1/eis)) * d ** ((1 - psi) * (1 - 1/eis)) - 1) / (1 - 1/eis)
#     else:
#         cbase =     psi   ** (1/xi) * c ** ((xi -1) / xi)
#         dbase = (1 - psi) ** (1/xi) * d ** ((xi -1) / xi)
#         complement_base = cbase + dbase
#         complement =  complement_base ** (xi / (xi -1))

#         u = (complement ** (1 - 1 / eis) - 1) / (1 - 1 / eis)

#     return u   

# @njit
# def margutil_c(c, d, psi=0.5, eis=1, xi=1):
    
#     if xi==1:
#         marguc =  psi * c ** (psi * (1 - 1/eis) - 1) * d ** ((1 - psi) * (1 - 1/eis))

#     else:
        
#         if eis==xi:
#             marguc = (psi / c) ** (1 / xi)
#         else:
#             cbase =     psi   ** (1/xi) * c ** ((xi -1) / xi)
#             dbase = (1 - psi) ** (1/xi) * d ** ((xi -1) / xi)
#             complement_base = cbase + dbase
#             complement =  complement_base ** (xi / (xi -1))

#             marguc = (psi / c) ** (1 / xi) * complement ** ((1 - xi / eis) / xi)

#     return marguc  

# @njit
# def margutil_d(c, d, psi=0.5, eis=1, xi=1):
    
#     if xi==1:
#         margud =  (1-psi) * c ** (psi * (1 - 1/eis)) * d ** ((1 - psi) * (1 - 1/eis) - 1)

#     else:
#         if eis==xi:
#             margud = ((1-psi) / d) ** (1 / xi)
#         else:
#             cbase =     psi   ** (1/xi) * c ** ((xi -1) / xi)
#             dbase = (1 - psi) ** (1/xi) * d ** ((xi -1) / xi)
#             complement_base = cbase + dbase
#             complement =  complement_base ** (xi / (xi -1))

#             margud = ((1-psi) / d) ** (1 / xi) * complement ** ((1 - xi / eis) / xi)

#     return margud       

# @njit
# def margutilinv_c(marguc, d, psi=0.5, eis=1, xi=1):
    
#     if xi==eis:
#         c =  psi / marguc
#         c = (psi ** (1 / xi) / marguc) ** eis 

#     elif xi==1:
#         c =  (marguc / psi) ** (1 / (psi * (1 - 1/eis) - 1)) * d ** ( - (1 - psi) * (1 - 1/eis) / (psi * (1 - 1/eis) - 1))

#     else:        
#         raise NotImplementedError
#         # newton solve
#         # f_uc(c, d, xi, eis, psi) - marguc
#         #     marguc - (psi / c) ** (1 / xi) * complement ** ((1 - xi / eis) / xi)

#     return c 

# def f_invucc(marguc, d, xi=1, eis=1, psi=0.5):
    
#     if xi!=1:
#         cbase =     psi   ** (1/xi) * c ** ((xi -1) / xi)
#         dbase = (1 - psi) ** (1/xi) * d ** ((xi -1) / xi)
#         complement_base = cbase + dbase
#         complement =  complement_base ** (xi / (xi -1))

#         cshare = cbase / complement_base

#         # verify sign
#         jac = ( - 1 / xi * psi ** (1 / xi) * c ** (-1 / xi - 1) * 
#                 (1 - xi * (1 - xi / eis) / (xi -1) * cshare) 
#                 * complement )

#     jac = sp.diags(jac.reshape([np.prod(np.shape(marguc)),]))

#     return jac           


# def f_ucc(c, uc, d, xi=1, eis=1, psi=0.5):
#     c = c.reshape(np.shape(uc))
    
#     if xi==1:
#         marguc = f_uc(c, uc, d, xi=xi, eis=eis, psi=psi)

#         jac =  (psi * (1 - 1/eis) - 1) * marguc / c

#     else:
#         cbase =     psi   ** (1/xi) * c ** ((xi -1) / xi)
#         dbase = (1 - psi) ** (1/xi) * d ** ((xi -1) / xi)
#         complement_base = cbase + dbase
#         complement =  complement_base ** ((1 - xi / eis) / (xi -1))

#         cshare = cbase / complement_base

#         jac = ( - 1 / xi * psi ** (1 / xi) * c ** (-1 / xi - 1) * 
#                 (1 - xi * (1 - xi / eis) / (xi -1) * cshare) 
#                 * complement )

#     jac = sp.diags(jac.reshape([np.prod(np.shape(c)),]))


#     return jac        
              