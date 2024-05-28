## SeqSpaceNK.py 

# Import packages
import numpy as np
import scipy as sp
import time 

# Solve steady state of the model using the Newton-Raphson method
class solveSeqSpaceNK:
    
    # Define the parameters
    def __init__(self, 
        sigma   = 1, 
        beta    = 0.99, 
        delta   = 0.025, 
        zeta    = 1, 
        alpha   = 1/3, 
        theta   = 0.75, 
        epsilon = 6, 
        psi     = 1, 
        phi_pi  = 1.5, 
        phi_y   = 0.5, 
        horizon = 100):

        self.sigma   = sigma
        self.beta    = beta
        self.delta   = delta
        self.zeta    = zeta
        self.alpha   = alpha
        self.theta   = theta 
        self.epsilon = epsilon
        self.psi     = psi
        self.phi_pi  = phi_pi
        self.phi_y   = phi_y
        self.horizon = horizon

        self.rho     = -np.log(self.beta)
        self.omega_c = 1 - (self.delta * self.alpha * (1 - 1/self.epsilon)) / ( self.rho + self.delta)

    # Solve the model using the sequential space method
    def solve_seqspace(self, shocks): 

        start = time.time()
        
        # Order of variables: i, q, y, c, w_real, rk_real, mc_real, pi, r 

        # define sparse identity, above-diagonal sparse matrix, below-diagonal sparse matrix, and zero matrix
        I   = sp.sparse.eye(self.horizon)
        Ip1 = sp.sparse.diags([np.ones(self.horizon-1)], [1], (self.horizon, self.horizon))
        Im1 = sp.sparse.diags([np.ones(self.horizon-1)], [-1], (self.horizon, self.horizon))
        Z   = sp.sparse.csr_matrix((self.horizon, self.horizon)) # sparse empty matrix of size self.horizon x self.horizon

        # Euler equation block
        PhiEE_i = Z
        PhiEE_q = Z
        PhiEE_y = Z
        PhiEE_c = -I + Ip1
        PhiEE_w_real = Z
        PhiEE_rk_real = Z
        PhiEE_mc_real = Z
        PhiEE_pi = 1/self.sigma * Ip1
        PhiEE_r = -1/self.sigma * I

        # Tobin's Q block
        PhiQ_i = Z
        PhiQ_q = self.beta * Ip1 - I
        PhiQ_y = Z
        PhiQ_c = Z
        PhiQ_w_real = Z
        PhiQ_rk_real = (1-self.beta*(1-self.delta)) * Ip1
        PhiQ_mc_real = Z
        PhiQ_pi = Ip1
        PhiQ_r = - I

        # combine matrix blocks in a single sparse matrix with the following structure:
        dHdY = sp.sparse.bmat([[PhiEE_i, PhiEE_q, PhiEE_y, PhiEE_c, PhiEE_w_real, PhiEE_rk_real, PhiEE_mc_real, PhiEE_pi, PhiEE_r],
                               [PhiQ_i, PhiQ_q, PhiQ_y, PhiQ_c, PhiQ_w_real, PhiQ_rk_real, PhiQ_mc_real, PhiQ_pi, PhiQ_r]])

        assert dHdY.shape == (2*self.horizon, 9*self.horizon)

        # Construct dY/dU and dY/dZ using the blocks. 

        ### Household investment block ###
        # Household investment block: nominal interest rate
        Phi_ik = 1/self.delta * I - ((1-self.delta)/self.delta) * Im1
        Phi_in = Z
        Phi_ia = Z
        Phi_inu = Z

        # Household investment block: Tobin's q

        # intermediate effect
        PhiIE_qi = self.zeta * I

        # total effect
        Phi_qk = PhiIE_qi * Phi_ik - self.zeta * Im1
        Phi_qn = PhiIE_qi * Phi_in
        Phi_qa = PhiIE_qi * Phi_ia
        Phi_qnu = PhiIE_qi * Phi_inu

        # combine household investment block matrices
        dYHHinvdU = sp.sparse.bmat([[Phi_ik, Phi_in],
                                    [Phi_qk, Phi_qn]])

        dYHHinvdZ = sp.sparse.bmat([[Phi_ia, Phi_inu],
                                    [Phi_qa, Phi_qnu]])

        assert dYHHinvdU.shape == (2*self.horizon, 2*self.horizon)
        assert dYHHinvdZ.shape == (2*self.horizon, 2*self.horizon)

        ### Aggregate resource constraint ###

        Phi_yk = self.alpha * Im1
        Phi_yn = (1-self.alpha) * I
        Phi_ya = I
        Phi_ynu = Z

        # combine aggregate resource constraint matrices
        dYARdU = sp.sparse.bmat([[Phi_yk, Phi_yn]])

        dYARdZ = sp.sparse.bmat([[Phi_ya, Phi_ynu]])

        assert dYARdU.shape == (1*self.horizon, 2*self.horizon)
        assert dYARdZ.shape == (1*self.horizon, 2*self.horizon)

        ### Goods market clearing ###

        # intermediate effect
        PhiIE_cy = 1/self.omega_c * I
        PhiIE_ci = - (1-self.omega_c)/self.omega_c * I 

        # total effect
        Phi_ck  = PhiIE_cy * Phi_yk + PhiIE_ci * Phi_ik
        Phi_cn  = PhiIE_cy * Phi_yn + PhiIE_ci * Phi_in
        Phi_ca  = PhiIE_cy * Phi_ya + PhiIE_ci * Phi_ia
        Phi_cnu = PhiIE_cy * Phi_ynu + PhiIE_ci * Phi_inu

        # combine goods market clearing matrices
        dYGMdU = sp.sparse.bmat([[Phi_ck, Phi_cn]])
        dYGMdZ = sp.sparse.bmat([[Phi_ca, Phi_cnu]])

        assert dYGMdU.shape == (1*self.horizon, 2*self.horizon)
        assert dYGMdZ.shape == (1*self.horizon, 2*self.horizon)
        
        ### Household labor supply ###
        
        # intermediate effect
        PhiIE_w_realn = self.psi * I
        PhiIE_w_realc = self.sigma * I

        # total effect
        Phi_w_realk  = PhiIE_w_realc * Phi_ck
        Phi_w_realn  = PhiIE_w_realn + PhiIE_w_realc * Phi_cn
        Phi_w_reala  = PhiIE_w_realc * Phi_ca
        Phi_w_realnu = PhiIE_w_realc * Phi_cnu

        # combine household labor supply matrices
        dYHLSdU = sp.sparse.bmat([[Phi_w_realk, Phi_w_realn]])
        dYHLSdZ = sp.sparse.bmat([[Phi_w_reala, Phi_w_realnu]])

        assert dYHLSdU.shape == (1*self.horizon, 2*self.horizon)
        assert dYHLSdZ.shape == (1*self.horizon, 2*self.horizon)

        ### Firm block matrices ###

        # firm block matrices: rental cost

        # intermediate effect
        PhiIE_rk_realw_real = I
        PhiIE_rk_realn = I
        PhiIE_rk_realk = -Im1

        # total effect
        Phi_rk_realk = PhiIE_rk_realw_real * Phi_w_realk + PhiIE_rk_realk 
        Phi_rk_realn = PhiIE_rk_realw_real * Phi_w_realn + PhiIE_rk_realn 
        Phi_rk_reala = PhiIE_rk_realw_real * Phi_w_reala
        Phi_rk_realnu = PhiIE_rk_realw_real * Phi_w_realnu

        # firm block matrices: marginal cost
        
        # intermediate effect
        PhiIE_mc_realr_rk_real = self.alpha * I
        PhiIE_mc_realw_real = (1-self.alpha) * I
        
        # total effect
        Phi_mc_realk = PhiIE_mc_realr_rk_real * Phi_rk_realk + PhiIE_mc_realw_real * Phi_w_realk
        Phi_mc_realn = PhiIE_mc_realr_rk_real * Phi_rk_realn + PhiIE_mc_realw_real * Phi_w_realn
        Phi_mc_reala = PhiIE_mc_realr_rk_real * Phi_rk_reala + PhiIE_mc_realw_real * Phi_w_reala - I
        Phi_mc_realnu = PhiIE_mc_realr_rk_real * Phi_rk_realnu + PhiIE_mc_realw_real * Phi_w_realnu

        # firm block matrices: inflation

        # intermediate effect
        PhiIE_pi_mc_real = (1-self.theta)*(1-self.beta*self.theta)/self.theta * sp.sparse.linalg.inv( I - self.beta*Ip1)

        # total effect
        Phi_pik = PhiIE_pi_mc_real * Phi_mc_realk
        Phi_pin = PhiIE_pi_mc_real * Phi_mc_realn
        Phi_pia = PhiIE_pi_mc_real * Phi_mc_reala
        Phi_pinu = PhiIE_pi_mc_real * Phi_mc_realnu

        # combine firm block matrices
        dYFdU = sp.sparse.bmat([[Phi_rk_realk, Phi_rk_realn],
                                [Phi_mc_realk, Phi_mc_realn],
                                [Phi_pik, Phi_pin]])
        
        dYFdZ = sp.sparse.bmat([[Phi_rk_reala, Phi_rk_realnu],
                                [Phi_mc_reala, Phi_mc_realnu],
                                [Phi_pia, Phi_pinu]])

        assert dYFdU.shape == (3*self.horizon, 2*self.horizon)
        assert dYFdZ.shape == (3*self.horizon, 2*self.horizon)

        ### Monetary policy ###

        # intermediate effect
        PhiIE_rpi = self.phi_pi * I
        PhiIE_ry = self.phi_y * I

        # total effect  
        Phi_rk = PhiIE_rpi * Phi_pik + PhiIE_ry * Phi_yk
        Phi_rn = PhiIE_rpi * Phi_pin + PhiIE_ry * Phi_yn
        Phi_ra = PhiIE_rpi * Phi_pia + PhiIE_ry * Phi_ya
        Phi_rnu = PhiIE_rpi * Phi_pinu + PhiIE_ry * Phi_ynu + I

        # combine monetary policy matrices
        dYMPdU = sp.sparse.bmat([[Phi_rk, Phi_rn]])
        dYMPdZ = sp.sparse.bmat([[Phi_ra, Phi_rnu]])

        assert dYMPdU.shape == (1*self.horizon, 2*self.horizon)
        assert dYMPdZ.shape == (1*self.horizon, 2*self.horizon)

        # stack dYHHinvdU, dYARdU, dYGMdU, dYHLSdU, dYFdU, dYMPdU to get dYdU
        dYdU = sp.sparse.bmat([[dYHHinvdU],
                               [dYARdU],
                               [dYGMdU],
                               [dYHLSdU],
                               [dYFdU],
                               [dYMPdU]])
        
        # stack dYHHinvdZ, dYARdZ, dYGMdZ, dYHLSdZ, dYFdZ, dYMPdZ to get dYdZ
        dYdZ = sp.sparse.bmat([[dYHHinvdZ],
                               [dYARdZ],
                               [dYGMdZ],
                               [dYHLSdZ],
                               [dYFdZ],
                               [dYMPdZ]])
        
        assert dYdU.shape == (9*self.horizon, 2*self.horizon)
        assert dYdZ.shape == (9*self.horizon, 2*self.horizon)

        # compute dHdU using the chain rule dHdU = dHdY @ dYdU (@ is the python matrix multiplication operator)
        dHdU = dHdY @ dYdU 

        # compute dHdZ using the chain rule dHdZ = dHdY @ dYdZ (@ is the python matrix multiplication operator)
        dHdZ = dHdY @ dYdZ

        assert sp.sparse.issparse(dHdZ) == True
        assert sp.sparse.issparse(dHdU) == True

        assert dHdU.shape == (2*self.horizon, 2*self.horizon)
        assert dHdZ.shape == (2*self.horizon, 2*self.horizon)

        # compute the Jacobian of the model
        dUdZ = - sp.sparse.linalg.spsolve(dHdU, dHdZ)
        dYdZ = dYdU @ dUdZ + dYdZ

        dXdZ = sp.sparse.bmat([[dUdZ],
                               [dYdZ]])

        assert dUdZ.shape == (2*self.horizon, 2*self.horizon)
        assert dYdZ.shape == (9*self.horizon, 2*self.horizon)
        assert dXdZ.shape == (11*self.horizon, 2*self.horizon)

        # compute impulse response functions
        X = dXdZ @ shocks

        print(f'Elapsed time is {time.time()-start:.2f} seconds.')    

        return X

