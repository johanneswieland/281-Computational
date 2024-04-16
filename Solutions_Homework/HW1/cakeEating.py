# cake_eating.py

# Packages
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import time

# Class 1: Cake Eating using grid search (No Depreciation)
class CakeEatingGS: 
    
    # Initialize instance's attributes
    def __init__(self, #Instance of class    
                 
                # Default structural parameters: 
                beta = 0.99, # Discount factor
                xmin = 1e-10, # Minimum value of asset grid
                xmax = 1, # Maximum value of asset grid
                v0 = None, # Initial guess of value
                delta = 0, # Depcreciation rate

                # Default simulation parameters: 
                max_iter = 5000, 
                tolerance = 1e-10, 
                grid_size_x = 100, 
                grid_size_c = 500): 
        
        self.beta = beta
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.grid_size_x = grid_size_x
        self.grid_size_c = grid_size_c

        # Set up asset and consumption grid
        self.setup_grid()
        
        # Initialize value function 
        if v0 == None:
            self.v0 = 1/(1-self.beta) * self.utility(self.x) # Default v0
        elif v0 == 0:
            self.v0 = np.zeros_like(self.x)

        # True value function (only for no depreciation)
        self.v_true_delta0 = np.log(1-self.beta)/(1-self.beta) + self.beta/((1-self.beta))**2 * np.log(self.beta) + 1/(1-self.beta) * np.log(self.x)

        # True consumption policy function (only for no depreciation)
        self.c_true_delta0 = (1-self.beta)*self.x

    # Utility
    def utility(self, c):
        return np.log(c)

    # Set up grid
    def setup_grid(self): 
        
        # Define asset grid with more points in low x regions
        gridvec_x = np.linspace(0, 1, self.grid_size_x)
        self.x = self.xmin + (self.xmax - self.xmin)*gridvec_x**4

        # Asset grid for interpolation
        self.xinterp = self.x.copy()
        self.xinterp[0] = 0

        # Consumption grid
        gridvec_c = np.linspace(0, 1, self.grid_size_c)
        cmax = (1-self.delta)
        cmin = (1-self.delta)*self.x[0] 
        self.c_candidates = cmin + (cmax - cmin)*gridvec_c**4
        self.xprime_candidates = (1-self.delta)*self.x[np.newaxis, :] - self.c_candidates[:, np.newaxis]

    # Interpolate 
    def lin_interp(self, v):
        '''
        Need to make sure that for each column, there are no penalty values only 
        -> use xinterp to let agents choose first consumption
        '''
        return interp1d(self.xinterp, v, kind='linear', bounds_error=False, fill_value=(-1e10,0)) 
    
    # Bellman operation
    def bellman_operation(self, v_old):
        f_interp = self.lin_interp(v_old)
        vprime_candidates = f_interp(self.xprime_candidates)
        utility_values = self.utility(self.c_candidates[:, np.newaxis]) + self.beta * vprime_candidates

        Tv = np.max(utility_values, axis=0)
        c_index = np.argmax(utility_values, axis=0)
        return Tv, c_index
    
    # Solve Bellman equation
    def solve_model(self): 

        start = time.time()

        v_old = self.v0.copy()

        for iteration in range(self.max_iter):
            Tv, c_index = self.bellman_operation(v_old)

            # Update policy and value functions
            self.c_policy = self.c_candidates[c_index]
            self.v = Tv
            
            # Check convergence
            if np.max(np.abs(Tv - v_old)) < self.tolerance: 
                print(f'Converged in {iteration} iterations. Elapsed time is {time.time()-start:.2f} seconds.')    
                break

            v_old = Tv.copy()

        else: # else executed if loop finishes
            print("Warning: Model did not converge or is a Finite-Horizon Problem.")

            
