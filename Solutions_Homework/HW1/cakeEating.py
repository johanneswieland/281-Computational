# cakeEating.py

# Packages
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import time

# Class 1: Infinite-Horizon Cake Eating using grid search
class CakeEatingGS: 
    
    # Initialize instance's attributes
    def __init__(self, #Instance of class    
                 
                # Default structural parameters: 
                beta = 0.99, # Discount factor
                delta = 0, # Depcreciation rate
                xmin = 1e-10, # Minimum value of asset grid
                xmax = 1, # Maximum value of asset grid

                # Default simulation parameters: 
                max_iter = 5000,
                tolerance = 1e-10, 
                grid_size_x = 100, 
                grid_size_c = 500, 
                xgrid_curvature = 2, 
                cgrid_curvature = 2): 

        self.beta = beta
        self.delta = delta
        self.xmin = xmin
        self.xmax = xmax

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.grid_size_x = grid_size_x
        self.grid_size_c = grid_size_c
        self.xgrid_curvature = xgrid_curvature
        self.cgrid_curvature = cgrid_curvature

        # Set up asset and consumption grid
        self.setup_grid()

        self.v0 = 1/(1-self.beta) * self.utility(self.x) # Default v0

        # True value function
        self.v_true = np.log(1-self.beta) / (1-self.beta) + \
                    ( self.beta / (1-self.beta)**2 ) * np.log(self.beta) + \
                    np.log(1-self.delta) / ((1-self.beta)**2) + \
                    1/(1-self.beta) * np.log(self.x)

        # True consumption policy function 
        self.c_true = (1-self.beta)*(1-self.delta)*self.x

    # Utility
    def utility(self, c):
        return np.log(c)

    # Set up grid
    def setup_grid(self): 
        
        # Define asset grid with more points in low x regions
        gridvec_x = np.linspace(0, 1, self.grid_size_x)
        self.x = self.xmin + (self.xmax - self.xmin)*gridvec_x**self.xgrid_curvature

        # Asset grid for interpolation
        self.xinterp = self.x.copy()
        self.xinterp[0] = 0

        # Consumption grid
        gridvec_c = np.linspace(0, 1, self.grid_size_c)
        
        cmax = (1-self.delta)
        cmin = (1-self.delta)*self.x[0] 
        self.c_candidates = cmin + (cmax - cmin)*gridvec_c**self.cgrid_curvature

        # Savings grid
        self.xprime_candidates = (1-self.delta)*self.x[np.newaxis, :] - self.c_candidates[:, np.newaxis]

    # Interpolate 
    def lin_interp(self, v):
        '''
        Need to make sure that for each column, there are no penalty values only 
        -> use xinterp to let agents choose at least consumption in the first grid
        '''
        return interp1d(self.xinterp, v, kind='linear', bounds_error=False, fill_value=(-1e100, 0))

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

        self.v = self.v0.copy()
        v_old = self.v0.copy()
        self.c_policy = np.zeros_like(self.x)
        self.savings_policy = np.zeros_like(self.x)

        for iteration in range(self.max_iter):
            
            Tv, c_index = self.bellman_operation(v_old)

            # Update policy and value functions
            self.c_policy = self.c_candidates[c_index]
            self.savings_policy = (1-self.delta)*self.x - self.c_policy
            self.v = Tv

            # Check convergence
            if (np.max(np.abs(Tv - v_old)) < self.tolerance): 
                print(f'Converged in {iteration} iterations. Elapsed time is {time.time()-start:.2f} seconds.')    
                break

            v_old = Tv.copy()

        else: # else executed if loop finishes
            print("Warning: Model did not converge.")
    
    # Solve consumption and savings path
    def solve_path(self, horizon = 50):
        c_path = np.zeros((horizon, self.grid_size_x))
        savings_path = np.zeros((horizon, self.grid_size_x))

        c_path[0,:] = self.c_policy
        savings_path[0,:] = self.savings_policy
        
        savings_interp = interp1d(self.xinterp, self.savings_policy, kind='linear', bounds_error=True)
        c_interp = interp1d(self.xinterp, self.c_policy, kind='linear', bounds_error=True)
        
        for t in range(1, horizon):
            savings_path[t,:] = savings_interp(savings_path[t-1,:]) 
            c_path[t,:] = c_interp(savings_path[t-1,:]) 

        return c_path, savings_path

        
# Class 2: Finite-Horizon Cake Eating using grid search
class CakeEatingGS_FH: 
    
    # Initialize instance's attributes
    def __init__(self, #Instance of class    
                 
                # Default structural parameters: 
                beta = 0.99, # Discount factor
                delta = 0, # Depcreciation rate
                xmin = 1e-10, # Minimum value of asset grid
                xmax = 1, # Maximum value of asset grid

                # Default simulation parameters: 
                horizon = 5000,
                grid_size_x = 100, 
                grid_size_c = 500, 
                xgrid_curvature = 2, 
                cgrid_curvature = 2): 

        self.beta = beta
        self.delta = delta
        self.xmin = xmin
        self.xmax = xmax

        self.horizon = horizon 
        self.grid_size_x = grid_size_x
        self.grid_size_c = grid_size_c
        self.xgrid_curvature = xgrid_curvature
        self.cgrid_curvature = cgrid_curvature

        # Set up asset and consumption grid
        self.setup_grid()

    # Utility
    def utility(self, c):
        return np.log(c)

    # Set up grid
    def setup_grid(self): 
        
        # Define asset grid with more points in low x regions
        gridvec_x = np.linspace(0, 1, self.grid_size_x)
        self.x = self.xmin + (self.xmax - self.xmin)*gridvec_x**self.xgrid_curvature

        # Asset grid for interpolation
        self.xinterp = self.x.copy()
        self.xinterp[0] = 0

        # Consumption grid
        gridvec_c = np.linspace(0, 1, self.grid_size_c)
        
        cmax = (1-self.delta)
        cmin = (1-self.delta)*self.x[0] 
        self.c_candidates = cmin + (cmax - cmin)*gridvec_c**self.cgrid_curvature

        # Savings grid
        self.xprime_candidates = (1-self.delta)*self.x[np.newaxis, :] - self.c_candidates[:, np.newaxis]

    # Interpolate 
    def lin_interp(self, v):
        '''
        Need to make sure that for each column, there are no penalty values only 
        -> use xinterp to let agents choose at least consumption in the first grid
        '''
        return interp1d(self.xinterp, v, kind='linear', bounds_error=False, fill_value=(-1e100, 0))

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

        v_old = np.zeros(self.grid_size_x)
        self.v = np.zeros((self.horizon + 1, self.grid_size_x))
        self.c_policy = np.zeros((self.horizon, self.grid_size_x))
        self.savings_policy = np.zeros((self.horizon, self.grid_size_x))

        # Backward induction 
        for n in range(self.horizon-1, -1, -1):
            
            Tv, c_index = self.bellman_operation(v_old)

            # Update policy and value functions
            self.c_policy[n,:] = self.c_candidates[c_index] 
            self.savings_policy[n,:] = (1-self.delta)*self.x - self.c_policy[n,:]
            self.v[n,:] = Tv

            v_old = Tv.copy()
        print(f'Elapsed time is {time.time()-start:.2f} seconds.')    

    # Solve consumption and savings path
    def solve_path(self):

        v_path = np.zeros((self.horizon + 1, self.grid_size_x))
        c_path = np.zeros((self.horizon, self.grid_size_x))
        savings_path = np.zeros((self.horizon, self.grid_size_x))
        
        v_path[0,:] = self.v[0,:]
        c_path[0,:] = self.c_policy[0,:]
        savings_path[0,:] = self.savings_policy[0,:]
        
        for t in range(1, self.horizon):
            savings_interp = interp1d(self.xinterp, self.savings_policy[t-1,:], kind='linear', bounds_error=True)
            c_interp = interp1d(self.xinterp, self.c_policy[t-1,:], kind='linear', bounds_error=True)
            v_interp = interp1d(self.xinterp, self.v[t-1,:], kind='linear', bounds_error=True)

            savings_path[t,:] = savings_interp(savings_path[t-1,:]) 
            c_path[t,:] = c_interp(savings_path[t-1,:]) 
            v_path[t,:] = v_interp(savings_path[t-1,:]) 

        return c_path, savings_path, v_path
