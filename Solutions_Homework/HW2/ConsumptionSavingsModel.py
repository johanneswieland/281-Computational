# -------------------------------------------------------------- #
# ConsumptionSavingModel.py
# Jung Hyun Kim
# -------------------------------------------------------------- #

# -------------------------------------------------------------- #
# Packages
# -------------------------------------------------------------- #
import numpy as np
from scipy.interpolate import interp1d
import time
from IPython.utils import io

# -------------------------------------------------------------- #
# ConsumptionSavingModel Class
# -------------------------------------------------------------- #

class ConsumptionSavings: 

    # ---------------------------------------------------------- #
    # Initialization
    # ---------------------------------------------------------- #
    def __init__(self, 
                 
        # Default structural parameters: 
        beta        = 0.99, 
        sigma       = 1, 
        y           = 1, 
        amin        = -1, 
        amax        = 1, 
        grid_size_a = 200, 
        gird_size_c = 500,
        asupply     = 0, 

        # Default simulation parameters:
        max_iter        = 5000, 
        max_howard_iter = 100, 
        tolerance       = 1e-7, 
        agrid_curvature = 1,
        cgrid_curvature = 1): 

        # Structural parameters
        self.beta        = beta
        self.sigma       = sigma
        self.y           = y
        self.amin        = amin
        self.amax        = amax
        self.grid_size_a = grid_size_a
        self.grid_size_c = gird_size_c
        self.asupply     = asupply

        # Simulation parameters
        self.max_iter        = max_iter
        self.max_howard_iter = max_howard_iter
        self.tolerance       = tolerance
        self.agrid_curvature = agrid_curvature
        self.cgrid_curvature = cgrid_curvature
        
        # Set up asset and consumption grid
        self.setup_grid()

    # ---------------------------------------------------------- #
    # Grid setup
    # ---------------------------------------------------------- #
    def setup_grid(self):

        # Define asset grid with more points in low x regions
        gridvec_a         = np.linspace(0, 1, self.grid_size_a)
        self.a            = self.amin + (self.amax - self.amin)*gridvec_a**self.agrid_curvature

        # Consumption grid
        self.c_candidates = self.y + np.linspace(-0.5, 0.5, self.grid_size_c)**self.cgrid_curvature

    # ---------------------------------------------------------- #
    # Useful functions
    # ---------------------------------------------------------- #
    # Define utility function
    def utility(self, c): 
        if self.sigma == 1: 
            return np.log(c)
        else: 
            return c**(1-self.sigma)/(1-self.sigma)

    # Define marginal utility function 
    def margutility(self, c):
        if self.sigma == 1: 
            return 1 / c
        else: 
            return c**-self.sigma

    # Define inverse marginal utility function
    def invmargutility(self, uderiv):
        if self.sigma == 1: 
            return 1 / uderiv
        else: 
            return uderiv**(-1/self.sigma)

    # Define derivative function
    def derivative(self, f, x, epsilon=1e-5, *args, **kwargs): 
        return (f(x, *args, **kwargs) - f(x - epsilon, *args, **kwargs)) / (epsilon)

    # ---------------------------------------------------------- #
    # Solver for the HH problem
    # ---------------------------------------------------------- #
    def solve_HH(self, r, solver_HH, update=0.2):

        if solver_HH   == 'VFI':
            return self.solve_HH_VFI(r)
        
        elif solver_HH == 'VFI_Howard':
            return self.solve_HH_VFI_Howard(r)
        
        elif solver_HH == 'PFI':
            return self.solve_HH_PFI(r, update)
        
        elif solver_HH == 'FOC':
            return self.solve_HH_FOC_value(r, update)
        
        else: 
            print('Solver for the HH problem not recognized.')

    # ---------------------------------------------------------- #
    # VFI method
    # ---------------------------------------------------------- #
    def solve_HH_VFI(self, r):

        start              = time.time()
        self.v0            = np.linspace(1, 10, self.grid_size_a)
        v                  = self.v0.copy()
        v_old              = self.v0.copy()
        a_prime_candidates = (1+r)*self.a[np.newaxis, :] + self.y - self.c_candidates[:, np.newaxis] 

        for iteration in range(self.max_iter):

            # Interpolate the value function
            f_interp           =  interp1d(self.a, v_old, kind='linear', bounds_error=False, fill_value=(-1e10, 0))
            v_prime_candidates = f_interp(a_prime_candidates)

            # Calculate utility for all possible consumption choices
            utility_values = self.utility(self.c_candidates[:, np.newaxis]) + self.beta * v_prime_candidates

            # Find the consumption choice that maximizes utility
            Tv      = np.max(utility_values, axis=0)
            c_index = np.argmax(utility_values, axis=0) 
            
            # Update policy and value functions
            c_policy       = self.c_candidates[c_index]
            savings_policy = (1+r)*self.a + self.y - c_policy
            v              = Tv

            # Check convergence
            if np.max(np.abs(Tv-v_old)) < self.tolerance:
                self.time = time.time()-start
                print(f'Converged in {iteration} iterations. Elapsed time of VFI is {self.time:.3f} seconds.')
                break
            
            v_old = Tv.copy()

        else: 
            print('Warning: Model did not converge.')
    
        return v, c_policy, savings_policy

    # ------------------------------------------------------------- #
    # VFI with policy iteration improvement step (Howard Improvement)
    # ------------------------------------------------------------- #
    def solve_HH_VFI_Howard(self, r):

        start              = time.time()
        self.v0            = np.linspace(1, 10, self.grid_size_a)
        v                  = self.v0.copy()
        v_old              = self.v0.copy()
        a_prime_candidates = (1+r)*self.a[np.newaxis, :] + self.y - self.c_candidates[:, np.newaxis] 

        for iteration in range(self.max_iter):

            # Interpolate the value function
            f_interp           = interp1d(self.a, v_old, kind='linear', bounds_error=False, fill_value=(-1e10, 0))
            v_prime_candidates = f_interp(a_prime_candidates)

            # Calculate utility for all possible consumption choices
            utility_values = self.utility(self.c_candidates[:, np.newaxis]) + self.beta * v_prime_candidates

            # Find the consumption choice that maximizes utility
            Tv      = np.max(utility_values, axis=0)
            c_index = np.argmax(utility_values, axis=0) 

            # Update policy and value functions
            c_policy        = self.c_candidates[c_index]
            savings_policy  = (1+r)*self.a + self.y - c_policy
            v               = Tv

            # Howard Iteration
            c_policy_fix = c_policy.copy()
            policy_iteration = 0 
            while policy_iteration < self.max_howard_iter: 

                v_old              = Tv
                f_interp           = interp1d(self.a, v_old, kind='linear', bounds_error=False, fill_value=(-1e10, 0))
                v_prime_candidates = f_interp(a_prime_candidates[c_index, range(self.grid_size_a)])
                Tv                 = self.utility(c_policy_fix) + self.beta * v_prime_candidates

                policy_iteration += 1

            # Check convergence
            if np.max(np.abs(Tv-v_old)) < self.tolerance:
                self.time = time.time()-start
                print(f'Converged in {iteration} iterations. Elapsed time of VFI with Howard improvement is {self.time:.3f} seconds.')
                break
            
            v_old = Tv.copy()

        else: 
            print('Warning: Model did not converge.')

        return v, c_policy, savings_policy

    # ------------------------------------------------------------- #
    # Policy Function Iteration
    # ------------------------------------------------------------- #
    def solve_HH_PFI(self, r, update):

        start = time.time()
        c_old = self.y + np.zeros(self.a.shape)
        c_new = self.y + np.zeros(self.a.shape) 

        iteration = 0
        while iteration < self.max_iter: 

            # start with initial guess for the consumption policy function
            c_old    = update*c_new + (1-update)*c_old 
            aprime   = ( 1 + r ) * self.a + self.y - c_old
                
            # interpolate c_old to get cprime 
            f_interp = interp1d(self.a, c_old, kind='linear', bounds_error=False, fill_value=(1e-5, (1 + r) * self.amax + self.y)) 
            cprime   = f_interp(aprime)

            # the new consumption policy function is
            c_new    = self.invmargutility( self.beta * (1 + r) * self.margutility(cprime) )

            if np.max(np.abs(c_new - c_old)) < self.tolerance:
                self.time = time.time()-start
                print(f'Converged in {iteration} iterations. Elapsed time of PFI is {self.time:.3f} seconds.')
                break
            
            iteration += 1
        
        else: 
            print('Warning: Model did not converge.')
        
        c_policy       = c_new
        savings_policy = (1+r)*self.a + self.y - c_policy        

        return c_policy, savings_policy

    # ------------------------------------------------------------- #
    # FOC from Value Function
    # ------------------------------------------------------------- #
    def solve_HH_FOC_value(self, r, update):

        start      = time.time()
        # Initial guess of derivative of value function
        vderiv_old = np.ones_like(self.a)/( (1+r) + 1) # marginal utility given maximum consumption possible

        iteration = 0
        while iteration < self.max_iter:  
            
            consumption   = self.invmargutility(vderiv_old)
            aprime        = (1 + r) * self.a + self.y - consumption

            f_interp      = interp1d(self.a, vderiv_old, kind='linear', bounds_error=False, fill_value=(1e10, 0))
            vderiv_aprime = f_interp(aprime)
            vderiv_new    = self.beta * (1 + r) * vderiv_aprime
            
            if np.max(np.abs(vderiv_new - vderiv_old)) < self.tolerance:
                self.time = time.time()-start
                print(f'Converged in {iteration} iterations. Elapsed time of FOC method from value function is {self.time:.3f} seconds.')
                break

            vderiv_old = update*vderiv_new.copy() + (1 - update)*vderiv_old

            iteration += 1
        else: 
            print('Warning: Model did not converge.')
        
        c_policy       = self.invmargutility(vderiv_new)
        savings_policy = (1+r)*self.a + self.y - c_policy

        return c_policy, savings_policy

    # ---------------------------------------------------------- #
    # Excess demand function
    # ---------------------------------------------------------- #

    def excess_demand(self, r, solver_HH, update=0.2):
        
        with io.capture_output() as captured: # suppress printing the output
            *_, savings_policy      = self.solve_HH(r, solver_HH, update) # Unpack only the last output (savings policy)

        index                   = np.argmin(np.abs(self.a - self.asupply))
        excess_demand           = savings_policy[index] - self.asupply

        return excess_demand

    # ---------------------------------------------------------- #
    # Solver for interest rate (market clearing)
    # ---------------------------------------------------------- #
    def solve_R(self, min_r, max_r, solver_HH, solver_r, update=0.2):

        # single excess demand function
        single_excess_demand = lambda r: self.excess_demand(r, solver_HH, update)

        # Solve for the interest rate
        if solver_r == 'NR':
            r0      = min_r + (max_r - min_r) / 2
            start   = time.time()
            rstar   = self.NR(single_excess_demand, r0)
            self.time = time.time()-start
            print(f'Elapsed time of Newton-Raphson method is {self.time:.3f} seconds.')
            return rstar
        
        elif solver_r == 'bisection':
            start   = time.time()
            rstar     = self.bisection(single_excess_demand, min_r, max_r)
            self.time = time.time()-start
            print(f'Elapsed time of bisection method is {self.time:.3f} seconds.')
            return rstar
        
        else: 
            print('Solver for intereset rate not recognized.')

    # ---------------------------------------------------------- #
    # Methods for numerical optimization
    # ---------------------------------------------------------- #
    # Bisection method
    def bisection(self, f, a, b, *args, **kwargs):
        
        if f(a, *args, **kwargs) * f(b, *args, **kwargs) > 0:
            print("Bisection method fails.")
            return None
        
        c = a
        while (b-a) >= 1e-8:
            # new midpoint
            c = (a + b) / 2
            if abs(f(c, *args, **kwargs))<1e-8: # if f(c) is small, we consider it as 0
                break
            elif f(c, *args, **kwargs) * f(a, *args, **kwargs) < 0: # if f(c) and f(a) have different signs, the root is in the interval [a, c]
                b = c
            else: # if f(c) and f(b) have same signs, the root is in the interval [c, b]
                a = c
        
        return c
    
    # Newton-Raphson method
    def NR(self, f, x0, tol=1e-8, max_iter_NR=1000, *args, **kwargs):
        x = x0
        for i in range(max_iter_NR):
            x_new = x - f(x, *args, **kwargs) / self.derivative(f,x, *args, **kwargs)
            if abs(x_new - x) < tol:
                break
            x = x_new
        return x_new