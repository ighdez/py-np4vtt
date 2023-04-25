#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
from random import uniform

# Create VTT midpoints
def vtt_midpoints(vtt_grid):
    
    # Calculate midpoints
    vtt_grid_midpoints = ((np.append(vtt_grid, 0) + np.append(0,vtt_grid)))/2
    
    # Replace last value of the VTT midpoints by the last value of the VTT array
    vtt_grid_midpoints[-1] = vtt_grid[-1]

    return vtt_grid_midpoints

# Create predicted VTT for each respondent, based in the choice probability and a VTT grid.
def predicted_vtt(ecdf,grid,NP):
    pp = ((np.append(ecdf, 0)-np.append(0,ecdf)))
    count_data = []
    for n in range(1,len(grid)):
        dat = [uniform(grid[n-1], grid[n]) for p in range(0, np.round((pp[n]*NP)).astype(int))]
        count_data = np.append(count_data,dat)
    return count_data

"""BFGSMin functions"""
# BFGSMin functions
# Written by Jose Ignacio Hernandez

# Import packages
import numpy as np

# BFGS Minimizer function
def _bfgsmin(f,x0,maxiter=1000,tol=np.sqrt(np.finfo(float).eps),verbose=False,difftype='central',diffeps=np.sqrt(np.finfo(float).eps),steptol=1e-30,args=()):


    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize parameters
    x = np.array(x0)                            # Initial value of x
    f_val = f(x,*args)                          # Initial value for objective function
    g0 = _numgr(f,x,difftype,diffeps,*args)      # Initial value of gradient
    H0 = np.eye(x.shape[0])                     # Initial value of approximate Hessian is a positive definite matrix (e.g. identity matrix)
    g_diff = np.inf                             # Initial difference in gradient norm is set to infinity
    c1 = 1e-4                                   # Internal scalar for the Armijo-Goldstein condition (for step size computation)
    lambd = 1                                   # Initial value of the step size
    convergence = -1                            # Set convergence flag to -1. If zero, the algorithm converged

    # Print inital value of the objective function
    if verbose:
        print('Initial F-value: ' + str(round(f_val,2)))

    # Start algorithm
    for iter in range(maxiter):
        
        # If gradient norm is less than tolerance value, convergence is achieved and the loop is broken
        if g_diff < tol:
            convergence = 0

            if verbose:
                print('\nLocal minimum found. G-norm below tolerance')

            break

        # Set step size = 1 on each new iteration
        lambd = 1

        # Construct direction vector and relative gradient

        # d = np.linalg.solve(-H0,np.eye(len(x))) @ g0
        d = np.linalg.inv(-H0) @ g0
        m = d.T @ g0
        # Select step size that satisfies the Armijo-Goldstein condition
        while True:
            
            # If lambd decreases less than tol, then stop and return convergence = 3
            if lambd < steptol:
                if verbose:
                    print('\nLocal minimum possible. Step size tolerance limit reached.')
                convergence = 5
                break

            # Evaluate objective function using current step size
            x1 = x + lambd*d
            f1 = f(x1,*args)

            # Construct test for A-G condition
            ftest = f_val + c1*lambd*m

            # If A-G condition is satisfied (i.e current step size generates a decrease of the objective function), continue with BFGS algorithm
            if (np.isnan(f1) == False) and (f1 <= ftest) and (f1 > 0):
                break

            # ...else, decrease step size
            else:
                lambd = lambd/2

        # If step size calculation loop was broken, then break the algorithm and return convergence code + partial results
        if convergence == 5:
            break

        # BFGS ALGORITHM: construct the improvement and gradient improvement
        g1 = _numgr(f,x1,difftype,diffeps,*args)
        s0 = (lambd*d)[:,np.newaxis]
        y0 = (g1 - g0)[:,np.newaxis]

        # Update Hessian using BFGS formula
        H0 = H0 + (y0 @ y0.T) / (y0.T @ s0) - (((H0 @ s0) @ s0.T) @ H0)/((s0.T @ H0) @ s0)
        
        # Store new gradient and compute the new value of objective function
        x = x1.copy()
        g0 = g1.copy()
        f_val = f1.copy()

        # Compute infinite norm of gradient vector 
        g_diff = np.abs(m)
        # Print output
        if verbose:
            print('Iter No. ' + str(int(iter+1)) + ': F-value: ' + str(round(f_val,2)) + ' / Step size: ' + str(round(lambd,6)) + ' / G-norm: ' + str(round(g_diff,6)))

    # If the algorithm reaches the maximum iterations, it prints convergence message = 2
    if iter == maxiter:
        convergence = 2

    # Return the Hessian approximation
    H = H0

    # Return convergence flag, iterations, final f value, final x value, and final approx. hessian
    return({'convergence': convergence, 'iterations': iter+1, 'fun': f_val, 'x': x, 'hessian': H})

# Numeric gradient fuction
def _numgr(f,param,difftype='forward',eps=np.sqrt(np.finfo(float).eps),*args):    
    
    # Define scalars and initialize vectors
    K = len(param)                          # No. of parameters
    gr = np.full(K,np.nan)                  # Initialize gradient vector
    ej = np.eye(K)*eps                      # Vector of eps
    
    # If difftype == 'central', then:
    if difftype == 'central':
        for k in range(K):
            gr[k] = (f(param + ej[:,k],*args) - f(param - ej[:,k],*args))*0.5/eps

    # ...else, if difftype == 'forward' (default):
    elif difftype == 'forward':
        f0 = f(param,*args)
        for k in range(K):
            gr[k] = (f(param + ej[:,k],*args) - f0)/eps

    # ...else, return error
    else:
        raise ValueError('''difftype must be either 'forward' or 'central' ''')

    # Return gradient vector
    return(gr)

# Numeric Hessian function
class numhess:
    
    def __init__(self,f,eps=np.sqrt(np.finfo(float).eps)):
        self.f = f
        self.eps = eps
    
    def __call__(self,param,*args):
        eps = self.eps
        f = self.f

        # Define scalars and initialize vectors
        K = len(param)                          # No. of parameters
        hs = np.full((K,K),np.nan)              # Initialize Hessian vector
        ej = np.eye(K)*eps                      # Vector of eps

        # f0 = f(param,*args)
        for i in range(K):
            for j in range(K):
                # f1 = f(param + ej[:,i] + ej[:,j],*args)
                # f2 = f(param + ej[:,i],*args)
                # f3 = f(param + ej[:,j],*args)

                # hs[i,j] = (f1-f2-f3+f0)/(eps**2)
                # if i != j:
                #     hs[i,j] = (f1-f2-f3+f0)/(eps**2)
                
                f1 = f(param + ej[:,i] + ej[:,j],*args)
                f2 = f(param + ej[:,i] - ej[:,j],*args)
                f3 = f(param - ej[:,i] + ej[:,j],*args)
                f4 = f(param - ej[:,i] - ej[:,j],*args)

                hs[i,j] = (f1-f2-f3+f4)/(4*eps*eps)
                if i != j:
                    hs[j,i] = (f1-f2-f3+f4)/(4*eps*eps)
    
        return hs
