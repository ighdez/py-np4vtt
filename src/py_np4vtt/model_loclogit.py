#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Modules to configure and estimate a Local logit model."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from py_np4vtt.data_format import ModelArrays
from py_np4vtt.utils import vtt_midpoints, predicted_vtt

@dataclass
class ConfigLocLogit:
    """Configuration class of the local logit model.
    
    This class stores the configuration parameters of a local logit model 
    and performs integrity checks before being passed to the model object.
    
    Parameters
    ----------
    
    minimum : float
        Minimum value of the VTT grid
    maximum : float
        Maximum value of the VTT grid
    supportPoints : int
        Number of support points of the VTT grid. The VTT grid will contain
        `(supportPoints-1)` intervals. Must be greater than zero
    """
    minimum: float
    maximum: float
    supportPoints: int

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.maximum > self.minimum:
            errorList.append('Max must be greater than minimum.')

        if not self.supportPoints > 0:
            errorList.append('No. of support points must be greater than zero.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelLocLogit:
    """Local logit model.
    
    This is the model class that prepares the data and estimates a local 
    logit model [1].
    
    Parameters
    -----------
    params : ConfigLocLogit
        A configuration class of a local logit model.
    arrays : ModelArrays
        Model arrays created with `make_modelarrays`
        
    Attributes
    ----------
    vtt_grid : numpy.ndarray
        The VTT grid created with the specifications of `ConfigLocLogit`.
    vtt_mid : numpy.ndarray
        The mid points of the VTT grid.

    Methods
    -------
    run():
        Estimates the local logit model.
    
    References
    ----------
    [1] Fosgerau, Mogens. "Using nonparametrics to specify a model to measure 
    the value of travel time." Transportation Research Part A: Policy and 
    Practice 41.9 (2007): 842-856.
    """
    def __init__(self, params: ConfigLocLogit, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the midpoints of the VTT grid
        self.vtt_mid = vtt_midpoints(self.vtt_grid)

        # Compute distance between points at each support point
        dist = self.vtt_grid[1] - self.vtt_grid[0]

        # Print message of the support points
        print("Created a VTT grid of " + str(self.params.supportPoints) + \
            " points between " + str(self.params.minimum) + " and " + str(self.params.maximum) + ".")

        print("Distance between points of the VTT grid is " + str(dist))
        
    def run(self):
        """Estimates the local logit model.
        
        Parameters
        ----------
        None.

        Returns
        -------
        p : numpy.ndarray
            The estimates of the cumulative choice probability (CDF), evaluated 
            at each interval of the VTT grid. The first point is always zero 
            while the last point is always equal to the second last for 
            compatibility with plots.
        vtt : numpy.ndarray
            The estimated VTT per respondent, based in the estimated 
            CDF points (`p`) and the sample.
        ll : float
            The log-likelihood function at the optimum of the estimation.
        """
        # Compute the kernel width
        k = np.r_[self.vtt_grid, 0.] - np.r_[0., self.vtt_grid]
        k = k[:-2]
        k[0] = k[1].copy()

        YX = self.arrays.Choice.T.flatten()

        # Perform a weighted logit for each support point
        p = []
        fval = 0.
        for n in range(len(self.vtt_grid)-1):
            x, fval_x = ModelLocLogit.initLocalLogit(n, k[n], self.arrays.BVTT, YX, self.vtt_grid)
            p.append(x[0])
            fval = fval + fval_x
        
        # Return probability array and -ll
        p = np.array(p)
        ll = -fval

        # Compute the predicted VTT at the midpoints
        vtt = predicted_vtt(p,self.vtt_grid,self.arrays.NP)

        # Add point 0 in the estimated CDF and repeat last point to make coincide with point zero and last point in the VTT mid point
        p = np.concatenate((0,p,p[-1]),axis=None)

        return p, vtt, ll

    @staticmethod
    def initLocalLogit(n, k, BVTT, YX, vtt_grid):
        # Get observations
        BVTT_flat = BVTT.T.flatten()
        xn = BVTT_flat[(BVTT_flat > (vtt_grid[n]-k)) & (BVTT_flat < (vtt_grid[n] + k))]
        x0 = vtt_grid[n]
        y_local = YX[(BVTT_flat > (vtt_grid[n]-k)) & (BVTT_flat < (vtt_grid[n] + k))]
        dist = np.abs(x0-xn)
        weight = (k-dist)/k

        # Search function
        coef_start = np.array([0., 0.])
        args = (y_local, xn, x0, weight)
        results = minimize(ModelLocLogit.objectiveFunction, coef_start, args=args, method='L-BFGS-B',options={'gtol': 1e-6})

        # Collect results
        x = results['x']
        fval = results['fun']

        # Convert to probabilities
        x = 1 - (np.exp(x)/(1+np.exp(x)))

        return x, fval

    @staticmethod
    def objectiveFunction(coef: np.ndarray, y_local: np.ndarray, xn: np.ndarray, x0: np.ndarray, weight: np.ndarray):
        acc = coef[0] + coef[1]*(xn-x0)
        P = np.exp(acc)/(1+np.exp(acc))
        LL = np.log(P*(y_local == 1) + (1-P)*(y_local == 0))
        LLw = -sum(weight*LL)

        return LLw
