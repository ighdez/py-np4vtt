#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Modules to configure and estimate a Logistic regression-based model."""
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize
from numdifftools import Hessian
import numpy as np
import warnings

from py_np4vtt.data_format import ModelArrays

warnings.filterwarnings('ignore')

@dataclass
class ConfigLogistic:
    """Configuration class of the logistic regression (ANN-0) model.
    
    This class stores the configuration parameters of a logistic regression 
    model and performs integrity checks before being passed to the model 
    object.
    
    Parameters
    ----------
    startScale : float
        The starting value of the scale parameter.
    startIntercept : float
        The starting value of the intercept parameter.
    startParameter: float
        The starting value of the VTT parameter.
    maxIterations: int
        Maximum number of iterations of the estimation routine.
    seed: Optional[int]
        Random seed
    """
    startScale: float
    startIntercept: float
    startParameter: float

    maxIterations: int

    seed: Optional[int]

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.startScale > 0:
            errorList.append('Scale starting value must be positive.')

        if not self.maxIterations > 0:
            errorList.append('Max iterations must be greater than zero.')

        if not self.seed >= 0:
            errorList.append('Seed must be non-negative.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelLogistic:
    """Logistic regression model.
    
    This is the model class that prepares the data and estimates 
    the logistic regression.
    
    Parameters
    -----------
    params : ConfigLogistic
        A configuration class of a logistic regression model.
    arrays : ModelArrays
        Model arrays created with `make_modelarrays`
        
    Attributes
    ----------
    None.
    """
    def __init__(self, cfg: ConfigLogistic, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def run(self):
        """Estimates the logistic regression model.
        
        Parameters
        ----------
        None.

        Returns
        -------
        x : np.ndarray
            The estimated parameters of the scale, intercept and VTT parameter.
        se : np.ndarray
            The standard errors of the estimated parameters
        vtt: np.ndarray
            The estimated VTT per respondent, based in the estimated 
            parameters and the sample.
        init_ll : float
            Log-likelihood at the starting values
        ll : float
            Log-likelihood in the optimum.
        exitflag : int
            Exit flag of the optimisation routine. If `exitflag=0`, the 
            optimisation succeeded. Otherwise, check the configuration parameters.
        """
        # Use passed seed if desired
        if self.cfg.seed:
            np.random.seed(self.cfg.seed)

        # Prepare data
        i_obs_y = \
            np.tile(np.random.randint(1, self.arrays.T, size=(self.arrays.NP, 1)), (1, self.arrays.T)) == \
            np.tile(np.arange(1, self.arrays.T + 1), (self.arrays.NP, 1))
        
        i_obs_x = (i_obs_y == 0)

        # Set vector of starting values of parameters to estimate
        x0 = np.array([self.cfg.startScale, self.cfg.startIntercept, self.cfg.startParameter])

        BVTT=np.sum(i_obs_y * self.arrays.BVTT, axis=1)
        sumYBVTT=np.sum(i_obs_x * self.arrays.BVTT * self.arrays.Choice, axis=1)
        y_regress=np.sum(self.arrays.Choice * i_obs_y, axis=1)

        # LL at the start values
        init_ll = -ModelLogistic.objectiveFunction(x0, sumYBVTT, BVTT, y_regress)

        # Starting arguments and values for minimizer
        argTuple = (sumYBVTT, BVTT, y_regress)
        
        # Start minimization routine
        results = minimize(ModelLogistic.objectiveFunction, x0, args=argTuple, method='L-BFGS-B',options={'gtol': 1e-6,'maxiter': self.cfg.maxIterations})

        # Collect results
        x = results['x']
        hess = Hessian(ModelLogistic.objectiveFunction,method='forward')(x,sumYBVTT, BVTT, y_regress)
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        ll = -results['fun']
        exitflag = results['status']

        # Compute VTT
        vtt = x[1] + x[2]*((self.arrays.T-1)/self.arrays.T)*np.sum(self.arrays.Choice*self.arrays.BVTT,1)
        vtt = np.concatenate((0.,vtt),axis=None)
        
        return x, se, vtt, init_ll ,ll, exitflag

    @staticmethod
    def objectiveFunction(x: np.ndarray, sumYBVTT: np.ndarray, BVTT: np.ndarray, y_regress: np.ndarray):
        
        # Separate parameters: x is the estimated (multi-dimensional) parameter
        scale, intercept, parameter = x

        # Create value functions
        VTT = intercept + parameter * sumYBVTT
        V1 = scale * BVTT
        V2 = scale * VTT

        dV = V1 - V2
        dV[dV>700] = 700

        # Create choice probability and Log-likelihood
        p = 1 / (1 + np.exp(-dV))
        ll = np.log(p * (y_regress == 0) + (1 - p) * (y_regress == 1))
        ll[~np.isfinite(ll)] = 0
        
        # Return choice probability
        return -np.sum(ll)