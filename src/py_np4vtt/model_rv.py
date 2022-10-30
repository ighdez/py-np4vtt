#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Modules to configure and estimate a Random Valuation model."""
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize
from numdifftools import Hessian
import numpy as np

from py_np4vtt.data_format import ModelArrays


@dataclass
class ConfigRV:
    """Configuration class of the random valuation model.
    
    This class stores the configuration parameters of a random valuation
    model and performs integrity checks before being passed to the model 
    object.
    
    Parameters
    ----------
    startScale : float
        Starting value of the scale parameter
    startVTT : float
        Starting value of the VTT parameter
    maxIterations : int
        Maximum number of iterations of the estimation routine.
    """
    startScale: float
    startVTT: float

    maxIterations: int

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.startScale > 0:
            errorList.append('Scale starting value must be positive.')

        if not self.maxIterations > 0:
            errorList.append('Max iterations must be greater than zero.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelRV:
    """Random valuation model.
    
    This is the model class that prepares the data and estimates 
    the random valuation model [1].
    
    Parameters
    -----------
    params : ConfigRV
        A configuration class of a random valuation model.
    arrays : ModelArrays
        Model arrays created with `make_modelarrays`
        
    Attributes
    ----------
    None.

    References
    ----------
    [1] Ojeda-Cabral, Manuel, Richard Batley, and Stephane Hess. "The 
    value of travel time: random utility versus random valuation." 
    Transportmetrica A: Transport Science 12.3 (2016): 230-248.
    """
    def __init__(self, cfg: ConfigRV, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def run(self):
        """Estimates the random valuation model.
        
        Parameters
        ----------
        None.

        Returns
        -------
        x : np.ndarray
            The estimated parameters of the scale and VTT parameter.
        se : np.ndarray
            The standard errors of the estimated parameters
        init_ll : float
            Log-likelihood at the starting values
        ll : float
            Log-likelihood in the optimum.
        exitflag : int
            Exit flag of the optimisation routine. If `exitflag=0`, the 
            optimisation succeeded. Otherwise, check the configuration parameters.
        """
        # Set vector of starting values of parameters to estimate
        x0 = np.array([self.cfg.startScale, self.cfg.startVTT])

        BVTT=self.arrays.BVTT.flatten()
        y_regress=self.arrays.Choice.flatten()

        init_ll = -ModelRV.objectiveFunction(x0, BVTT, y_regress)

        # Starting arguments and values for minimizer
        argTuple = (BVTT, y_regress)

        # Start minimization routine
        results = minimize(ModelRV.objectiveFunction, x0, args=argTuple, method='L-BFGS-B',options={'gtol': 1e-6,'maxiter': self.cfg.maxIterations})

        # Collect results
        x = results['x']
        hess = Hessian(ModelRV.objectiveFunction,method='forward')(x, BVTT, y_regress)
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        ll = -results['fun']
        exitflag = results['status']

        return x, se, init_ll, ll, exitflag

    @staticmethod
    def objectiveFunction(x: np.ndarray, BVTT: np.ndarray, y_regress: np.ndarray):
        # Separate parameters: x is the estimated (multi-dimensional) parameter
        scale, VTT = x

        # Create value functions
        V1 = scale * BVTT
        V2 = scale * VTT

        dV = V2 - V1
        dV[dV>700] = 700

        # Create choice probability and Log-likelihood
        p = 1 / (1 + np.exp(-dV))
        ll = - np.sum(np.log(p * (y_regress == 1) + (1 - p) * (y_regress == 0)))

        # Return choice probability
        return ll