#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize
from numdifftools import Hessian
import numpy as np

from py_np4vtt.data_format import ModelArrays


@dataclass
class ConfigRV:
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
    def __init__(self, cfg: ConfigRV, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def run(self):

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