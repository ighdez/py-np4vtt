#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from py_np4vtt.data_format import ModelArrays


@dataclass
class ConfigLocLogit:
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
    def __init__(self, params: ConfigLocLogit, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

    def run(self):
        # Create grid of support points
        vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the kernel width
        k = np.r_[vtt_grid, 0.] - np.r_[0., vtt_grid]
        k = k[:-2]
        k[0] = k[1].copy()

        # Flatten choice
        YX = self.arrays.Choice.T.flatten()

        # Perform a weighted logit for each support point
        p = []
        fval = 0.
        for n in range(len(vtt_grid)-1):
            x, fval_x = ModelLocLogit.initLocalLogit(n, k[n], self.arrays.BVTT, YX, vtt_grid)
            p.append(x[0])
            fval = fval + fval_x
        
        # Return probability array and -ll
        p = np.array(p)
        fval = -fval
        
        return p, fval, vtt_grid

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
        results = minimize(ModelLocLogit.objectiveFunction, coef_start, args=args, method='Nelder-Mead')

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
