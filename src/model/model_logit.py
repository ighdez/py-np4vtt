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
import numpy as np

from model.data_format import ModelArrays


@dataclass
class ConfigLogit:
    mleIntercept: float
    mleParameter: float
    mleScale: float

    mleMaxIterations: int

    seed: Optional[int]

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.mleScale > 0:
            errorList.append('Scale starting value must be positive.')

        if not self.mleMaxIterations > 0:
            errorList.append('Max iterations must be greater than zero.')

        if not self.seed >= 0:
            errorList.append('Seed must be non-negative.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList


class ModelLogit:
    def __init__(self, cfg: ConfigLogit, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def run(self):
        # Use passed seed if desired
        if self.cfg.seed:
            np.random.seed(self.cfg.seed)

        # Prepare data
        i_obs_y = \
            np.tile(np.random.randint(1, self.arrays.T, size=(self.arrays.NP, 1)), (1, self.arrays.T)) \
            == np.tile(np.arange(1, self.arrays.T + 1), (self.arrays.NP, 1))
        i_obs_x = (i_obs_y == 0)

        BVTT = np.sum(i_obs_y * self.arrays.BVTT, axis=1)
        sumYBVTT = np.sum(i_obs_x * self.arrays.BVTT * self.arrays.Choice, axis=1)
        y_regress = np.sum(self.arrays.Choice * i_obs_y, axis=1)

        # Starting values and arguments for minimizer
        x0 = np.array([self.cfg.mleScale, self.cfg.mleIntercept, self.cfg.mleParameter])
        args = (sumYBVTT, BVTT, y_regress)

        # Start minimization routine
        results = minimize(ModelLogit.objectiveFunction, x0, args=args, method='Nelder-Mead')

        # Collect results
        x = results['x']
        fval = results['fun']
        exitflag = results['status']
        output = results['message']

        return x, fval, exitflag, output

    @staticmethod
    def objectiveFunction(x: Tuple[float, float, float], sumYBVTT: np.float64, BVTT: np.float64, y_regress: np.float64):
        # Separate parameters: x is the estimated (multi-dimensional) parameter
        scale, intercept, parameter = x

        # Create value functions
        VTT = intercept + parameter * sumYBVTT
        V1 = scale * BVTT
        V2 = scale * VTT

        # Create choice probability and Log-likelihood
        p = np.exp(V1) / (np.exp(V1) + np.exp(V2))
        ll = - np.sum(np.log(p * (y_regress == 0) + (1 - p) * (y_regress == 1)))

        # Return choice probability
        return ll
