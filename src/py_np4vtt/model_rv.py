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
    mleScale: float
    mleVTT: float

    mleMaxIterations: int

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.mleScale > 0:
            errorList.append('Scale starting value must be positive.')

        if not self.mleMaxIterations > 0:
            errorList.append('Max iterations must be greater than zero.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList


@dataclass
class InitialArgsRV:
    BVTT: np.ndarray
    y_regress: np.ndarray
    x0: np.ndarray


class ModelRV:
    def __init__(self, cfg: ConfigRV, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def setupInitialArgs(self) -> Tuple[InitialArgsRV, float]:

        # Set vector of starting values of parameters to estimate
        x0 = np.array([self.cfg.mleScale, self.cfg.mleVTT])

        initialArgs = InitialArgsRV(
            BVTT=self.arrays.BVTT.flatten(),
            y_regress=self.arrays.Choice.flatten(),
            x0 = np.array([self.cfg.mleScale, self.cfg.mleVTT])
        )

        initialVal = -ModelRV.objectiveFunction(x0, initialArgs.BVTT, initialArgs.y_regress)

        # TODO: add an integrity check: initialVal should be finite. Otherwise, rise an error.

        return initialArgs, initialVal

    def run(self, args: InitialArgsRV):
        # Starting arguments and values for minimizer
        argTuple = (args.BVTT, args.y_regress)
        x0 = args.x0

        # Start minimization routine
        results = minimize(ModelRV.objectiveFunction, x0, args=argTuple, method='L-BFGS-B',options={'gtol': 1e-6})

        # Collect results
        x = results['x']
        hess = Hessian(ModelRV.objectiveFunction,method='forward')(x, args.BVTT, args.y_regress)
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        fval = -results['fun']
        exitflag = results['status']
        output = results['message']

        return x, se, fval, exitflag, output

    @staticmethod
    def objectiveFunction(x: np.ndarray, BVTT: np.ndarray, y_regress: np.ndarray):
        # Separate parameters: x is the estimated (multi-dimensional) parameter
        scale, VTT = x

        # Create value functions
        V1 = scale * BVTT
        V2 = scale * VTT

        # Create choice probability and Log-likelihood
        p = np.exp(V1) / (np.exp(V1) + np.exp(V2))
        ll = - np.sum(np.log(p * (y_regress == 0) + (1 - p) * (y_regress == 1)))

        # Return choice probability
        return ll