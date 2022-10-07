#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from numdifftools import Hessian
import numpy as np

from py_np4vtt.data_format import ModelArrays


@dataclass
class ConfigRouwendal:
    minimum: float
    maximum: float
    supportPoints: int
    startQ: float

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.maximum > self.minimum:
            errorList.append('Max must be greater than minimum.')

        if not self.supportPoints > 0:
            errorList.append('No. of support points must be greater than zero.')

        if not (0 < self.startQ < 1):
            errorList.append('Probability of consistent choice must be in the interval (0,1).')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

@dataclass
class InitialArgsRouwendal:
    NP: int
    T: int
    BVTT: np.ndarray
    Choice: np.ndarray
    vtt_grid: np.ndarray


class ModelRouwendal:
    def __init__(self, cfg: ConfigRouwendal, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

    def setupInitialArgs(self) -> Tuple[InitialArgsRouwendal, float]:
        # Create grid of support points
        vtt_grid = np.linspace(self.cfg.minimum, self.cfg.maximum, self.cfg.supportPoints)

        # Set vector of starting values of xameters to estimate
        q0 = np.log(self.cfg.startQ/(1-self.cfg.startQ))
        x0 = np.hstack([q0, np.zeros(len(vtt_grid))])

        initialArgs = InitialArgsRouwendal(
            NP=self.arrays.NP,
            T=self.arrays.T,
            BVTT=self.arrays.BVTT,
            Choice=self.arrays.Choice,
            vtt_grid=vtt_grid)

        initialVal = -ModelRouwendal.objectiveFunction(x0, initialArgs.NP, initialArgs.T, initialArgs.BVTT,
                                                      initialArgs.Choice, initialArgs.vtt_grid)

        # TODO: add an integrity check: initialVal should be finite. Otherwise, rise an error.

        return initialArgs, initialVal

    def run(self, args: InitialArgsRouwendal):

        # Starting values
        q0 = np.log(self.cfg.startQ/(1-self.cfg.startQ))
        x0 = np.hstack([q0, np.zeros(len(args.vtt_grid))])
        argTuple = (args.NP, args.T, args.BVTT, args.Choice, args.vtt_grid)

        # Start optimization
        results = minimize(ModelRouwendal.objectiveFunction, x0, args=argTuple, method='L-BFGS-B',options={'gtol': 1e-6})

        # Collect results
        x = results['x']
        hess = Hessian(ModelRouwendal.objectiveFunction,method='forward')(x,args.NP, args.T, args.BVTT, args.Choice, args.vtt_grid)
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        fval = -results['fun']
        exitflag = results['status']

        # Compute standard errors
        # TODO: standard errors

        # Get estimated probability of consistent choice
        q_prob = np.exp(x[0])/(1+np.exp(x[0]))
        q = x[0]
        q_se = np.sqrt((np.exp(q)/(1+np.exp(q)))**2)**2 * se[0]**2

        # Get estimated FVTT and xameters
        x = x[1:]
        se = se[1:]
        fvtt = np.exp(x)/np.sum(np.exp(x))
        ecdf = np.cumsum(fvtt)

        # Return output
        return q, q_se, q_prob, x, se, ecdf, args.vtt_grid, fval, exitflag

    @staticmethod
    def objectiveFunction(x, NP, T, BVTT, Choice, vtt_grid):
        
        # Re-scale Q and FVTT to fit between zero and one
        q = np.exp(x[0]) / (1 + np.exp(x[0]))
        fvtt = np.exp(x[1:]) / np.sum(np.exp(x[1:]))

        # Conditional probability
        P = np.zeros((NP, len(vtt_grid)))

        for n in range(len(vtt_grid)):
            tau = np.sum(((vtt_grid[n] > BVTT) == Choice).astype(int), axis=1)
            P[:, n] = (q**tau) * ((1-q) ** (T-tau))
            
        # Maximise log-likelihood. L is computed by multiplying conditional P
        # with density fvtt, average and sum across all obs
        L = -np.sum(np.log(np.sum(fvtt*P, axis=1)))

        return L
