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
import warnings
from py_np4vtt.data_format import ModelArrays
from py_np4vtt.utils import vtt_midpoints, predicted_vtt

warnings.filterwarnings('ignore')

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

class ModelRouwendal:
    def __init__(self, cfg: ConfigRouwendal, arrays: ModelArrays):
        self.cfg = cfg
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.cfg.minimum, self.cfg.maximum, self.cfg.supportPoints)

        # Compute the midpoints of the VTT grid
        self.vtt_mid = vtt_midpoints(self.vtt_grid)

        # Compute distance between points at each support point
        dist = self.vtt_grid[1] - self.vtt_grid[0]

        # Print message of the support points
        print("Created a VTT grid of " + str(self.cfg.supportPoints) + \
            " points between " + str(self.cfg.minimum) + " and " + str(self.cfg.maximum) + ".")

        print("Distance between points of the VTT grid is " + str(dist))
        
    def run(self):

        # Set vector of starting values of xameters to estimate
        q0 = np.log(self.cfg.startQ/(1-self.cfg.startQ))
        x0 = np.hstack([q0, np.zeros(len(self.vtt_grid))])

        # Initial value of the log-likelihood function
        init_ll = -ModelRouwendal.objectiveFunction(x0, self.arrays.NP, self.arrays.T, self.arrays.BVTT,
                                                      self.arrays.Choice, self.vtt_grid)

        # TODO: add an integrity check: initialVal should be finite. Otherwise, rise an error.

        # Starting values
        argTuple = (self.arrays.NP, self.arrays.T, self.arrays.BVTT, self.arrays.Choice, self.vtt_grid)

        # Start optimization
        results = minimize(ModelRouwendal.objectiveFunction, x0, args=argTuple, method='L-BFGS-B',options={'gtol': 1e-6})

        # Collect results
        x = results['x']
        hess = Hessian(ModelRouwendal.objectiveFunction,method='forward')(x,self.arrays.NP, self.arrays.T, self.arrays.BVTT, self.arrays.Choice, self.vtt_grid)
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        ll = -results['fun']
        exitflag = results['status']

        # Get estimated probability of consistent choice
        q_prob = np.exp(x[0])/(1+np.exp(x[0]))
        q_est = x[0]
        q_se = np.sqrt((np.exp(q_est)/(1+np.exp(q_est)))**2)**2 * se[0]**2

        # Get estimated FVTT and xameters
        x = x[1:]
        se = se[1:]
        fvtt = np.exp(x)/np.sum(np.exp(x))
        p = np.cumsum(fvtt)

        # Compute the predicted VTT at the midpoints
        vtt = predicted_vtt(p,self.vtt_mid,self.arrays.NP)

        # Return output
        return q_est, q_se, q_prob, x, se, p, vtt, init_ll, ll, exitflag

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
