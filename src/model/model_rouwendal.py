#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass

from model.data_format import ModelArrays

import numpy as np

from scipy.optimize import minimize

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

        if not (self.startQ > 0 and self.startQ < 1):
            errorList.append('Probability of consistent choice must be in the interval (0,1).')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelRouwendal:
    def __init__(self, params: ConfigRouwendal, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

    def run(self):

        # Create grid of support points
        vtt_grid = np.linspace(self.params.minimum,self.params.maximum,self.params.supportPoints)

        # Starting values
        q0 = np.log(self.params.startQ/(1-self.params.startQ))
        x0 = np.hstack([q0,np.zeros(len(vtt_grid))])
        args = (self.arrays.NP,self.arrays.T,self.arrays.BVTT,self.arrays.Choice,vtt_grid)

        # Start optimization
        results = minimize(ModelRouwendal.objectiveFunction,x0,args=args,method='BFGS')

        # Collect results
        x = results['x']
        fval = results['fun']
        exitflag = results['status']
        output = results['message']

        # Compute standard errors
        # (not implemented yet)

        # Get estimated probability of consistent choice
        q_prob = np.exp(x[0])/(1+np.exp(x[0]))
        q_est = x[0]

        # Get estimated FVTT and parameters
        par = x[1:]
        fvtt = np.exp(par)/np.sum(np.exp(par))
        cumsum_fvtt = np.cumsum(fvtt)

        # Return output
        return q_prob, q_est, par, fvtt, cumsum_fvtt, vtt_grid, fval, exitflag, output

    @staticmethod
    def objectiveFunction(x,NP,T,BVTT,Choice,vtt_grid):
        
        # Re-scale Q and FVTT to fit between zero and one
        q = np.exp(x[0])/(1+np.exp(x[0]))
        fvtt = np.exp(x[1:])/np.sum(np.exp(x[1:]))

        # Conditional probability
        P = np.zeros((NP,len(vtt_grid)))

        for n in range(len(vtt_grid)):
            tau = np.sum(((vtt_grid[n] > BVTT)==Choice).astype(int),axis=1)
            P[:,n] = (q**tau)*((1-q)**(T-tau))
            
        # Maximise log-likelihood. L is computed by multiplying conditional P
        # with density fvtt, average and sum across all obs
        L = -np.sum(np.log(np.sum(fvtt*P,axis=1)))

        return L