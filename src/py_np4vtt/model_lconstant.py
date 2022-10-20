#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from py_np4vtt.data_format import ModelArrays
from py_np4vtt.utils import predicted_vtt

@dataclass
class ConfigLConstant:
    minimum: float
    maximum: float
    supportPoints: int
    kernelWidth: float

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.maximum > self.minimum:
            errorList.append('Max must be greater than minimum.')

        if not self.supportPoints > 0:
            errorList.append('No. of support points must be greater than zero.')

        if not self.kernelWidth > 0:
            errorList.append('Kernel width must be greater than zero.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelLConstant:
    def __init__(self, params: ConfigLConstant, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute distance between points at each support point
        dist = self.vtt_grid[1] - self.vtt_grid[0]

        # Print message of the support points
        print("Created a VTT grid of " + str(self.params.supportPoints) + \
            " points between " + str(self.params.minimum) + " and " + str(self.params.maximum) + ".")

        print("Distance between points of the VTT grid is " + str(dist))

    def run(self):
        
        mean_f = ModelLConstant.nadaraya_watson(self.vtt_grid,~self.arrays.Choice.flatten(),self.arrays.BVTT.flatten(),self.params.kernelWidth)

        # Create counts per point of the VTT grid
        vtt = predicted_vtt(mean_f,self.vtt_grid,self.arrays.NP)

        return mean_f, vtt

    # Nadaraya-Watson estimator with gaussian kernel
    @staticmethod
    def nadaraya_watson(x,Y,X,h):
        
        g = np.empty(shape=x.shape)

        for i in range(g.shape[0]):
            xi_minus_X = (x[i] - X[X<=x[i]])/h
            g_num = np.sum(norm.pdf(xi_minus_X)*Y[X<=x[i]])
            g_den = np.sum(norm.pdf(xi_minus_X))

            g[i] = g_num/g_den

        return g

        # # Compute the difference between x and X for each point in x
        # xi_minus_X = (x[:,np.newaxis] - X)/h

        # # Compute the numerator and denominator of g(x)
        # g_num = np.sum(norm.pdf(xi_minus_X)*Y[np.newaxis,:],axis=1)
        # g_den = np.sum(norm.pdf(xi_minus_X),axis=1)

        # # Return g(x)
        # return g_num/g_den