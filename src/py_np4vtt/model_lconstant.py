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
from scipy.optimize import minimize
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

        # Compute the kernel width
        self.k = self.params.kernelWidth

        # Create the choice indicator
        self.YX = ~self.arrays.Choice.flatten()
        self.BVTT = self.arrays.BVTT.flatten()
        
    def run(self):
        
        ecdf = ModelLConstant.nadaraya_watson(self.vtt_grid,self.YX,self.BVTT,self.k)

        return ecdf, self.vtt_grid

    # Nadaraya-Watson estimator with gaussian kernel
    @staticmethod
    def nadaraya_watson(x,Y,X,h):

        # Compute the difference between x and X for each point in x
        xi_minus_X = (x[:,np.newaxis] - X)/h

        # Compute the numerator and denominator of g(x)
        g_num = np.sum(norm.pdf(xi_minus_X)*Y[np.newaxis,:],axis=1)
        g_den = np.sum(norm.pdf(xi_minus_X),axis=1)

        # Return g(x)
        return g_num/g_den