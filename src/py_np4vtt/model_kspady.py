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
class ConfigKSpady:
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

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList

class ModelKSpady:
    def __init__(self, params: ConfigKSpady, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the kernel width
        self.k = self.params.kernelWidth#np.diff(self.vtt_grid).mean()

        # Create the choice indicator
        self.YX = ~self.arrays.Choice.flatten()
        self.BVTT = self.arrays.BVTT.flatten()
        
    def run(self):
        
        ecdf = ModelKSpady.nadaraya_watson(self.vtt_grid,self.YX,self.BVTT,self.k)
        # res = []
        # startv = 0.

        # # Run the Klein-Spady estimator
        # res = minimize(ModelKSpady.klein_spady_ll,startv,args = (self.YX,self.BVTT,self.k),method='L-BFGS-B',options={'gtol': 1e-6})

        # coef = res['x'].flatten()
        # ecdf = ModelKSpady.nw_pred(coef,self.vtt_grid,self.YX,self.BVTT,self.k)

        return ecdf, self.vtt_grid

    # # Klein-Spady log-likelihood
    # @staticmethod
    # def klein_spady_ll(coef,Y,X,h):
        
    #     gamma = coef
        
    #     g = ModelKSpady.loo_nw(gamma,Y, X, h)

    #     ll = np.log((Y==1)*g + (Y==0)*(1-g))

    #     return -np.sum(ll)

    # # leave-one-out NW
    # @staticmethod
    # def loo_nw(x,Y,X,h):
        
    #     g = np.empty(shape=X.shape)
    #     for i in range(X.shape[0]):
    #         X_i = X[i]
    #         X_minus_i = np.delete(X,i)
    #         Y_minus_i = np.delete(Y,i)

    #         # Compute the difference between x and X for each point in x
    #         xi_minus_X = ((X_i - X_minus_i)*x)/h

    #         # Compute the numerator and denominator of g(x)
    #         g_num = np.sum(norm.pdf(xi_minus_X)*Y_minus_i)
    #         g_den = np.sum(norm.pdf(xi_minus_X))

    #         g[i] = g_num/g_den

    #     # Return g(x)
    #     return g

    # # leave-one-out NW
    # @staticmethod
    # def nw_pred(gamma,x,Y,X,h):
        
    #     # Compute the difference between x and X for each point in x
    #     xi_minus_X = ((x[:,np.newaxis] - X)*gamma)/h

    #     # Compute the numerator and denominator of g(x)
    #     g_num = np.sum(norm.pdf(xi_minus_X)*Y[np.newaxis,:],axis=1)
    #     g_den = np.sum(norm.pdf(xi_minus_X),axis=1)

    #     # Return g(x)
    #     return g_num/g_den

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