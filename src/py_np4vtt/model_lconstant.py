#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Modules to configure and estimate a Local constant model."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from py_np4vtt.data_format import ModelArrays
from py_np4vtt.utils import predicted_vtt, vtt_midpoints

@dataclass
class ConfigLConstant:
    """Configuration class of the local constant model.
    
    This class stores the configuration parameters of a local constant model 
    and performs integrity checks before being passed to the model object.
    
    Parameters
    ----------
    
    minimum : float
        Minimum value of the VTT grid
    maximum : float
        Maximum value of the VTT grid
    supportPoints : int
        Number of support points of the VTT grid. The VTT grid will contain 
        `(supportPoints-1)` intervals. Must be greater than zero
    kernelWidth : float
        Kernel width for the Nadaraya-Watson estimator. Must be greater than 
        zero.
    """
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
    """Local constant model.
    
    This is the model class that prepares the data and estimates a local
    constant model [1,2].
    
    Parameters
    -----------
    params : ConfigLConstant
        A configuration class of a local constant model.
    arrays : ModelArrays
        Model arrays created with `make_modelarrays`
        
    Attributes
    ----------
    vtt_grid : numpy.ndarray
        The VTT grid created with the specifications of `ConfigLConstant`
    vtt_mid : numpy.ndarray
        The mid points of the VTT grid.

    Methods
    -------
    run():
        Estimates the local constant model.
    
    References
    ----------
    [1] Fosgerau, Mogens. "Investigating the distribution of the value of 
    travel time savings." Transportation Research Part B: Methodological 
    40.8 (2006): 688-707.
    [2] Fosgerau, Mogens. "Using nonparametrics to specify a model to measure 
    the value of travel time." Transportation Research Part A: Policy and 
    Practice 41.9 (2007): 842-856.
    """
    def __init__(self, params: ConfigLConstant, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the midpoints of the VTT grid
        self.vtt_mid = vtt_midpoints(self.vtt_grid)

        # Compute distance between points at each support point
        dist = self.vtt_grid[1] - self.vtt_grid[0]

        # Print message of the support points
        print("Created a VTT grid of " + str(self.params.supportPoints) + \
            " points between " + str(self.params.minimum) + " and " + str(self.params.maximum) + ".")

        print("Distance between points of the VTT grid is " + str(dist))

    def run(self):
        """Estimates the local constant model.
        
        Parameters
        ----------
        None.

        Returns
        -------
        p : numpy.ndarray
            The estimates of the cumulative choice probability (CDF), evaluated 
            at each mid point of the VTT grid. The first point is always zero 
            while the last point is always equal to the second last for 
            compatibility with plots.
        vtt : numpy.ndarray
            The estimated VTT per respondent, based in the estimated 
            CDF points (`p`) and the sample.
        """
        p = ModelLConstant.nadaraya_watson(self.vtt_mid[1:-1],~self.arrays.Choice.flatten(),self.arrays.BVTT.flatten(),self.params.kernelWidth)

        # Create counts per point of the VTT mid points
        vtt = predicted_vtt(p,self.vtt_grid,self.arrays.NP)

        # Add point 0 in the estimated CDF and repeat last point to make coincide with point zero and last point in the VTT mid point
        p = np.concatenate((0,p,p[-1]),axis=None)

        return p, vtt

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