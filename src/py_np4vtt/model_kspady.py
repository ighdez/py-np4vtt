#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from py_np4vtt.data_format import ModelArrays

@dataclass
class ConfigKSpady:
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

class ModelKSpady:
    def __init__(self, params: ConfigKSpady, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

        # Create grid of support points
        self.vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the kernel width
        self.k = np.diff(self.vtt_grid).mean()

        # Create the choice indicator
        self.YX = ~self.arrays.Choice.flatten()
        self.BVTT = self.arrays.BVTT.flatten()
        
    def run(self):

        # Run the Klein-Spady estimator
        krreg = KernelReg(self.YX, self.BVTT, var_type='u', reg_type='lc', ukertype= 'gaussian', bw=[self.k])
        ecdf = krreg.fit(self.vtt_grid)[0]
        return ecdf, self.vtt_grid