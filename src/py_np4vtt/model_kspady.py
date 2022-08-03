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

@dataclass
class InitialArgsKSpady:
    vtt_grid: np.ndarray
    BVTT: np.ndarray
    k: np.ndarray
    YX: np.ndarray

class ModelKSpady:
    def __init__(self, params: ConfigKSpady, arrays: ModelArrays):
        self.params = params
        self.arrays = arrays

    def setupInitialArgs(self) -> InitialArgsKSpady:

        # Create grid of support points
        vtt_grid = np.linspace(self.params.minimum, self.params.maximum, self.params.supportPoints)

        # Compute the kernel width
        k = np.diff(vtt_grid).mean()

        initialArgs = InitialArgsKSpady(
            vtt_grid = vtt_grid,
            k = k,
            YX = ~self.arrays.Choice.flatten(),
            BVTT = self.arrays.BVTT.flatten()
        )

        return initialArgs

    def run(self, args: InitialArgsKSpady):

        # Run the Klein-Spady estimator
        krreg = KernelReg(args.YX, args.BVTT, var_type='u', reg_type='lc', ukertype= 'gaussian', bw=[args.k])
        ecdf = krreg.fit(args.vtt_grid)[0]
        return ecdf, args.vtt_grid