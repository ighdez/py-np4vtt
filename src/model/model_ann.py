#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import List, Optional

from model.data_format import ModelArrays


@dataclass
class ConfigANN:
    hiddenLayerNodes: List[int]

    trainingRepeats: int
    shufflesPerRepeat: int

    seed: Optional[int]

    def validate(self):
        # Create errormessage list
        errorList = []

        if not self.trainingRepeats > 0:
            errorList.append('Number of repeats must be positive.')

        if not self.shufflesPerRepeat > 0:
            errorList.append('Number of shuffles per repeats must be positive.')

        # Whoever calls this validator knows that empty errorList means validator success
        return errorList


@dataclass
class InitialArgsANN:
    pass


class ModelANN:
    def __init__(self, cfg: ConfigANN, arrays: ModelArrays):
        pass

    def setupInitialArgs(self) -> InitialArgsANN:
        pass

    def run(self) -> None:
        pass
