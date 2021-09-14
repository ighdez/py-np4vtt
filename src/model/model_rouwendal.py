#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass


@dataclass
class ConfigRouwendal:
    minimum: float
    maximum: float
    supportPoints: int

    def validate(self):
        
        # Create errormessage list
        errorMessage = []

        # Max must be greater than minimum
        isMaxGreaterThanMin = (self.maximum > self.minimum)

        if not isMaxGreaterThanMin:
            errorMessage.append('Max must be greater than minimum.')

        # No. of support points must be greater than zero
        isPositiveSuppPoints = (self.supportPoints > 0)

        if not isPositiveSuppPoints:
            errorMessage.append('No. of support points must be greater than zero.')

        # Create integrity check list
        integrityCheckList = [isMaxGreaterThanMin,isPositiveSuppPoints]

        # Test if all statements are true
        integrityCheck = all(integrityCheckList)

        # Return True if all OK, otherwise return False and a message.
        return(integrityCheck, errorMessage)
