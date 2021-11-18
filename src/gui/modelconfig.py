#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from PyQt5.QtWidgets import QDialog

from gui.formclasses.ui_modelconfig_ann import Ui_configANNDialog
from gui.formclasses.ui_modelconfig_locallogit import Ui_configLocLogitDialog
from gui.formclasses.ui_modelconfig_logit import Ui_configLogitDialog
from gui.formclasses.ui_modelconfig_rouwendal import Ui_configRouwendalDialog

import controller


class ModelConfigLocLogit(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_configLocLogitDialog()
        self.ui.setupUi(self)

    def accept(self) -> None:
        # TODO: Validators for the float fields with Regex
        minimum = float(self.ui.fieldParamMinimum.text())
        maximum = float(self.ui.fieldParamMaximum.text())
        numPoints = int(self.ui.fieldParamSupportPoints.text())

        controller.modelConfig_loclogit(minimum, maximum, numPoints)
        super().accept()


class ModelConfigLogit(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_configLogitDialog()
        self.ui.setupUi(self)

    def accept(self):
        # TODO: Validators for the fields with Regex
        intercept = float(self.ui.fieldValueIntercept.text())
        parameter = float(self.ui.fieldValueParameter.text())
        scale = float(self.ui.fieldValueScale.text())

        iterations = int(self.ui.fieldOptionLimit.text())

        seed = None
        if self.ui.groupBoxOptionSeed.isChecked():
            seed = int(self.ui.fieldOptionSeed.text())

        controller.modelConfig_logit(intercept, parameter, scale, iterations, seed)
        super().accept()


class ModelConfigRouwendal(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_configRouwendalDialog()
        self.ui.setupUi(self)

    def accept(self) -> None:
        minimum = float(self.ui.fieldPointsMinimum.text())
        maximum = float(self.ui.fieldPointsMaximum.text())
        numPoints = int(self.ui.fieldPointsSupport.text())

        probConsistent = float(self.ui.sliderProbConsistent.value()) / 100
        maxIterations = int(self.ui.fieldMaxIterations.text())

        controller.modelConfig_rouwendal(minimum, maximum, numPoints, probConsistent, maxIterations)
        super().accept()


class ModelConfigANN(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_configANNDialog()
        self.ui.setupUi(self)

    def accept(self) -> None:
        # TODO: get the table rows as list
        hiddenLayers = None

        numRepeats = int(self.ui.fieldOptionRepeats.value())
        numShuffles = int(self.ui.fieldOptionShuffles.value())

        seed = None
        if self.ui.groupBoxSeed.isChecked():
            seed = int(self.ui.fieldSeed.text())

        controller.modelConfig_ann(hiddenLayers, numRepeats, numShuffles, seed)
        super().accept()
