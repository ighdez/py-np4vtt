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


class ModelConfig:
    pass


class ModelConfigLocLogit(QDialog, ModelConfig):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_configLocLogitDialog()
        self.ui.setupUi(self)


class ModelConfigLogit(QDialog, ModelConfig):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_configLogitDialog()
        self.ui.setupUi(self)


class ModelConfigRouwendal(QDialog, ModelConfig):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_configRouwendalDialog()
        self.ui.setupUi(self)


class ModelConfigANN(QDialog, ModelConfig):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_configANNDialog()
        self.ui.setupUi(self)
