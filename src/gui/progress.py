#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from time import sleep
from typing import Union
from enum import Enum, auto, unique

from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog

from gui.formclasses.ui_progress import Ui_progressDialog

from model.model_loclogit import ConfigLocLogit
from model.model_logit import ConfigLogit
from model.model_rouwendal import ConfigRouwendal
from model.model_ann import ConfigANN


@unique
class MethodType(Enum):
    MethodLocLogit = auto()
    MethodLogit = auto()
    MethodRouwendal = auto()
    MethodANN = auto()


MethodConfig = Union[ConfigLocLogit, ConfigLogit, ConfigRouwendal, ConfigANN]


class EstimationWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        for i in range(5):
            sleep(1.0)
            self.progress.emit(i + 1)
        self.finished.emit()


class EstimationProgress(QDialog):
    def __init__(self, methodType: MethodType, methodCfg: MethodConfig, parent=None):
        super().__init__(parent)

        self.methodType = methodType
        self.methodCfg = methodCfg

        self.ui = Ui_progressDialog()
        self.ui.setupUi(self)

    def setStatusWithColor(self, colorCSSName: str, statusMsg: str) -> None:
        richText = (
            '<html><head></head><body>'
            '<p><span style="color:{color};">{message}</span></p>'
            '</body></html>'
        ).format(color=colorCSSName, message=statusMsg)

        self.ui.labelStatusBody.setTextFormat(Qt.TextFormat.RichText)
        self.ui.labelStatusBody.setText(richText)

    def setStatusOptimizing(self) -> None:
        self.setStatusWithColor('blue', 'Optimizing')

    def reportStatus(self, progressValue: int) -> None:
        print(f"EstimationProgress.reportStatus({progressValue})")
        self.ui.textLog.append(f"EstimationProgress.reportStatus({progressValue})")
