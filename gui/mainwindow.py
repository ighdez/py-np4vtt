#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from gui.formclasses.ui_mainwindow import Ui_mainWindow
from gui.import_variables import ImportVariables
import controller


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        self.btnsConf = [
            self.ui.btnLocalLogitConf, self.ui.btnLogisticRegressionConf,
            self.ui.btnRouwendalConf, self.ui.btnANNConf,
        ]

        self.btnsEstimate = [
            self.ui.btnLocalLogitEstimate, self.ui.btnLogisticRegressionEstimate,
            self.ui.btnRouwendalEstimate, self.ui.btnANNTrain,
        ]

        self.btnsOutput = [
            self.ui.btnLocalLogitOutput, self.ui.btnLogisticRegressionOutput,
            self.ui.btnRouwendalOutput, self.ui.btnANNOutput,
            self.ui.btnPlot, self.ui.btnResultsExport,
        ]

        self.disableButtonsStep1()

        self.connectSignals()

    def connectSignals(self):
        self.ui.btnImportWizard.clicked.connect(self.importWizardClicked)

    # Slots
    @staticmethod
    def importWizardClicked():
        openDialog = QFileDialog()
        openDialog.setFileMode(QFileDialog.ExistingFile)
        openDialog.setNameFilter('Tab-separated values (*.txt);; Comma-separated values (*.csv)')

        openDialog.exec_()
        files = openDialog.selectedFiles()

        if files:
            column_names = controller.openDataset(Path(files[0]))
            import_dialog = ImportVariables(column_names)
            import_dialog.exec_()

    def disableButtonsStep1(self):
        """Before loading data, all other buttons are disabled"""
        self.disableButtonsOutput()
        self.disableButtonsEstimate()
        self.disableButtonsConfigure()
        self.ui.btnDescriptives.setEnabled(False)

    def disableButtonsConfigure(self):
        """Buttons related to model configuration only should be enabled after data is loaded"""
        for btn in self.btnsConf:
            btn.setEnabled(False)

    def disableButtonsEstimate(self):
        """Buttons to perform estimation only should be enabled after a model is configured"""
        for btn in self.btnsEstimate:
            btn.setEnabled(False)

    def disableButtonsOutput(self):
        """Buttons to show estimation outputs, export, etc. only should be enabled after estimation is done"""
        for btn in self.btnsOutput:
            btn.setEnabled(False)
