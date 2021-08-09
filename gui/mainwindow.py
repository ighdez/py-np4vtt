#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog

import controller
from gui.formclasses.ui_mainwindow import Ui_mainWindow
from gui.import_variables import ImportVariables
from model.data_format import VarDescriptives


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

    def connectSignals(self) -> None:
        self.ui.btnImportWizard.clicked.connect(self.importWizardClicked)

    def handleImportDone(self, varDescriptives: VarDescriptives) -> None:
        descText = str(varDescriptives)  # Maybe some rich text in here?
        self.ui.textDataInfo.setText(descText)

        self.setStateButtonsConfigure(True)

    # Slots
    def importWizardClicked(self) -> None:
        openDialog = QFileDialog()
        openDialog.setFileMode(QFileDialog.ExistingFile)
        openDialog.setNameFilter('Tab-separated values (*.txt);; Comma-separated values (*.csv)')

        openDialog.exec_()
        files = openDialog.selectedFiles()

        if files:
            columnNames = controller.openDataset(Path(files[0]))
            importDialog = ImportVariables(columnNames)
            if importDialog.exec_() == QDialog.Accepted:
                self.handleImportDone(importDialog.getVarDescriptives())

    def disableButtonsStep1(self) -> None:
        """Before loading data, all other buttons are disabled"""
        self.setStateButtonsOutput(False)
        self.setStateButtonsEstimate(False)
        self.setStateButtonsConfigure(False)
        self.ui.btnDescriptives.setEnabled(False)

    def setStateButtonsConfigure(self, newState: bool) -> None:
        """Buttons related to model configuration only should be enabled after data is loaded"""
        for btn in self.btnsConf:
            btn.setEnabled(newState)

    def setStateButtonsEstimate(self, newState: bool) -> None:
        """Buttons to perform estimation only should be enabled after a model is configured"""
        for btn in self.btnsEstimate:
            btn.setEnabled(newState)

    def setStateButtonsOutput(self, newState: bool) -> None:
        """Buttons to show estimation outputs, export, etc. only should be enabled after estimation is done"""
        for btn in self.btnsOutput:
            btn.setEnabled(newState)
