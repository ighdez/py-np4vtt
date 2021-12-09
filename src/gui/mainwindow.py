#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from pathlib import Path

from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog, QLabel

import controller
from gui.formclasses.ui_mainwindow import Ui_mainWindow
from gui.import_variables import ImportVariables
from gui.modelconfig import ModelConfigLocLogit, ModelConfigLogit, ModelConfigRouwendal, ModelConfigANN
from gui.progress import EstimationProgress, MethodType

from model.data_format import DescriptiveStatsBasic


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

        self.labelsStatus = [
            self.ui.labelLocalLogitStatus, self.ui.labelLogisticRegressionStatus,
            self.ui.labelRouwendalStatus, self.ui.labelANNStatus,
        ]

        self.disableTaskButtonsStep1()
        self.connectSignals()

    def connectSignals(self) -> None:
        self.ui.btnImportWizard.clicked.connect(self.importWizardClicked)

        self.ui.btnLocalLogitConf.clicked.connect(self.handleConfigLocLogit)
        self.ui.btnLogisticRegressionConf.clicked.connect(self.handleConfigLogit)
        self.ui.btnRouwendalConf.clicked.connect(self.handleConfigRouwendal)
        self.ui.btnANNConf.clicked.connect(self.handleConfigANN)

        self.ui.btnLocalLogitEstimate.clicked.connect(self.handleEstimateLocLogit)
        self.ui.btnLogisticRegressionEstimate.clicked.connect(self.handleEstimateLogit)
        self.ui.btnRouwendalEstimate.clicked.connect(self.handleEstimateLogit)
        self.ui.btnANNTrain.clicked.connect(self.handleEstimateANN)

    def handleImportDone(self, varDescriptives: DescriptiveStatsBasic) -> None:
        descText = str(varDescriptives)
        self.ui.textDataInfo.setText(descText)

        self.setTaskButtonsConfigure(True)
        self.setStatusLabelsConfigureAll()

    @staticmethod
    def setStatusLabel(label: QLabel, colorCSSName: str, noticeText: str) -> None:
        richText = (
            '<html><head></head><body><p>'
            '<span style="color:{color};">{notice}</span>'
            '</p></body></html>'
        ).format(color=colorCSSName, notice=noticeText)

        label.setTextFormat(Qt.TextFormat.RichText)
        label.setText(richText)

    def setStatusLabelsConfigureAll(self) -> None:
        for lab in self.labelsStatus:
            self.setStatusLabel(lab, 'blue', 'Configure Options')

    def setStatusLabelEstimate(self, label: QLabel) -> None:
        self.setStatusLabel(label, 'green', 'Ready to estimate')

    def disableTaskButtonsStep1(self) -> None:
        """Before loading data, all other buttons are disabled"""
        self.setTaskButtonsOutput(False)
        self.setTaskButtonsEstimate(False)
        self.setTaskButtonsConfigure(False)
        self.ui.btnDescriptives.setEnabled(False)

    def setTaskButtonsConfigure(self, newState: bool) -> None:
        """Buttons related to model configuration only should be enabled after data is loaded"""
        for btn in self.btnsConf:
            btn.setEnabled(newState)

    def setTaskButtonsEstimate(self, newState: bool) -> None:
        """Buttons to perform estimation only should be enabled after a model is configured"""
        for btn in self.btnsEstimate:
            btn.setEnabled(newState)

    def setTaskButtonsOutput(self, newState: bool) -> None:
        """Buttons to show estimation outputs, export, etc. only should be enabled after estimation is done"""
        for btn in self.btnsOutput:
            btn.setEnabled(newState)

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
            if importDialog.exec() == QDialog.Accepted:
                self.handleImportDone(importDialog.getVarDescriptives())

    def handleConfigLocLogit(self) -> None:
        dialog = ModelConfigLocLogit()
        dialog.exec()
        self.setStatusLabelEstimate(self.ui.labelLocalLogitStatus)
        self.ui.btnLocalLogitEstimate.setEnabled(True)

    def handleConfigLogit(self) -> None:
        dialog = ModelConfigLogit()
        dialog.exec()
        self.setStatusLabelEstimate(self.ui.labelLogisticRegressionStatus)
        self.ui.btnLogisticRegressionEstimate.setEnabled(True)

    def handleConfigRouwendal(self) -> None:
        dialog = ModelConfigRouwendal()
        dialog.exec()
        self.setStatusLabelEstimate(self.ui.labelRouwendalStatus)
        self.ui.btnRouwendalEstimate.setEnabled(True)

    def handleConfigANN(self) -> None:
        dialog = ModelConfigANN()
        dialog.exec()
        self.setStatusLabelEstimate(self.ui.labelANNStatus)
        self.ui.btnANNTrain.setEnabled(True)

    def handleEstimateLocLogit(self) -> None:
        dialog = EstimationProgress(MethodType.MethodLocLogit, controller.modelcfg_loclogit)
        dialog.exec()
        dialog.ui.textLog.setText(str(controller.modelcfg_loclogit))

    def handleEstimateLogit(self) -> None:
        dialog = EstimationProgress(MethodType.MethodLogit, controller.modelcfg_logit)
        dialog.exec()
        dialog.ui.textLog.setText(str(controller.modelcfg_logit))

    def handleEstimateRouwendal(self) -> None:
        dialog = EstimationProgress(MethodType.MethodRouwendal, controller.modelcfg_rouwendal)
        dialog.exec()
        dialog.ui.textLog.setText(str(controller.modelcfg_rouwendal))

    def handleEstimateANN(self) -> None:
        dialog = EstimationProgress(MethodType.MethodANN, controller.modelcfg_ann)
        dialog.exec()
        dialog.ui.textLog.setText(str(controller.modelcfg_ann))
