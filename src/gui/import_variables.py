#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List, Optional

from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QDialog, QLineEdit

from gui.formclasses.ui_import_variables import Ui_variablesDialog

import controller
from model.data_format import VarMapping, VarDescriptives


class ImportVariables(QDialog):
    def __init__(self, column_names: List[str], parent=None):
        super().__init__(parent)

        self.column_names = column_names
        self.varMapping: Optional[VarMapping] = None
        self.varDescriptives: Optional[VarDescriptives] = None

        self.ui = Ui_variablesDialog()
        self.ui.setupUi(self)

        self.fillColumnNames()
        self.setNoticeInitial()

        self.connectSignals()

    def connectSignals(self) -> None:
        self.ui.btnVarID.clicked.connect(lambda: self.setVarField(self.ui.fieldVarID))
        self.ui.btnVarChosenAlt.clicked.connect(lambda: self.setVarField(self.ui.fieldVarChosenAlt))

        self.ui.btnVarCost1.clicked.connect(lambda: self.setVarField(self.ui.fieldVarCost1))
        self.ui.btnVarTime1.clicked.connect(lambda: self.setVarField(self.ui.fieldVarTime1))
        self.ui.btnVarCost2.clicked.connect(lambda: self.setVarField(self.ui.fieldVarCost2))
        self.ui.btnVarTime2.clicked.connect(lambda: self.setVarField(self.ui.fieldVarTime2))

    def setNoticeWithColor(self, colorCSSName: str, noticeText: str) -> None:
        richText = (
            '<html><head></head><body>'
            '<p><span style="font-size:12pt;">'
            'Notice: <span style="color:{color};">{notice}</span>'
            '</span></p>'
            '</body></html>'
        ).format(color=colorCSSName, notice=noticeText)

        self.ui.labelNotice.setTextFormat(Qt.TextFormat.RichText)
        self.ui.labelNotice.setText(richText)

    def setNoticeInitial(self) -> None:
        self.setNoticeWithColor('green', 'To proceed, map each study variable to a column in the dataset')

    def setNoticeIncompleteMapping(self) -> None:
        self.setNoticeWithColor('red', 'Some study variables are not yet mapped to a dataset column')

    def getVarDescriptives(self) -> VarDescriptives:
        return self.varDescriptives

    def fillColumnNames(self) -> None:
        self.ui.listColumns.addItems(self.column_names)

    # Slots
    def setVarField(self, fieldWidget: QLineEdit) -> None:
        selectedItem = self.ui.listColumns.takeItem(self.ui.listColumns.currentRow())
        fieldWidget.setText(selectedItem.text())

    # override in QDialog
    def done(self, retCode: int) -> None:
        if retCode == QDialog.DialogCode.Accepted:
            nameVarId = self.ui.fieldVarID.text()
            nameVarChosenAlt = self.ui.fieldVarChosenAlt.text()
            nameVarCost1 = self.ui.fieldVarCost1.text()
            nameVarTime1 = self.ui.fieldVarTime1.text()
            nameVarCost2 = self.ui.fieldVarCost2.text()
            nameVarTime2 = self.ui.fieldVarTime2.text()

            # All fields must be filled for this dialog to actually be accepted
            if nameVarId and nameVarChosenAlt and nameVarCost1 and nameVarTime1 and nameVarCost2 and nameVarTime2:
                self.varMapping = VarMapping(
                    varId=nameVarId,
                    varChosenAlt=nameVarChosenAlt,
                    varCost1=nameVarCost1,
                    varTime1=nameVarTime1,
                    varCost2=nameVarCost2,
                    varTime2=nameVarTime2,
                )

                self.varDescriptives = controller.importMappedDataset(self.varMapping)

                super().done(retCode)
            # Otherwise just ask the user to do a complete mapping
            else:
                self.setNoticeIncompleteMapping()
        else:
            super().done(retCode)
