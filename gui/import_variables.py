#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List, Optional

from PyQt5.QtWidgets import QDialog, QLineEdit

from gui.formclasses.ui_import_variables import Ui_variablesDialog
from model.data_format import VarMapping


class ImportVariables(QDialog):
    def __init__(self, column_names: List[str], parent=None):
        super().__init__(parent)

        self.column_names = column_names
        self.varMapping: Optional[VarMapping] = None

        self.ui = Ui_variablesDialog()
        self.ui.setupUi(self)

        self.fillColumnNames()

        self.connectSignals()

    def connectSignals(self) -> None:
        self.ui.btnVarID.clicked.connect(lambda: self.setVarField(self.ui.fieldVarID))
        self.ui.btnVarChosenAlt.clicked.connect(lambda: self.setVarField(self.ui.fieldVarChosenAlt))

        self.ui.btnVarCost1.clicked.connect(lambda: self.setVarField(self.ui.fieldVarCost1))
        self.ui.btnVarTime1.clicked.connect(lambda: self.setVarField(self.ui.fieldVarTime1))
        self.ui.btnVarCost2.clicked.connect(lambda: self.setVarField(self.ui.fieldVarCost2))
        self.ui.btnVarTime2.clicked.connect(lambda: self.setVarField(self.ui.fieldVarTime2))

    def fillColumnNames(self) -> None:
        self.ui.listColumns.addItems(self.column_names)

    def getVarMapping(self) -> Optional[VarMapping]:
        return self.varMapping

    # Slots
    def setVarField(self, fieldWidget: QLineEdit) -> None:
        selectedItem = self.ui.listColumns.takeItem(self.ui.listColumns.currentRow())
        fieldWidget.setText(selectedItem.text())

    # override in QDialog
    def done(self, retCode: int) -> None:
        if retCode == QDialog.DialogCode.Accepted:
            # All fields must be filled for this dialog to actually be accepted
            nameVarId = self.ui.fieldVarID.text()
            nameVarChosenAlt = self.ui.fieldVarChosenAlt.text()
            nameVarCost1 = self.ui.fieldVarCost1.text()
            nameVarTime1 = self.ui.fieldVarTime1.text()
            nameVarCost2 = self.ui.fieldVarCost2.text()
            nameVarTime2 = self.ui.fieldVarTime2.text()

            if nameVarId and nameVarChosenAlt and nameVarCost1 and nameVarTime1 and nameVarCost2 and nameVarTime2:
                self.varMapping = VarMapping(
                    varId=nameVarId,
                    varChosenAlt=nameVarChosenAlt,
                    varCost1=nameVarCost1,
                    varTime1=nameVarTime1,
                    varCost2=nameVarCost2,
                    varTime2=nameVarTime2,
                )

                super().done(retCode)
            else:
                pass
        else:
            super().done(retCode)
