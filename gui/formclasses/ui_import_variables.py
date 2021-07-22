# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_import_variables.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_variablesDialog(object):
    def setupUi(self, variablesDialog):
        variablesDialog.setObjectName("variablesDialog")
        variablesDialog.resize(757, 442)
        self.verticalLayout = QtWidgets.QVBoxLayout(variablesDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.layoutMain = QtWidgets.QHBoxLayout()
        self.layoutMain.setObjectName("layoutMain")
        self.layoutColumns = QtWidgets.QVBoxLayout()
        self.layoutColumns.setObjectName("layoutColumns")
        self.labelColumns = QtWidgets.QLabel(variablesDialog)
        self.labelColumns.setObjectName("labelColumns")
        self.layoutColumns.addWidget(self.labelColumns)
        self.listColumns = QtWidgets.QListView(variablesDialog)
        self.listColumns.setObjectName("listColumns")
        self.layoutColumns.addWidget(self.listColumns)
        self.layoutMain.addLayout(self.layoutColumns)
        self.layoutRightPane = QtWidgets.QVBoxLayout()
        self.layoutRightPane.setObjectName("layoutRightPane")
        self.labelVarsDescription = QtWidgets.QLabel(variablesDialog)
        self.labelVarsDescription.setObjectName("labelVarsDescription")
        self.layoutRightPane.addWidget(self.labelVarsDescription)
        spacerItem = QtWidgets.QSpacerItem(20, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layoutRightPane.addItem(spacerItem)
        self.layoutVars = QtWidgets.QHBoxLayout()
        self.layoutVars.setObjectName("layoutVars")
        self.layoutVarsLeft = QtWidgets.QVBoxLayout()
        self.layoutVarsLeft.setObjectName("layoutVarsLeft")
        self.layoutVarID = QtWidgets.QFormLayout()
        self.layoutVarID.setObjectName("layoutVarID")
        self.fieldVarID = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarID.setReadOnly(True)
        self.fieldVarID.setObjectName("fieldVarID")
        self.layoutVarID.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldVarID)
        self.btnVarID = QtWidgets.QPushButton(variablesDialog)
        self.btnVarID.setObjectName("btnVarID")
        self.layoutVarID.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btnVarID)
        self.layoutVarsLeft.addLayout(self.layoutVarID)
        spacerItem1 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layoutVarsLeft.addItem(spacerItem1)
        self.layoutCostTime1 = QtWidgets.QFormLayout()
        self.layoutCostTime1.setObjectName("layoutCostTime1")
        self.btnVarCost1 = QtWidgets.QPushButton(variablesDialog)
        self.btnVarCost1.setObjectName("btnVarCost1")
        self.layoutCostTime1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btnVarCost1)
        self.fieldVarCost1 = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarCost1.setReadOnly(True)
        self.fieldVarCost1.setObjectName("fieldVarCost1")
        self.layoutCostTime1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldVarCost1)
        self.btnVarTime1 = QtWidgets.QPushButton(variablesDialog)
        self.btnVarTime1.setObjectName("btnVarTime1")
        self.layoutCostTime1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.btnVarTime1)
        self.fieldVarTime1 = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarTime1.setReadOnly(True)
        self.fieldVarTime1.setObjectName("fieldVarTime1")
        self.layoutCostTime1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fieldVarTime1)
        self.layoutVarsLeft.addLayout(self.layoutCostTime1)
        self.layoutVars.addLayout(self.layoutVarsLeft)
        spacerItem2 = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layoutVars.addItem(spacerItem2)
        self.layoutVarsRight = QtWidgets.QVBoxLayout()
        self.layoutVarsRight.setObjectName("layoutVarsRight")
        self.layoutVarChosenAlt = QtWidgets.QFormLayout()
        self.layoutVarChosenAlt.setObjectName("layoutVarChosenAlt")
        self.fieldVarChosenAlt = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarChosenAlt.setReadOnly(True)
        self.fieldVarChosenAlt.setObjectName("fieldVarChosenAlt")
        self.layoutVarChosenAlt.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldVarChosenAlt)
        self.btnVarChosenAlt = QtWidgets.QPushButton(variablesDialog)
        self.btnVarChosenAlt.setObjectName("btnVarChosenAlt")
        self.layoutVarChosenAlt.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btnVarChosenAlt)
        self.layoutVarsRight.addLayout(self.layoutVarChosenAlt)
        spacerItem3 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layoutVarsRight.addItem(spacerItem3)
        self.layoutCostTime2 = QtWidgets.QFormLayout()
        self.layoutCostTime2.setObjectName("layoutCostTime2")
        self.btnVarCost2 = QtWidgets.QPushButton(variablesDialog)
        self.btnVarCost2.setObjectName("btnVarCost2")
        self.layoutCostTime2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btnVarCost2)
        self.fieldVarCost2 = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarCost2.setReadOnly(True)
        self.fieldVarCost2.setObjectName("fieldVarCost2")
        self.layoutCostTime2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldVarCost2)
        self.btnVarTime2 = QtWidgets.QPushButton(variablesDialog)
        self.btnVarTime2.setObjectName("btnVarTime2")
        self.layoutCostTime2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.btnVarTime2)
        self.fieldVarTime2 = QtWidgets.QLineEdit(variablesDialog)
        self.fieldVarTime2.setReadOnly(True)
        self.fieldVarTime2.setObjectName("fieldVarTime2")
        self.layoutCostTime2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fieldVarTime2)
        self.layoutVarsRight.addLayout(self.layoutCostTime2)
        self.layoutVars.addLayout(self.layoutVarsRight)
        self.layoutRightPane.addLayout(self.layoutVars)
        self.layoutMain.addLayout(self.layoutRightPane)
        self.verticalLayout.addLayout(self.layoutMain)
        self.btnDialogBox = QtWidgets.QDialogButtonBox(variablesDialog)
        self.btnDialogBox.setOrientation(QtCore.Qt.Horizontal)
        self.btnDialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.btnDialogBox.setObjectName("btnDialogBox")
        self.verticalLayout.addWidget(self.btnDialogBox)

        self.retranslateUi(variablesDialog)
        self.btnDialogBox.accepted.connect(variablesDialog.accept)
        self.btnDialogBox.rejected.connect(variablesDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(variablesDialog)

    def retranslateUi(self, variablesDialog):
        _translate = QtCore.QCoreApplication.translate
        variablesDialog.setWindowTitle(_translate("variablesDialog", "Mapping study variables to data columns"))
        self.labelColumns.setText(_translate("variablesDialog", "<html><head/><body><p><b>Data columns</b></p></body></html>"))
        self.labelVarsDescription.setText(_translate("variablesDialog", "<html>\n"
"<head/>\n"
"<body>\n"
"<p>From the column names in the left panel, choose which to use for each variable:</p>\n"
"<dl>\n"
"<dt><b>ID:</b></dt>\n"
"<dd>Unique numeric identifier for each decision maker</dd>\n"
"<dt><b>Chosen Alt.:</b></dt>\n"
"<dd>Chosen alternative among those presented (must be either \'1\' or \'2\')</dd>\n"
"<dt><b>Cost Alt. 1:</b></dt>\n"
"<dd>Cost variable of alternative 1</dd>\n"
"<dt><b>Time Alt. 1:</b></dt>\n"
"<dd>Time variable of alternative 1</dd>\n"
"<dt><b>Cost Alt. 2:</b></dt>\n"
"<dd>Cost variable of alternative 2</dd>\n"
"<dt><b>Time Alt. 2:</b></dt>\n"
"<dd>Time variable of alternative 2</dd>\n"
"</dl>\n"
"</body>\n"
"</html>"))
        self.fieldVarID.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))
        self.btnVarID.setText(_translate("variablesDialog", "ID →"))
        self.btnVarCost1.setText(_translate("variablesDialog", "Cost Alt. 1 →"))
        self.fieldVarCost1.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))
        self.btnVarTime1.setText(_translate("variablesDialog", "Time Alt. 1 →"))
        self.fieldVarTime1.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))
        self.fieldVarChosenAlt.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))
        self.btnVarChosenAlt.setText(_translate("variablesDialog", "Chosen Alt. →"))
        self.btnVarCost2.setText(_translate("variablesDialog", "Cost Alt. 2 →"))
        self.fieldVarCost2.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))
        self.btnVarTime2.setText(_translate("variablesDialog", "Time Alt. 2→"))
        self.fieldVarTime2.setPlaceholderText(_translate("variablesDialog", "<not chosen yet>"))