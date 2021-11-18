# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_modelconfig_locallogit.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_configLocLogitDialog(object):
    def setupUi(self, configLocLogitDialog):
        configLocLogitDialog.setObjectName("configLocLogitDialog")
        configLocLogitDialog.resize(318, 211)
        self.layoutToplevel = QtWidgets.QVBoxLayout(configLocLogitDialog)
        self.layoutToplevel.setObjectName("layoutToplevel")
        self.groupBox = QtWidgets.QGroupBox(configLocLogitDialog)
        self.groupBox.setObjectName("groupBox")
        self.layoutMain = QtWidgets.QFormLayout(self.groupBox)
        self.layoutMain.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.layoutMain.setObjectName("layoutMain")
        self.labelParamMinimum = QtWidgets.QLabel(self.groupBox)
        self.labelParamMinimum.setObjectName("labelParamMinimum")
        self.layoutMain.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelParamMinimum)
        self.labelParamMaximum = QtWidgets.QLabel(self.groupBox)
        self.labelParamMaximum.setObjectName("labelParamMaximum")
        self.layoutMain.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelParamMaximum)
        self.labelParamSupportPoints = QtWidgets.QLabel(self.groupBox)
        self.labelParamSupportPoints.setObjectName("labelParamSupportPoints")
        self.layoutMain.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelParamSupportPoints)
        self.fieldParamSupportPoints = QtWidgets.QSpinBox(self.groupBox)
        self.fieldParamSupportPoints.setMinimum(1)
        self.fieldParamSupportPoints.setObjectName("fieldParamSupportPoints")
        self.layoutMain.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.fieldParamSupportPoints)
        self.fieldParamMinimum = QtWidgets.QSpinBox(self.groupBox)
        self.fieldParamMinimum.setObjectName("fieldParamMinimum")
        self.layoutMain.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldParamMinimum)
        self.fieldParamMaximum = QtWidgets.QSpinBox(self.groupBox)
        self.fieldParamMaximum.setObjectName("fieldParamMaximum")
        self.layoutMain.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fieldParamMaximum)
        self.layoutToplevel.addWidget(self.groupBox)
        self.btnDialogBox = QtWidgets.QDialogButtonBox(configLocLogitDialog)
        self.btnDialogBox.setOrientation(QtCore.Qt.Horizontal)
        self.btnDialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.btnDialogBox.setCenterButtons(True)
        self.btnDialogBox.setObjectName("btnDialogBox")
        self.layoutToplevel.addWidget(self.btnDialogBox)

        self.retranslateUi(configLocLogitDialog)
        self.btnDialogBox.accepted.connect(configLocLogitDialog.accept)
        self.btnDialogBox.rejected.connect(configLocLogitDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(configLocLogitDialog)

    def retranslateUi(self, configLocLogitDialog):
        _translate = QtCore.QCoreApplication.translate
        configLocLogitDialog.setWindowTitle(_translate("configLocLogitDialog", "Loc. Logit - Configuration"))
        self.groupBox.setTitle(_translate("configLocLogitDialog", "Set the VTT support points"))
        self.labelParamMinimum.setText(_translate("configLocLogitDialog", "Minimum:"))
        self.labelParamMaximum.setText(_translate("configLocLogitDialog", "Maximum:"))
        self.labelParamSupportPoints.setText(_translate("configLocLogitDialog", "Number of support points:"))
