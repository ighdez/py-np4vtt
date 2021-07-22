# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_descriptives.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_descriptivesDialog(object):
    def setupUi(self, descriptivesDialog):
        descriptivesDialog.setObjectName("descriptivesDialog")
        descriptivesDialog.resize(439, 278)
        self.layoutToplevel = QtWidgets.QVBoxLayout(descriptivesDialog)
        self.layoutToplevel.setObjectName("layoutToplevel")
        self.groupBoxStep1 = QtWidgets.QGroupBox(descriptivesDialog)
        self.groupBoxStep1.setObjectName("groupBoxStep1")
        self.layoutGrpStep1 = QtWidgets.QVBoxLayout(self.groupBoxStep1)
        self.layoutGrpStep1.setObjectName("layoutGrpStep1")
        self.checkBoxTimes = QtWidgets.QCheckBox(self.groupBoxStep1)
        self.checkBoxTimes.setChecked(True)
        self.checkBoxTimes.setObjectName("checkBoxTimes")
        self.layoutGrpStep1.addWidget(self.checkBoxTimes)
        self.checkBoxBoundaryVTT = QtWidgets.QCheckBox(self.groupBoxStep1)
        self.checkBoxBoundaryVTT.setChecked(True)
        self.checkBoxBoundaryVTT.setObjectName("checkBoxBoundaryVTT")
        self.layoutGrpStep1.addWidget(self.checkBoxBoundaryVTT)
        self.layoutToplevel.addWidget(self.groupBoxStep1)
        self.groupBoxStep2 = QtWidgets.QGroupBox(descriptivesDialog)
        self.groupBoxStep2.setObjectName("groupBoxStep2")
        self.layoutGrpStep2 = QtWidgets.QGridLayout(self.groupBoxStep2)
        self.layoutGrpStep2.setObjectName("layoutGrpStep2")
        self.checkBoxMean = QtWidgets.QCheckBox(self.groupBoxStep2)
        self.checkBoxMean.setObjectName("checkBoxMean")
        self.layoutGrpStep2.addWidget(self.checkBoxMean, 0, 0, 1, 1)
        self.checkBoxMin = QtWidgets.QCheckBox(self.groupBoxStep2)
        self.checkBoxMin.setObjectName("checkBoxMin")
        self.layoutGrpStep2.addWidget(self.checkBoxMin, 0, 1, 1, 1)
        self.checkBoxMedian = QtWidgets.QCheckBox(self.groupBoxStep2)
        self.checkBoxMedian.setObjectName("checkBoxMedian")
        self.layoutGrpStep2.addWidget(self.checkBoxMedian, 0, 2, 1, 1)
        self.checkBoxStdDev = QtWidgets.QCheckBox(self.groupBoxStep2)
        self.checkBoxStdDev.setObjectName("checkBoxStdDev")
        self.layoutGrpStep2.addWidget(self.checkBoxStdDev, 1, 0, 1, 1)
        self.checkBoxMax = QtWidgets.QCheckBox(self.groupBoxStep2)
        self.checkBoxMax.setObjectName("checkBoxMax")
        self.layoutGrpStep2.addWidget(self.checkBoxMax, 1, 1, 1, 2)
        self.layoutToplevel.addWidget(self.groupBoxStep2)
        self.btnDialogBox = QtWidgets.QDialogButtonBox(descriptivesDialog)
        self.btnDialogBox.setOrientation(QtCore.Qt.Horizontal)
        self.btnDialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close|QtWidgets.QDialogButtonBox.Ok)
        self.btnDialogBox.setCenterButtons(False)
        self.btnDialogBox.setObjectName("btnDialogBox")
        self.layoutToplevel.addWidget(self.btnDialogBox)

        self.retranslateUi(descriptivesDialog)
        self.btnDialogBox.accepted.connect(descriptivesDialog.accept)
        self.btnDialogBox.rejected.connect(descriptivesDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(descriptivesDialog)

    def retranslateUi(self, descriptivesDialog):
        _translate = QtCore.QCoreApplication.translate
        descriptivesDialog.setWindowTitle(_translate("descriptivesDialog", "Compute extra descriptives"))
        self.groupBoxStep1.setTitle(_translate("descriptivesDialog", "Step 1: Choose variables to compute descriptives"))
        self.checkBoxTimes.setText(_translate("descriptivesDialog", "Times each alternative is chosen"))
        self.checkBoxBoundaryVTT.setText(_translate("descriptivesDialog", "Boundary of Value of Travel Time"))
        self.groupBoxStep2.setTitle(_translate("descriptivesDialog", "Step 2: Choose descriptives to compute"))
        self.checkBoxMean.setText(_translate("descriptivesDialog", "Mean"))
        self.checkBoxMin.setText(_translate("descriptivesDialog", "Min"))
        self.checkBoxMedian.setText(_translate("descriptivesDialog", "Median"))
        self.checkBoxStdDev.setText(_translate("descriptivesDialog", "Std. Dev."))
        self.checkBoxMax.setText(_translate("descriptivesDialog", "Max"))
