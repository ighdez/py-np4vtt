# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_modelconfig_rouwendal.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_configRouwendalDialog(object):
    def setupUi(self, configRouwendalDialog):
        configRouwendalDialog.setObjectName("configRouwendalDialog")
        configRouwendalDialog.resize(641, 261)
        self.layoutToplevel = QtWidgets.QVBoxLayout(configRouwendalDialog)
        self.layoutToplevel.setObjectName("layoutToplevel")
        self.layoutMain = QtWidgets.QHBoxLayout()
        self.layoutMain.setObjectName("layoutMain")
        self.groupBoxStep1 = QtWidgets.QGroupBox(configRouwendalDialog)
        self.groupBoxStep1.setObjectName("groupBoxStep1")
        self.layoutGrpStep1 = QtWidgets.QFormLayout(self.groupBoxStep1)
        self.layoutGrpStep1.setObjectName("layoutGrpStep1")
        self.labelPointsMinimum = QtWidgets.QLabel(self.groupBoxStep1)
        self.labelPointsMinimum.setObjectName("labelPointsMinimum")
        self.layoutGrpStep1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelPointsMinimum)
        self.fieldPointsMinimum = QtWidgets.QSpinBox(self.groupBoxStep1)
        self.fieldPointsMinimum.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fieldPointsMinimum.setObjectName("fieldPointsMinimum")
        self.layoutGrpStep1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldPointsMinimum)
        self.labelPointsMaximum = QtWidgets.QLabel(self.groupBoxStep1)
        self.labelPointsMaximum.setObjectName("labelPointsMaximum")
        self.layoutGrpStep1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelPointsMaximum)
        self.fieldPointsMaximum = QtWidgets.QSpinBox(self.groupBoxStep1)
        self.fieldPointsMaximum.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fieldPointsMaximum.setObjectName("fieldPointsMaximum")
        self.layoutGrpStep1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fieldPointsMaximum)
        self.labelPointsSupport = QtWidgets.QLabel(self.groupBoxStep1)
        self.labelPointsSupport.setObjectName("labelPointsSupport")
        self.layoutGrpStep1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelPointsSupport)
        self.fieldPointsSupport = QtWidgets.QSpinBox(self.groupBoxStep1)
        self.fieldPointsSupport.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fieldPointsSupport.setMinimum(1)
        self.fieldPointsSupport.setObjectName("fieldPointsSupport")
        self.layoutGrpStep1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.fieldPointsSupport)
        self.layoutMain.addWidget(self.groupBoxStep1)
        self.layoutGrpsRight = QtWidgets.QVBoxLayout()
        self.layoutGrpsRight.setObjectName("layoutGrpsRight")
        self.groupBoxStep2 = QtWidgets.QGroupBox(configRouwendalDialog)
        self.groupBoxStep2.setObjectName("groupBoxStep2")
        self.layoutGrpStep2 = QtWidgets.QVBoxLayout(self.groupBoxStep2)
        self.layoutGrpStep2.setObjectName("layoutGrpStep2")
        self.layoutProbConsistent = QtWidgets.QVBoxLayout()
        self.layoutProbConsistent.setObjectName("layoutProbConsistent")
        self.labelProbConsistent = QtWidgets.QLabel(self.groupBoxStep2)
        self.labelProbConsistent.setObjectName("labelProbConsistent")
        self.layoutProbConsistent.addWidget(self.labelProbConsistent)
        self.layoutProbSlider = QtWidgets.QHBoxLayout()
        self.layoutProbSlider.setObjectName("layoutProbSlider")
        self.labelProbSliderMin = QtWidgets.QLabel(self.groupBoxStep2)
        self.labelProbSliderMin.setObjectName("labelProbSliderMin")
        self.layoutProbSlider.addWidget(self.labelProbSliderMin)
        self.sliderProbConsistent = QtWidgets.QSlider(self.groupBoxStep2)
        self.sliderProbConsistent.setProperty("value", 90)
        self.sliderProbConsistent.setOrientation(QtCore.Qt.Horizontal)
        self.sliderProbConsistent.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderProbConsistent.setTickInterval(10)
        self.sliderProbConsistent.setObjectName("sliderProbConsistent")
        self.layoutProbSlider.addWidget(self.sliderProbConsistent)
        self.labelProbSliderMax = QtWidgets.QLabel(self.groupBoxStep2)
        self.labelProbSliderMax.setObjectName("labelProbSliderMax")
        self.layoutProbSlider.addWidget(self.labelProbSliderMax)
        self.layoutProbConsistent.addLayout(self.layoutProbSlider)
        self.layoutGrpStep2.addLayout(self.layoutProbConsistent)
        spacerItem = QtWidgets.QSpacerItem(20, 31, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layoutGrpStep2.addItem(spacerItem)
        self.layoutGrpsRight.addWidget(self.groupBoxStep2)
        self.groupBoxAdvanced = QtWidgets.QGroupBox(configRouwendalDialog)
        self.groupBoxAdvanced.setObjectName("groupBoxAdvanced")
        self.layoutGrpAdvanced = QtWidgets.QFormLayout(self.groupBoxAdvanced)
        self.layoutGrpAdvanced.setObjectName("layoutGrpAdvanced")
        self.labelMaxIterations = QtWidgets.QLabel(self.groupBoxAdvanced)
        self.labelMaxIterations.setObjectName("labelMaxIterations")
        self.layoutGrpAdvanced.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelMaxIterations)
        self.fieldMaxIterations = QtWidgets.QLineEdit(self.groupBoxAdvanced)
        self.fieldMaxIterations.setObjectName("fieldMaxIterations")
        self.layoutGrpAdvanced.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fieldMaxIterations)
        self.layoutGrpsRight.addWidget(self.groupBoxAdvanced)
        self.layoutMain.addLayout(self.layoutGrpsRight)
        self.layoutToplevel.addLayout(self.layoutMain)
        self.btnDialogBox = QtWidgets.QDialogButtonBox(configRouwendalDialog)
        self.btnDialogBox.setOrientation(QtCore.Qt.Horizontal)
        self.btnDialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.btnDialogBox.setCenterButtons(True)
        self.btnDialogBox.setObjectName("btnDialogBox")
        self.layoutToplevel.addWidget(self.btnDialogBox)

        self.retranslateUi(configRouwendalDialog)
        self.btnDialogBox.accepted.connect(configRouwendalDialog.accept)
        self.btnDialogBox.rejected.connect(configRouwendalDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(configRouwendalDialog)

    def retranslateUi(self, configRouwendalDialog):
        _translate = QtCore.QCoreApplication.translate
        configRouwendalDialog.setWindowTitle(_translate("configRouwendalDialog", "Rouwendal\'s Method - Configuration"))
        self.groupBoxStep1.setTitle(_translate("configRouwendalDialog", "Step 1: Set the VTT support points"))
        self.labelPointsMinimum.setText(_translate("configRouwendalDialog", "Minimum:"))
        self.labelPointsMaximum.setText(_translate("configRouwendalDialog", "Maximum:"))
        self.labelPointsSupport.setText(_translate("configRouwendalDialog", "Number of support points:"))
        self.groupBoxStep2.setTitle(_translate("configRouwendalDialog", "Step 2: Set the starting value for MLE estimation"))
        self.labelProbConsistent.setText(_translate("configRouwendalDialog", "Probability of consistent choice:"))
        self.labelProbSliderMin.setText(_translate("configRouwendalDialog", "0%"))
        self.labelProbSliderMax.setText(_translate("configRouwendalDialog", "100%"))
        self.groupBoxAdvanced.setTitle(_translate("configRouwendalDialog", "Advanced setup (optional)"))
        self.labelMaxIterations.setText(_translate("configRouwendalDialog", "Maximum iterations for MLE:"))
        self.fieldMaxIterations.setText(_translate("configRouwendalDialog", "10000"))
