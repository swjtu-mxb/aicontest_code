# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(536, 582)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgLabel = QtWidgets.QLabel(Form)
        self.imgLabel.setMinimumSize(QtCore.QSize(512, 512))
        self.imgLabel.setMaximumSize(QtCore.QSize(512, 512))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.verticalLayout.addWidget(self.imgLabel)
        self.imgName = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.imgName.setFont(font)
        self.imgName.setAlignment(QtCore.Qt.AlignCenter)
        self.imgName.setObjectName("imgName")
        self.verticalLayout.addWidget(self.imgName)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.imgName.setText(_translate("Form", "Result"))
