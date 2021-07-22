from PyQt5.QtWidgets import QMainWindow
from gui.formclasses.ui_mainwindow import Ui_mainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
