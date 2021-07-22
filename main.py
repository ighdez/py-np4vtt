import sys
from PyQt5.QtWidgets import QApplication
from gui.mainwindow import MainWindow


if __name__ == '__main__':
    app = QApplication([])
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
