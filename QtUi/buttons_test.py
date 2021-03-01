from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import *
import sys

class Window(QWidget):
    def __init__(self, val):
        super().__init__()
        self.title = "PyQt5 Scroll Bar"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        formLayout =QFormLayout()
        groupBox = QGroupBox("This Is Group Box")
        labelLis = []
        comboList = []
        for i in  range(val):
            labelLis.append(QLabel("Label"))
            comboList.append(QPushButton("Click Me"))
            formLayout.addRow(labelLis[i], comboList[i])
        groupBox.setLayout(formLayout)
        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(400)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        self.show()

App = QApplication(sys.argv)
window = Window(30)
sys.exit(App.exec())