import sys
from PyQt5 import QtWidgets as qtw

app = qtw.QApplication(sys.argv)
window = qtw.QWidget()
layout = qtw.QVBoxLayout(window)

list_widget = qtw.QListWidget()
list_widget.setAlternatingRowColors(True)
# list_widget.setDragDropMode(qtw.QAbstractItemView.InternalMove)
list_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)

list_widget.addItems(["One", "Two", "Three", "Four", "Five"])

layout.addWidget(list_widget)
window.show()

sys.exit(app.exec_())