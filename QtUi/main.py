from PyQt5 import QtWidgets as qtw
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import sys
from pathlib import Path

SUPPORTED_IMAGE_FORMATS = "(*.png *.jpg)"

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("QtUi/main.ui", self)

        self.load_images_action.triggered.connect(self.load_images)
        self.save_project_action.triggered.connect(self.save_project)
        self.export_output_action.triggered.connect(self.export_output)

        self.show()

    def load_images(self):
        print("Loading images")
        home_dir = str(Path.home())
        # Prompt file selection screen
        files = QFileDialog.getOpenFileNames(self, 'Select images to load.', home_dir, "Image files " + SUPPORTED_IMAGE_FORMATS)

        # load images and prompt loading screen (if necessary)
        image_progress = QProgressDialog("Loading images...", "Cancel", 0, 100, self)
        image_progress.setWindowTitle("Loading images...")
        image_progress.setWindowModality(Qt.WindowModal)
        image_progress.setValue(0)
        image_progress.show()
        for i in range(101):
            loop = QEventLoop()
            QTimer.singleShot(100, loop.quit)
            loop.exec_()
            image_progress.setValue(i)


    def save_project(self):
        print("Saving project")

    def export_output(self):
        print("Exporting output image")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())