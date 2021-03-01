from PyQt5 import QtWidgets as qtw
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import sys
from pathlib import Path
import os
import dask
import time

SUPPORTED_IMAGE_FORMATS = "(*.png *.jpg)"

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("QtUi/main.ui", self)

        self.load_images_action.triggered.connect(self.load_images)
        self.save_project_action.triggered.connect(self.save_project)
        self.export_output_action.triggered.connect(self.export_output)

        self.show()

        # Initialize algorithm
        from QtUi.algorithm import MainAlgorithm
        self.Algorithm = MainAlgorithm()
        # Initialize Utilities
        from QtUi.utilities import Utilities
        self.Utilities = Utilities()

    def load_images(self):
        home_dir = str(Path.home())
        # Prompt file selection screen (discard filter that is returned)
        files, _ = QFileDialog.getOpenFileNames(self, 'Select images to load.', home_dir, "Image files " + SUPPORTED_IMAGE_FORMATS)


        def get_file_size_MB(path):
            if os.path.exists(path):
                size = os .path.getsize(path)
                return size / 1e+6 # Bytes to MegaBytes
            return 0

        # Get total size of all images to import
        total_size = 0
        for i, image_path in enumerate(files):
            file_size = get_file_size_MB(image_path)
            if file_size:
                total_size += file_size
        total_size = round(total_size, 2)

        # load images and prompt loading screen
        image_progress = QProgressDialog("Loading... ", "Cancel", 0, len(files), self)
        image_progress.setWindowTitle("Loading " + str(len(files)) + " images..., Total size: " + str(total_size) + " MB")
        image_progress.setValue(0)
        image_progress.setWindowModality(Qt.WindowModal)
        image_progress.show()

        index = 0
        start_time = time.time()
        for image_path in self.Algorithm.load_images_generator(files).compute():
            file_name = self.Utilities.get_file_name(image_path)
            index += 1
            # update label text
            image_progress.setLabelText("Just loaded: " + file_name)

            # Update progress slider
            image_progress.setValue(round(index))

            if image_progress.wasCanceled():
                break
        image_progress.setValue(len(files))

        # Display information dialog
        message = QMessageBox()
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Images loaded successfully!")
        message.setText(str(len(files)) + " images have been loaded.")
        message.setInformativeText("Execution time: " + self.Utilities.format_seconds(round(time.time() - start_time, 2)))
        # Get names of files
        file_names = []
        for image_path in files:
            file_names.append(self.Utilities.get_file_name(image_path) + ", ")
        # Display names inside "Detailed text"
        message.setDetailedText(''.join(str(file) for file in file_names))
        message.setStandardButtons(QMessageBox.Ok)
        message.exec_()


    def save_project(self):
        print("Saving project")

    def export_output(self):
        print("Exporting output image")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())