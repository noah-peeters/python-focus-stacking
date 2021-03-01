from PyQt5 import QtWidgets as qtw
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import sys
from pathlib import Path
import os
import time
import traceback, sys

SUPPORTED_IMAGE_FORMATS = "(*.png *.jpg)"

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("QtUi/main.ui", self)

        # Top bar setup
        self.load_images_action.triggered.connect(self.load_images)
        self.save_project_action.triggered.connect(self.save_project)
        self.export_output_action.triggered.connect(self.export_output)

        self.show()

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


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

        start_time = time.time()

        # Load images (separate thread)
        def process(progress_callback):
            for i, image_path in enumerate(files):
                self.Algorithm.load_image(image_path)
                progress_callback.emit(i)

                if image_progress.wasCanceled():
                    break

        # Update progress bar display
        counter = 0
        def update_progress(index_of_file):
            nonlocal counter
            # update label text
            image_progress.setLabelText("Just loaded: " + self.Utilities.get_file_name(files[index_of_file]))

            # Update progress slider
            counter += 1
            image_progress.setValue(counter)

        # Create info message about finished operation
        def create_info():
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

        # Run process on separate thread
        worker = Worker(process)
        worker.signals.finished.connect(create_info)
        worker.signals.progress.connect(update_progress)

        self.threadpool.start(worker)

        # Show progress bar
        image_progress.exec_()


    def save_project(self):
        print("Saving project")

    def export_output(self):
        print("Exporting output image")

app = QApplication([])
window = MainWindow()
sys.exit(app.exec_())