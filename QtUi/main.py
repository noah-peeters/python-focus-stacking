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
            # Return error
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()

class LoadImages(QThread):
    indexChanged = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, files):
        super(LoadImages, self).__init__()

        self.files = files
        self.is_killed = False

        # Initialize algorithm
        from QtUi.algorithm import MainAlgorithm
        self.Algorithm = MainAlgorithm()

    def run(self):
        start_time = time.time()
        counter = 0
        for i, image_path in enumerate(self.files):
            bool = self.Algorithm.load_image(image_path)

            if not bool or self.is_killed: # Operation failed or stopped from UI
                break

            counter += 1
            # Send progress back to UI
            self.indexChanged.emit(i)

        execution_time = round(time.time() - start_time, 2) # Execution time
        success = counter == len(self.files)                # All files loaded? no: display error, yes: display success
        self.finished.emit([execution_time, success, self.is_killed])
    
    def kill(self):
        self.is_killed = True

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("QtUi/main.ui", self)

        # Top bar setup
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
        image_progress.setLabelText("Preparing to load your images. This shouldn't take long. Please wait.")
        image_progress.setValue(0)
        image_progress.setWindowModality(Qt.WindowModal)

        def update_progress(new_index):
            # update label text
            image_progress.setLabelText("Just loaded: " + self.Utilities.get_file_name(files[new_index]))

            # Update progress slider
            image_progress.setValue(new_index + 1) # +1 as python lists start at 0

        def create_message(l):
            message = None
            image_progress.close() # Hide progress bar

            execution_time = l[0]
            success = l[1]
            killed_by_user = l[2]
            if success: # Display success message
                message = QMessageBox()
                message.setIcon(QMessageBox.Information)
                message.setWindowTitle("Images loaded successfully!")
                message.setText(str(len(files)) + " images have been loaded.")
                message.setInformativeText("Execution time: " + self.Utilities.format_seconds(execution_time))
                # Get names of files
                file_names = []
                for image_path in files:
                    file_names.append(self.Utilities.get_file_name(image_path) + ", ")
                # Display names inside "Detailed text"
                message.setDetailedText(''.join(str(file) for file in file_names))
                message.setStandardButtons(QMessageBox.Ok)

            elif not killed_by_user: # Display error message (error occured)
                message = QMessageBox()
                message.setIcon(QMessageBox.Critical)
                message.setWindowTitle("An error has occured!")
                message.setText("Something went wrong while loading your images! Please try again later.")
            else: # User has stopped process. confirm
                message = QMessageBox()
                message.setIcon(QMessageBox.Information)
                message.setWindowTitle("Cancelled operation.")
                message.setText("Image loading was successfully canceled by user. Images that were loaded before canceling are still available.")

            message.exec_()

        loading = LoadImages(files)
        loading.indexChanged.connect(update_progress)   # Update progress callback
        loading.finished.connect(create_message)        # Create message
        image_progress.canceled.connect(loading.kill)   # Stop image loading on cancel
        loading.start()

        image_progress.exec_()

    def save_project(self):
        print("Saving project")

    def export_output(self):
        print("Exporting output image")

app = QApplication([])
window = MainWindow()
sys.exit(app.exec_())