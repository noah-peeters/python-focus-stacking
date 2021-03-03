from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5.uic import loadUi

import sys
from pathlib import Path
import time
import collections
import dask

SUPPORTED_IMAGE_FORMATS = "(*.png *.jpg)"

class LoadImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(str)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, algorithm):
        super().__init__()

        self.files = files
        self.is_killed = False

        # Initialize algorithm
        self.Algorithm = algorithm

    def run(self):
        start_time = time.time()
        image_table = []
        for image_path in self.files:
            bool = self.Algorithm.load_image(image_path)

            if not bool: # Operation failed
                break

            image_table.append(image_path) # Append loaded image to table

            if self.is_killed:  # Operation stopped from UI (image load still successful)
                break
            # Send finished image path back to UI
            self.finishedImage.emit(image_path)

        # Operation ended
        self.finished.emit({
            "execution_time": round(time.time() - start_time, 4), 
            "image_table": image_table, 
            "killed_by_user": self.is_killed
            }
        )
    
    def kill(self):
        self.is_killed = True

class AlignImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(list)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, algorithm):
        super().__init__()

        self.files = files
        self.is_killed = False

        # Initialize algorithm
        self.Algorithm = algorithm

    def run(self):
        start_time = time.time()
        image0 = self.files[round(len(self.files)/2)] # Get middle image
        aligned_images = [] # Table for processed images (to check if all have been loaded)
        for image_path in self.files:
            image0, image1, success = self.Algorithm.align_image(image0, image_path)

            if not success or self.is_killed: # Operation failed or stopped from UI
                break

            aligned_images.append(image_path) # Append aligned image
            # Send progress back to UI
            self.finishedImage.emit([image0, image1])

        # Operation ended
        self.finished.emit({
            "execution_time": round(time.time() - start_time, 4), 
            "image_table": aligned_images, 
            "killed_by_user": self.is_killed
            }
        )
    
    def kill(self):
        self.is_killed = True

class StackImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(str)
    finished_laplacian = qtc.pyqtSignal(bool)
    row_update = qtc.pyqtSignal(int)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, gaussian_blur_size, laplacian_kernel_size, algorithm):
        super().__init__()

        self.files = files
        self.gaussian_blur_size = gaussian_blur_size
        self.laplacian_kernel_size = laplacian_kernel_size
        self.is_killed = False

        # Initialize algorithm
        self.Algorithm = algorithm

    def run(self):
        start_time = time.time()
        """
            Compute laplacian edges
        """
        laplacian_images = [] # Table for processed laplacians
        for image_path in self.files:
            success = self.Algorithm.compute_laplacian_image(image_path, self.gaussian_blur_size, self.laplacian_kernel_size)

            if not success or self.is_killed: # Operation failed or stopped from UI
                break

            laplacian_images.append(image_path) # Append aligned image
            # Send progress back to UI
            self.finishedImage.emit(image_path)
        
        operation_success = False
        if collections.Counter(laplacian_images) == collections.Counter(self.files): # All laplacian images computed?
            # Laplacian edges computation finished successfully, start stacking
            self.finished_laplacian.emit(True)
            """
                Start stacking operation using previously computed laplacian images
            """
            row_counter = 0
            for current_row in self.Algorithm.stack_images(self.files):
                if not current_row or self.is_killed: # Operation failed or stopped from UI
                    break
                row_counter += 1
                self.row_update.emit(current_row)
            
            if row_counter == self.Algorithm.get_image_shape()[0]: # have all rows been processed?
                operation_success = True

        else:
            # Failed to compute laplacians
            self.finished_laplacian.emit(False)

        # Operation ended
        self.finished.emit({
            "execution_time": round(time.time() - start_time, 4), 
            "operation_success": operation_success,
            "killed_by_user": self.is_killed
            }
        )
    
    def kill(self):
        self.is_killed = True

class MainWindow(qtw.QMainWindow):
    loaded_image_files = []

    def __init__(self):
        super().__init__()

        loadUi("QtUi/main.ui", self)
        # Top bar setup
        # File
        self.load_images_action.triggered.connect(self.load_images)
        self.save_project_action.triggered.connect(self.save_project)
        self.export_output_action.triggered.connect(self.export_output)
        # Processing
        self.align_images_action.triggered.connect(lambda: self.images_loaded_check([self.align_images]))
        self.stack_images_action.triggered.connect(lambda: self.images_loaded_check([self.stack_images]))
        self.align_and_stack_images_action.triggered.connect(lambda: self.images_loaded_check([self.align_images, self.stack_images])) # First align, then stack

        self.showMaximized() # Show main window in full screen

        # Initialize algorithm
        from QtUi.algorithm import MainAlgorithm
        self.Algorithm = MainAlgorithm()
        # Initialize Utilities
        from QtUi.utilities import Utilities
        self.Utilities = Utilities()

    def load_images(self):
        home_dir = str(Path.home())
        # Prompt file selection screen (discard filter that is returned)
        self.loaded_image_files, _ = qtw.QFileDialog.getOpenFileNames(self, 'Select images to load.', home_dir, "Image files " + SUPPORTED_IMAGE_FORMATS)

        if not self.loaded_image_files or len(self.loaded_image_files) <= 0:
            return # No images have been selected!

        # Get total size of all images to import
        total_size = 0
        for image_path in self.loaded_image_files:
            file_size = self.Utilities.get_file_size_MB(image_path)
            if file_size:
                total_size += file_size
        total_size = round(total_size, 2)

        # load images and prompt loading screen
        image_progress = qtw.QProgressDialog("Loading... ", "Cancel", 0, len(self.loaded_image_files), self)
        image_progress.setWindowTitle("Loading " + str(len(self.loaded_image_files)) + " images..., Total size: " + str(total_size) + " MB")
        image_progress.setLabelText("Preparing to load your images. This shouldn't take long. Please wait.")
        image_progress.setValue(0)
        image_progress.setWindowModality(qtc.Qt.WindowModal)

        counter = 0
        def update_progress(image_path):
            nonlocal counter
            counter += 1
            # update label text
            image_progress.setLabelText("Just loaded: " + self.Utilities.get_file_name(image_path))

            # Update progress slider
            image_progress.setValue(counter)

        loading = LoadImages(self.loaded_image_files, self.Algorithm)
        loading.finishedImage.connect(update_progress)  # Update progress callback

        def finished_loading(returned):
            self.loaded_image_files = returned["image_table"]   # Set loaded images

            """
            Create pop-up on operation finish.
            """
            props = {}
            props["progress_bar"] = image_progress
            # Success message
            props["success_message"] = qtw.QMessageBox(self)
            props["success_message"].setIcon(qtw.QMessageBox.Information)
            props["success_message"].setWindowTitle("Images loaded successfully!")
            props["success_message"].setText(str(len(self.loaded_image_files)) + " images have been loaded.")
            props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
            # Error message
            props["error_message"] = qtw.QMessageBox(self)
            props["error_message"].setIcon(qtw.QMessageBox.Critical)
            props["error_message"].setWindowTitle("Loading of images failed!")
            props["error_message"].setText("Something went wrong while loading your images. Please retry.")
            # User killed operation message
            props["user_killed_message"] = qtw.QMessageBox(self)
            props["user_killed_message"].setIcon(qtw.QMessageBox.Information)
            props["user_killed_message"].setWindowTitle("Operation canceled by user.")
            props["user_killed_message"].setText("Image loading has successfully been canceled by user. Images that were loaded are still available.")

            self.result_message(returned, props)                # Display message about operation
        
        loading.finished.connect(finished_loading)              # Connection on finished

        image_progress.canceled.connect(loading.kill)   # Stop image loading on cancel
        loading.start()

        image_progress.exec_()

    def save_project(self):
        print("Saving project")

    def export_output(self):
        print("Exporting output image")

    # Only run functions if files are loaded
    def images_loaded_check(self, func_table):
        if self.loaded_image_files and len(self.loaded_image_files) > 0: # Check if images have been loaded
            # Run functions
            for func in func_table:
                func()

        else: # Display message telling user to load images
            error_message = qtw.QMessageBox(self)
            error_message.setIcon(qtw.QMessageBox.Critical)
            error_message.setWindowTitle("No images are loaded!")
            error_message.setText("No images have been loaded. Please load them in from the 'file' menu.")
            error_message.exec_()

    def align_images(self):
        # Progress bar
        qtw.QProgressDialog()
        align_progress = qtw.QProgressDialog("", "Cancel", 0, len(self.loaded_image_files), self)
        align_progress.setWindowModality(qtc.Qt.WindowModal)
        align_progress.setValue(0)
        align_progress.setWindowTitle("Aligning " + str(len(self.loaded_image_files)) + " images.")
        align_progress.setLabelText("Preparing to align your images. This shouldn't take long. Please wait.")

        counter = 0
        def update_progress(l):
            nonlocal counter
            image0 = l[0]
            image1 = l[1]
            # update label text
            align_progress.setLabelText("Just aligned: " + self.Utilities.get_file_name(image1) + " to: " + self.Utilities.get_file_name(image0))
            counter += 1

            # Update progress slider
            align_progress.setValue(counter)


        aligning = AlignImages(self.loaded_image_files, self.Algorithm)
        aligning.finishedImage.connect(update_progress)

        def finished_loading(returned):
            """
                Create pop-up on operation finish.
            """
            props = {}
            props["progress_bar"] = align_progress
            # Success message
            props["success_message"] = qtw.QMessageBox(self)
            props["success_message"].setIcon(qtw.QMessageBox.Information)
            props["success_message"].setWindowTitle("Images aligned successfully!")
            props["success_message"].setText(str(len(self.loaded_image_files)) + " images have been aligned.")
            props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
            # Error message
            props["error_message"] = qtw.QMessageBox(self)
            props["error_message"].setIcon(qtw.QMessageBox.Critical)
            props["error_message"].setWindowTitle("Aligning of images failed!")
            props["error_message"].setText("Something went wrong while aligning your images. Please retry.")
            # User killed operation message
            props["user_killed_message"] = qtw.QMessageBox(self)
            props["user_killed_message"].setIcon(qtw.QMessageBox.Information)
            props["user_killed_message"].setWindowTitle("Operation canceled by user.")
            props["user_killed_message"].setText("Image alignment has successfully been canceled by user.")

            self.result_message(returned, props)

        aligning.finished.connect(finished_loading)
        align_progress.canceled.connect(aligning.kill) # Kill operation on "cancel" press

        aligning.start()
        align_progress.exec_()

    def stack_images(self):
        print("Stacking images")



    # Display result message after operation finished
    def result_message(self, returned_table, props):
        # "Unpack" values
        execution_time = returned_table["execution_time"]
        image_table = returned_table["image_table"]
        killed_by_user = returned_table["killed_by_user"]

        progress_bar = props["progress_bar"]
        success_message = props["success_message"]
        error_message = props["error_message"]
        user_killed_message = props["user_killed_message"]

        if progress_bar:
            progress_bar.close() # Hide progress bar

        # Are all loaded images inside of returned table?
        success = collections.Counter(image_table) == collections.Counter(self.loaded_image_files)
        if success and not killed_by_user: # Display success message
            # Get names of files
            file_names = []
            for image_path in self.loaded_image_files:
                file_names.append(self.Utilities.get_file_name(image_path) + ", ")

            success_message.setDetailedText(''.join(str(file) for file in file_names)) # Display names inside "Detailed text"
            success_message.setInformativeText("Execution time: " + self.Utilities.format_seconds(execution_time)) # Display execution time
            success_message.exec_()

        elif not killed_by_user: # Display error message (error occured)
            error_message.exec_()

        else: # User has stopped process. Show confirmation
            user_killed_message.exec_()



if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())