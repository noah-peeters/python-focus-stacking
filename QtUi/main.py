import sys
import collections
from pathlib import Path
import time
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg

from utilities import Utilities
Utilities = Utilities()

SUPPORTED_IMAGE_FORMATS = "(*.jpg *.png)"

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

class CalculateLaplacians(qtc.QThread):
    image_finished = qtc.pyqtSignal(str)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, gaussian_blur_size, laplacian_kernel_size, algorithm):
        super().__init__()

        self.start_time = 0
        self.laplacian_success = False
        self.is_killed = False

        self.files = files
        self.gaussian_blur_size = gaussian_blur_size
        self.laplacian_kernel_size = laplacian_kernel_size

        # Initialize algorithm
        self.Algorithm = algorithm

    def run(self):
        self.start_time = time.time()
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
            self.image_finished.emit(image_path)
        
        if collections.Counter(laplacian_images) == collections.Counter(self.files): # All laplacian images computed?
            self.laplacian_success = True   # Laplacian edges computation success
        
        # Operation ended
        self.finished.emit({
            "execution_time": round(time.time() - self.start_time, 4),
            "operation_success": self.laplacian_success,
            "killed_by_user": self.is_killed
            }
        )
    
    # Kill operation
    def kill(self):
        self.is_killed = True

class FinalStacking(qtc.QThread):
    row_finished = qtc.pyqtSignal(int)
    finished = qtc.pyqtSignal(dict)
    def __init__(self, files, algoritm):
        super().__init__()
        self.files = files
        self.Algorithm = algoritm
        self.is_killed = False
        self.stack_success = False
        self.start_time = 0

    def run(self):
        """
            Start stacking operation using previously computed laplacian images
        """
        self.start_time = time.time()
        row_reference = 0
        for current_row in self.Algorithm.stack_images(self.files):
            if type(current_row) != int or self.is_killed:   # Operation failed or stopped from UI
                break

            self.row_finished.emit(current_row)
            row_reference = current_row
        
        if row_reference == self.Algorithm.get_image_shape()[0]: # have all rows been processed?
            self.stack_success = True

        # Operation ended
        self.finished.emit({
            "execution_time": round(time.time() - self.start_time, 4),
            "operation_success": self.stack_success,
            "killed_by_user": self.is_killed
            }
        )
    
    def kill(self):
        self.is_killed = True


class MainWindow(qtw.QMainWindow):
    current_directory = None
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyStacker")
        self.setStatusBar(qtw.QStatusBar())     # Create status bar
        self.setup_file_menu()                  # Setup file menu (top bar)

        self.main_layout = MainLayout(self)     # Main layout setup
        self.setCentralWidget(self.main_layout)
        self.showMaximized()                    # Show fullscreen

        self.Preferences = Preferences(self)    # Init Preferences

        from algorithm import MainAlgorithm
        self.Algorithm = MainAlgorithm()        # Init algorithm

        from utilities import Utilities
        self.Utilities = Utilities()            # Init utilities

    # Setup file menu (topbar)
    def setup_file_menu(self):
        # Get built-in icon from name (shorthand)
        def get_icon(name):
            style = self.style()
            return style.standardIcon(getattr(qtw.QStyle, name))

        # Menu bar setup
        menu_bar = qtw.QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Menus
        file_menu = menu_bar.addMenu("&File")
        processing_menu = menu_bar.addMenu("&Processing")
        view_menu = menu_bar.addMenu("&View")
        image_preview_menu = view_menu.addMenu("&Image preview")
        edit_menu = menu_bar.addMenu("&Edit")
        help_menu = menu_bar.addMenu("&Help")

        # Create actions
        new_action = qtw.QAction("&New file", self, shortcut="Ctrl+N", triggered=self.create_new_file)
        open_action = qtw.QAction("&Open file", self, shortcut="Ctrl+O", triggered=self.open_file)
        save_file = qtw.QAction("&Save file", self, shortcut="Ctrl+S", triggered=self.save_file)
        load_images_action = qtw.QAction("&Load images", self, shortcut="Ctrl+L", triggered=self.load_images)
        clear_loaded_images_action = qtw.QAction("&Clear loaded images", self, shortcut="Ctrl+Alt+C", triggered=self.clear_loaded_images)
        export_action = qtw.QAction("&Export image", self, shortcut="Ctrl+E", triggered=self.export_image)
        quit_action = qtw.QAction("&Quit", self, shortcut="Ctrl+W", triggered=lambda: self.close())

        self.align_images_action = qtw.QAction("&Align images", self, shortcut="Ctrl+Shift+A", triggered=self.align_images, enabled=False)  # Disabled by default
        self.stack_images_action = qtw.QAction("&Stack Images", self, shortcut="Ctrl+Shift+S", triggered=self.stack_images, enabled=False)
        self.align_and_stack_images_action = qtw.QAction("&Align and stack images", self, shortcut="Ctrl+Shift+P", triggered=self.align_and_stack_images, enabled=False)

        self.image_preview_reset_zoom = qtw.QAction("&Reset zoom on preview", self, enabled=False, checkable=True, checked=True)

        preferences_action = qtw.QAction("&Preferences", self, shortcut="Ctrl+P", triggered=self.open_preferences)

        about_app_action = qtw.QAction("&About PyStacker", self, triggered=self.about_application)
        about_qt_action = qtw.QAction("&About Qt", self, triggered=qtw.qApp.aboutQt)

        # Setup help tips for actions
        new_action.setStatusTip("Create a new file. Unsaved progress will be lost!")
        open_action.setStatusTip("Open a file on disk. Unsaved progress will be lost!")
        save_file.setStatusTip("Save file to disk.")
        load_images_action.setStatusTip("Load images from disk.")
        clear_loaded_images_action.setStatusTip("Clear all loaded images.")
        export_action.setStatusTip("Export output image.")
        quit_action.setStatusTip("Exit the application. Unsaved progress will be lost!")

        self.align_images_action.setStatusTip("Align images.")
        self.stack_images_action.setStatusTip("Focus stack images.")
        self.align_and_stack_images_action.setStatusTip("Align images and focus stack them.")

        self.image_preview_reset_zoom.setStatusTip("Reset zoom on new image select.")

        preferences_action.setStatusTip("Preferences: themes and other settings.")

        about_app_action.setStatusTip("About this application.")
        about_qt_action.setStatusTip("About Qt, the framework that was used to design this ui.")

        # Icons for actions
        new_action.setIcon(get_icon("SP_FileIcon"))
        save_file.setIcon(get_icon("SP_DialogSaveButton"))
        open_action.setIcon(get_icon("SP_DialogOpenButton"))
        load_images_action.setIcon(get_icon("SP_FileDialogNewFolder"))
        clear_loaded_images_action.setIcon(get_icon("SP_DialogDiscardButton"))
        export_action.setIcon(get_icon("SP_DialogYesButton"))
        quit_action.setIcon(get_icon("SP_TitleBarCloseButton"))

        self.align_images_action.setIcon(get_icon("SP_FileDialogContentsView"))
        self.stack_images_action.setIcon(get_icon("SP_TitleBarNormalButton"))
        self.align_and_stack_images_action.setIcon(get_icon("SP_BrowserReload"))

        about_qt_action.setIcon(get_icon("SP_TitleBarMenuButton"))

        # Add actions to menu
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_file)
        file_menu.addAction(load_images_action)
        file_menu.addAction(clear_loaded_images_action)
        file_menu.addAction(export_action)
        file_menu.addAction(quit_action)

        processing_menu.addAction(self.align_images_action)
        processing_menu.addAction(self.stack_images_action)
        processing_menu.addAction(self.align_and_stack_images_action)

        image_preview_menu.addAction(self.image_preview_reset_zoom)

        edit_menu.addAction(preferences_action)

        help_menu.addAction(about_app_action)
        help_menu.addAction(about_qt_action)

    def create_new_file(self):
        print("Create new file")

    def open_file(self):
        print("Open file")

    def save_file(self):
        print("Save file")

    def load_images(self):
        dir = None
        if self.current_directory:
            dir = self.current_directory
        else:
            dir = str(Path.home)

        # Prompt file selection screen (discard filter that is returned)
        self.loaded_image_files, _ = qtw.QFileDialog.getOpenFileNames(self, 'Select images to load.', dir, "Image files " + SUPPORTED_IMAGE_FORMATS)

        if not self.loaded_image_files or len(self.loaded_image_files) <= 0:    # No images have been selected!
            self.toggle_actions("processing", False)    # Disable processing actions
            return
        
        self.toggle_actions("processing", True)             # Enable processing actions
        self.current_directory = self.loaded_image_files[0] # Set current directory

        # Get total size of all images to import
        total_size = 0
        for image_path in self.loaded_image_files:
            file_size = self.Utilities.get_file_size_MB(image_path)
            if file_size:
                total_size += file_size
        total_size = round(total_size, 2)

        # load images and prompt loading screen
        image_progress = self.create_progress_bar()
        image_progress.setWindowTitle("Loading " + str(len(self.loaded_image_files)) + " images..., Total size: " + str(total_size) + " MB")
        image_progress.setLabelText("Preparing to load your images. This shouldn't take long. Please wait.")

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
                Update listing of loaded images
            """
            self.main_layout.set_image_list(self.loaded_image_files)

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
            self.result_message(returned, props)        # Display message about operation
        
        loading.finished.connect(finished_loading)      # Connection on finished

        image_progress.canceled.connect(loading.kill)   # Stop image loading on cancel
        loading.start()

        image_progress.exec_()
    
    def align_images(self):
        # Progress bar
        align_progress = self.create_progress_bar()
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

            self.result_message(returned, props)

        aligning.finished.connect(finished_loading)
        align_progress.canceled.connect(aligning.kill) # Kill operation on "cancel" press

        aligning.start()
        align_progress.exec_()

    def stack_images(self):
        # Laplacians progress bar
        laplacian_progress = self.create_progress_bar()
        laplacian_progress.setWindowTitle("Calculating laplacians of " + str(len(self.loaded_image_files)) + " images.")
        laplacian_progress.setLabelText("Preparing to calculate laplacian gradients of your images. This shouldn't take long. Please wait.")

        laplacian_calc = CalculateLaplacians(self.loaded_image_files, 5, 5, self.Algorithm)

        counter = 0
        def laplacian_progress_update(image_path):
            nonlocal counter
            laplacian_progress.setLabelText("Just computed laplacian gradient for: " + self.Utilities.get_file_name(image_path))
            counter += 1
            laplacian_progress.setValue(counter)

        def laplacian_finished(returned):
            props = {}
            props["progress_bar"] = laplacian_progress
            # Success message
            props["success_message"] = qtw.QMessageBox(self)
            props["success_message"].setIcon(qtw.QMessageBox.Information)
            props["success_message"].setWindowTitle("Laplacian gradients calculation success!")
            props["success_message"].setText("Laplacian gradients have been calculated for " + str(len(self.loaded_image_files)) + " images.")
            props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
            self.result_message(returned, props)        # Display message about operation

            """
                Start final stacking process
            """
            # Progress bar setup
            stack_progress = qtw.QProgressDialog("Loading... ", "Cancel", 0, self.Algorithm.get_image_shape()[0], self)
            stack_progress.setWindowModality(qtc.Qt.WindowModal)
            stack_progress.setValue(0)
            stack_progress.setWindowTitle("Final stacking of image: " + str(self.Algorithm.get_image_shape()[0]) + " rows tall, " + str(self.Algorithm.get_image_shape()[1]) + " columns wide.")
            stack_progress.setLabelText("Preparing to calculate the final focus stack of your images. This shouldn't take long. Please wait.")

            final_stacking = FinalStacking(self.loaded_image_files, self.Algorithm)

            def row_progress_update(current_row):
                stack_progress.setLabelText("Just calculated row " + str(current_row) + " of " + str(self.Algorithm.get_image_shape()[0]) + ".")
                stack_progress.setValue(current_row)

            def stack_finished(returned):
                props = {}
                props["progress_bar"] = stack_progress
                # Success message
                props["success_message"] = qtw.QMessageBox(self)
                props["success_message"].setIcon(qtw.QMessageBox.Information)
                props["success_message"].setWindowTitle("Final stack calculation success!")
                props["success_message"].setText("The final stack has successfully been calculated.")
                props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
                self.result_message(returned, props)

            final_stacking.row_finished.connect(row_progress_update)
            final_stacking.finished.connect(stack_finished)

            final_stacking.start()
            stack_progress.exec_()

        laplacian_calc.image_finished.connect(laplacian_progress_update)
        laplacian_calc.finished.connect(laplacian_finished)
        laplacian_progress.canceled.connect(laplacian_calc.kill)

        laplacian_calc.start()
        laplacian_progress.exec_()

    def export_image(self):
        dir = None
        if self.current_directory:
            dir = self.current_directory
        else:
            dir = str(Path.home)
        file_path, _ = qtw.QFileDialog.getSaveFileName(self, "Export stacked image", dir, SUPPORTED_IMAGE_FORMATS)

        if file_path:
            self.current_directory = file_path
            # Export to path
            success = self.Algorithm.export_image(file_path)
            
            if success == True:
                # Display success message
                qtw.QMessageBox.information(self, "Exported image successfully!", "Output image was successfully exported.", qtw.QMessageBox.Ok)
            else:
                # Display error message
                qtw.QMessageBox.critical(self, "Image export failed!", "Failed to export output image. Have you completed the stacking process?", qtw.QMessageBox.Ok)

    def align_and_stack_images(self):
        self.align_images()
        self.stack_images()

    def clear_loaded_images(self):
        # Display confirmation dialog
        reply = qtw.QMessageBox.question(self, "Clear all loaded images?", "Are you sure you want to clear <b>all</b> loaded images?", qtw.QMessageBox.Yes|qtw.QMessageBox.No)
        if reply == qtw.QMessageBox.Yes:
            # Clear (user confirmed)
            self.loaded_image_files = []
            self.main_layout.set_image_list(self.loaded_image_files)    # Clear image list
            self.toggle_actions("processing", False)                    # Toggle image processing actions off (no images loaded)
            self.main_layout.image_preview.setImage(None)               # Remove image from preview

            self.Algorithm.clearTempFiles() # Clear all temp files

    # Display result message after operation finished
    def result_message(self, returned_table, props):
        # "Unpack" values
        execution_time = returned_table["execution_time"]
        image_table = None
        if "image_table" in returned_table:
            image_table = returned_table["image_table"]

        operation_success = None
        if "operation_success" in returned_table:
            operation_success = returned_table["operation_success"]
        killed_by_user = returned_table["killed_by_user"]

        progress_bar = props["progress_bar"]
        success_message = props["success_message"]

        if progress_bar:
            progress_bar.close() # Hide progress bar

        # Get operation success
        success = None
        if image_table:
            # Are all loaded images inside of returned table?
            success = collections.Counter(image_table) == collections.Counter(self.loaded_image_files)
        elif operation_success:
            success = operation_success

        if success and not killed_by_user: # Display success message
            # Get names of files
            file_names = []
            for image_path in self.loaded_image_files:
                file_names.append(Utilities.get_file_name(image_path) + ", ")

            success_message.setDetailedText(''.join(str(file) for file in file_names)) # Display names inside "Detailed text"
            success_message.setInformativeText("Execution time: " + Utilities.format_seconds(execution_time)) # Display execution time
            success_message.exec_()

        elif not killed_by_user: # Display error message (error occured)
            error_message = qtw.QMessageBox(self)
            error_message.setIcon(qtw.QMessageBox.Critical)
            error_message.setWindowTitle("Something went wrong!")
            error_message.setText("Operation failed. Please retry.")
            error_message.exec_()

        else: # User has stopped process. Show confirmation
            user_killed_message = qtw.QMessageBox(self)
            user_killed_message.setIcon(qtw.QMessageBox.Information)
            user_killed_message.setWindowTitle("Operation canceled by user.")
            user_killed_message.setText("Operation successfully canceled by user.")
            user_killed_message.exec_()

    # Progress bar shorthand
    def create_progress_bar(self):
        progress = None
        progress = qtw.QProgressDialog("Loading...", "Cancel", 0, len(self.loaded_image_files), self)
        progress.setWindowModality(qtc.Qt.WindowModal)
        progress.setValue(0)
        return progress

    def about_application(self):
        qtw.QMessageBox.about(self, "About PyStacker",
        "<b>PyStacker</b> is an application written in <b>Python</b> and <b>Qt</b> that is completely open-source: "
        "<a href='https://github.com/noah-peeters/python-focus-stacking'>PyStacker on github</a>"
        " and aims to provide an advanced focus stacking application for free."
        )

    # Enable or disable processing actions (only enable when images are loaded)
    def toggle_actions(self, menu_name, bool):
        if menu_name == "processing":
            self.align_images_action.setEnabled(bool)
            self.stack_images_action.setEnabled(bool)
            self.align_and_stack_images_action.setEnabled(bool)
        elif menu_name == "image_preview":
            self.image_preview_reset_zoom.setEnabled(bool)

    def open_preferences(self):
        self.Preferences.exec_()

# Main layout for MainWindow (splitter window)
class MainLayout(qtw.QWidget):
    image_names = []
    image_scale_factor = 1
    scale_step = 0.25
    image_scale_factor_range = [1, 5]

    def __init__(self, parent):        
        super().__init__(parent)
        self.list_widget = ImageListWidget(self)
        self.image_preview = ImageViewer(self)

        self.parent_status_bar = parent.statusBar()
        self.toggle_actions = parent.toggle_actions

        # List widget image clicked, show preview
        self.list_widget.loaded_images_list.itemClicked.connect(self.setImage)

        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.image_preview)

        screen_width = qtw.QDesktopWidget().availableGeometry(0).width()
        splitter.setSizes([round(screen_width / 5), screen_width])
        
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

        parent.image_preview_reset_zoom.triggered.connect(self.setZoomReset) # Reset zoom on new image lad, file menu connection
    
    # Update list of loaded images
    def set_image_list(self, image_paths):
        self.list_widget.loaded_images_list.clear()
        self.image_paths = image_paths
        
        # Get names from paths
        self.image_names = []
        for path in image_paths:
            self.image_names.append(Utilities.get_file_name(path))

        if len(self.image_names) > 0:
            self.list_widget.loaded_images_list.addItems(self.image_names) # Add image names
        else:
            self.list_widget.loaded_images_list.addItems(self.list_widget.loaded_images_default) # Add default
    
    # Update image of QGraphicsView
    def setImage(self, item):
        try:
            index = self.image_names.index(item.text())     # Try getting image index
        except ValueError:
            return

        image = self.image_paths[index]
        if image:
            self.image_preview.setImage(qtg.QPixmap(image)) # Set image inside QGraphicsView
            self.toggle_actions("image_preview", True)      # Enable Image preview menu
    
    # Set zoom reset on new image load of ImageViewer
    def setZoomReset(self, value):
        self.image_preview.reset_zoom = value

# Loaded images list
class ImageListWidget(qtw.QWidget):
    loaded_images_default = ["Loaded images will appear here.", "Please load them in from the 'file' menu."]
    processed_images_default = ["Processed images will appear here.", "Please load images first,", "and process them using the 'processing' menu."]
    def __init__(self, parent):
        super().__init__(parent)

        # Loaded images list
        self.loaded_images_list = qtw.QListWidget()
        self.loaded_images_list.setAlternatingRowColors(True)
        self.loaded_images_list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.loaded_images_list.addItems(self.loaded_images_default)

        # Processed images list
        self.processed_images_list = qtw.QListWidget()
        self.processed_images_list.setAlternatingRowColors(True)
        self.processed_images_list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.processed_images_list.addItems(self.processed_images_default)

        # Vertical splitter in between lists
        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.setOrientation(qtc.Qt.Vertical)
        splitter.addWidget(self.loaded_images_list)
        splitter.addWidget(self.processed_images_list)
        screen_width = qtw.QDesktopWidget().availableGeometry(0).height()
        splitter.setSizes([screen_width, round(screen_width / 2)])

        layout = qtw.QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

class ImageViewer(qtw.QGraphicsView):
    photoClicked = qtc.pyqtSignal(qtc.QPoint)

    def __init__(self, parent):
        super().__init__(parent)
        self.current_zoom_level = 0
        self.reset_zoom = True
        self.image_loaded = True
        self._scene = qtw.QGraphicsScene(self)
        self._photo = qtw.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)

        self.setScene(self._scene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)

    def hasImage(self):
        return not self.image_loaded

    # Fit image to view
    def fitInView(self, scale=True):
        rect = qtc.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self.current_zoom_level = 0

    # Set image
    def setImage(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self.image_loaded = False
            self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self.image_loaded = True
            self.setDragMode(qtw.QGraphicsView.NoDrag)
            self._photo.setPixmap(qtg.QPixmap())
        
        # Reset zoom
        if self.reset_zoom == True:
            self.current_zoom_level = 0
            self.fitInView()

    def wheelEvent(self, event):
        if self.hasImage():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self.current_zoom_level += 1
            else:
                factor = 0.8
                self.current_zoom_level -= 1
            if self.current_zoom_level > 0:
                self.scale(factor, factor)
            elif self.current_zoom_level == 0:
                self.fitInView()
            else:
                self.current_zoom_level = 0

    def toggleDragMode(self):
        if self.dragMode() == qtw.QGraphicsView.ScrollHandDrag:
            self.setDragMode(qtw.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super().mousePressEvent(event)

# preferences class
class Preferences(qtw.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.Application = parent
        self.setWindowTitle("Preferences")
        self.setWindowModality(qtc.Qt.ApplicationModal)

        self.theme_setting = qtw.QComboBox()
        self.theme_setting.insertItems(0, ["Dark theme", "Light theme"])

        apply_button = qtw.QPushButton()
        apply_button.setText("Apply")
        apply_button.clicked.connect(self.apply_settings)

        layout = qtw.QGridLayout()
        layout.addWidget(self.theme_setting, 0, 0, 1, 2)
        layout.addWidget(apply_button, 1, 1)
        self.setLayout(layout)

        # Load saved settings
        self.settings = qtc.QSettings("PyStacker", "Preferences")
        #self.settings.remove("Preferences")
        self.load_settings()

    # Load saved settings
    def load_settings(self):
        # Load saved theme
        applied_theme = self.settings.value("Theme", "Dark theme")
        self.set_color_theme(applied_theme)

        index = self.theme_setting.findText(applied_theme)
        if index != -1:   # -1 is not found
            self.theme_setting.setCurrentIndex(index)

    # Apply all settings
    def apply_settings(self):
        # Set theme
        applied_theme = self.theme_setting.currentText()
        self.set_color_theme(applied_theme)
        self.settings.setValue("Theme", applied_theme)

        index = self.theme_setting.findText(applied_theme)
        if index != -1:   # -1 is not found
            self.theme_setting.setCurrentIndex(index)
    
    # Set a color theme
    def set_color_theme(self, theme_name):
        palette = qtg.QPalette()
        if theme_name == "Dark theme":
            dark_gray = qtg.QColor(53, 53, 53)
            gray = qtg.QColor(128, 128, 128)
            black = qtg.QColor(25, 25, 25)
            blue = qtg.QColor(42, 130, 218)

            palette.setColor(qtg.QPalette.Window, dark_gray)
            palette.setColor(qtg.QPalette.WindowText, qtc.Qt.white)
            palette.setColor(qtg.QPalette.Base, black)
            palette.setColor(qtg.QPalette.AlternateBase, dark_gray)
            palette.setColor(qtg.QPalette.ToolTipBase, blue)
            palette.setColor(qtg.QPalette.ToolTipText, qtc.Qt.white)
            palette.setColor(qtg.QPalette.Text, qtc.Qt.white)
            palette.setColor(qtg.QPalette.Button, dark_gray)
            palette.setColor(qtg.QPalette.ButtonText, qtc.Qt.white)
            palette.setColor(qtg.QPalette.Link, blue)
            palette.setColor(qtg.QPalette.Highlight, blue)
            palette.setColor(qtg.QPalette.HighlightedText, black)

            palette.setColor(qtg.QPalette.Active, qtg.QPalette.Button, gray.darker())
            palette.setColor(qtg.QPalette.Disabled, qtg.QPalette.ButtonText, gray)
            palette.setColor(qtg.QPalette.Disabled, qtg.QPalette.WindowText, gray)
            palette.setColor(qtg.QPalette.Disabled, qtg.QPalette.Text, gray)
            palette.setColor(qtg.QPalette.Disabled, qtg.QPalette.Light, dark_gray)
            self.Application.setPalette(palette)
        elif theme_name == "Light theme":
            self.Application.setPalette(self.Application.style().standardPalette())

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    
    # Fore fusion style
    app.setStyle("Fusion")

    sys.exit(app.exec_())