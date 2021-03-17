import sys
import collections
from pathlib import Path
import os
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import random
import string
import logging
import ray
import psutil

# Initialize ray
ray.init(num_cpus=psutil.cpu_count(logical=False))

# Setup logging
class OneLineExceptionFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return repr(result)

    def format(self, record):
        result = super().format(record)
        if record.exc_text:
            result = result.replace("\n", "")
        return result


handler = logging.StreamHandler()
formatter = OneLineExceptionFormatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
root.addHandler(handler)

import src.focus_stack.QThreadWorkers as QThreads
import src.focus_stack.ParametersPopUp as ParametersPopUp

SUPPORTED_IMAGE_FORMATS = "(*.jpg *.png)"


class MainWindow(qtw.QMainWindow):
    current_directory = None

    def __init__(self):
        super().__init__()
        """
            Imports
        """
        from src.focus_stack.algorithm import ImageHandler

        self.ImageHandler = ImageHandler()

        from src.focus_stack.utilities import Utilities

        self.Utilities = Utilities()

        """
            Self setup
        """
        self.setWindowTitle("PyStacker")
        self.setStatusBar(qtw.QStatusBar())  # Create status bar
        self.setup_file_menu()  # Setup file menu (top bar)

        self.main_layout = MainLayout(self)  # Main layout setup
        self.setCentralWidget(self.main_layout)
        self.showMaximized()  # Show fullscreen

        self.Preferences = Preferences(self)  # Init Preferences

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
        new_action = qtw.QAction(
            "&New file", self, shortcut="Ctrl+N", triggered=self.create_new_file
        )
        open_action = qtw.QAction(
            "&Open file", self, shortcut="Ctrl+O", triggered=self.open_file
        )
        save_file = qtw.QAction(
            "&Save file", self, shortcut="Ctrl+S", triggered=self.save_file
        )
        load_images_action = qtw.QAction(
            "&Load images", self, shortcut="Ctrl+L", triggered=self.load_images
        )
        clear_loaded_images_action = qtw.QAction(
            "&Clear loaded images",
            self,
            shortcut="Ctrl+Alt+C",
            triggered=self.clear_loaded_images,
        )
        export_action = qtw.QAction(
            "&Export image", self, shortcut="Ctrl+E", triggered=self.exportImage
        )
        quit_action = qtw.QAction(
            "&Quit", self, shortcut="Ctrl+W", triggered=lambda: self.close()
        )

        self.align_images_action = qtw.QAction(
            "&Align images",
            self,
            shortcut="Ctrl+Shift+A",
            triggered=self.align_images,
            enabled=False,
        )  # Disabled by default
        self.stack_images_action = qtw.QAction(
            "&Stack Images",
            self,
            shortcut="Ctrl+Shift+S",
            triggered=self.stackImages_Laplacian,
            enabled=False,
        )
        self.align_and_stack_images_action = qtw.QAction(
            "&Align and stack images",
            self,
            shortcut="Ctrl+Shift+P",
            triggered=self.align_and_stack_images,
            enabled=False,
        )

        self.image_preview_reset_zoom = qtw.QAction(
            "&Reset zoom on preview", self, enabled=False, checkable=True, checked=True
        )

        preferences_action = qtw.QAction(
            "&Preferences",
            self,
            shortcut="Ctrl+P",
            triggered=lambda: self.Preferences.exec_(),
        )

        about_app_action = qtw.QAction(
            "&About PyStacker", self, triggered=self.about_application
        )
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
        self.align_and_stack_images_action.setStatusTip(
            "Align images and focus stack them."
        )

        self.image_preview_reset_zoom.setStatusTip("Reset zoom on new image select.")

        preferences_action.setStatusTip("Preferences: themes and other settings.")

        about_app_action.setStatusTip("About this application.")
        about_qt_action.setStatusTip(
            "About Qt, the framework that was used to design this ui."
        )

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
        self.loaded_image_files, _ = qtw.QFileDialog.getOpenFileNames(
            self,
            "Select images to load.",
            dir,
            "Image files " + SUPPORTED_IMAGE_FORMATS,
        )

        if (
            not self.loaded_image_files or len(self.loaded_image_files) <= 0
        ):  # No images have been selected!
            self.toggle_actions("processing", False)  # Disable processing actions
            return

        # Set absolute paths
        for val in self.loaded_image_files:
            val = os.path.abspath(val)

        self.toggle_actions("processing", True)  # Enable processing actions
        self.current_directory = self.loaded_image_files[0]  # Set current directory

        # Get total size of all images to import
        total_size = 0
        for image_path in self.loaded_image_files:
            file_size = self.Utilities.get_file_size_MB(image_path)
            if file_size:
                total_size += file_size
        total_size = round(total_size, 2)

        # load images and prompt loading screen
        image_progress = self.create_progress_bar()
        image_progress.setWindowTitle(
            "Loading "
            + str(len(self.loaded_image_files))
            + " images..., Total size: "
            + str(total_size)
            + " MB"
        )
        image_progress.setLabelText(
            "Preparing to load your images. This shouldn't take long. Please wait."
        )

        counter = 0

        def update_progress(image_path):
            nonlocal counter
            counter += 1
            # update label text
            image_progress.setLabelText(
                "Just loaded: " + self.Utilities.get_file_name(image_path)
            )

            # Update progress slider
            image_progress.setValue(counter)

        self.loading = QThreads.LoadImages(self.loaded_image_files, self.ImageHandler)
        self.loading.finishedImage.connect(update_progress)  # Update progress callback

        def finished_loading(returned):
            self.loaded_image_files = returned["image_table"]  # Set loaded images
            # Update listing of loaded images
            self.main_layout.set_image_list(
                self.loaded_image_files,
                self.main_layout.list_widget.loaded_images_list,
            )

            # Create pop-up on operation finish.
            props = {}
            props["progress_bar"] = image_progress
            # Success message
            props["success_message"] = qtw.QMessageBox(self)
            props["success_message"].setIcon(qtw.QMessageBox.Information)
            props["success_message"].setWindowTitle("Images loaded successfully!")
            props["success_message"].setText(
                str(len(self.loaded_image_files)) + " images have been loaded."
            )
            props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
            self.result_message(returned, props)  # Display message about operation

        self.loading.finished.connect(finished_loading)  # Connection on finished

        image_progress.canceled.connect(
            self.loading.kill
        )  # Stop image loading on cancel
        self.loading.start()

        image_progress.exec_()

    def align_images(self):
        def proceedToAlign(parameters, popup):
            if not parameters:
                return  # Some value was entered incorrectly, retry

            popup.close()  # Close parameters window
            # Progress bar
            align_progress = self.create_progress_bar()
            align_progress.setWindowTitle(
                "Aligning " + str(len(self.loaded_image_files)) + " images."
            )
            align_progress.setLabelText(
                "Preparing to align your images. This shouldn't take long. Please wait."
            )

            counter = 0

            def update_progress(imagename_tupple):
                nonlocal counter
                image0 = imagename_tupple[0]
                image1 = imagename_tupple[1]
                # update label text
                align_progress.setLabelText(
                    "Just aligned: "
                    + self.Utilities.get_file_name(image1)
                    + " to: "
                    + self.Utilities.get_file_name(image0)
                )
                counter += 1

                # Update progress slider
                align_progress.setValue(counter)

            self.aligning = QThreads.AlignImages(
                self.loaded_image_files, parameters, self.ImageHandler
            )
            self.aligning.finishedImage.connect(update_progress)

            def finished_loading(returned):
                # Add aligned images to processing list widget
                widget = self.main_layout.list_widget.aligned_images_list
                self.main_layout.set_image_list(returned["image_table"], widget)

                # Create pop-up on operation finish.
                props = {}
                props["progress_bar"] = align_progress
                # Success message
                props["success_message"] = qtw.QMessageBox(self)
                props["success_message"].setIcon(qtw.QMessageBox.Information)
                props["success_message"].setWindowTitle("Images aligned successfully!")
                props["success_message"].setText(
                    str(len(self.loaded_image_files)) + " images have been aligned."
                )
                props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
                self.result_message(returned, props)

            self.aligning.finished.connect(finished_loading)
            align_progress.canceled.connect(
                self.aligning.kill
            )  # Kill operation on "cancel" press

            self.aligning.start()
            align_progress.exec_()

        # Settings popup for image alignment
        popup = ParametersPopUp.AlignImagesPopUp(proceedToAlign)
        popup.exec_()

    def stackImages_Laplacian(self):
        def proceedToStacking(stacking_mode, parameters, popup):
            if not parameters:
                return  # Some value was entered incorrectly, retry

            popup.close()  # Close parameters window

            # Start laplacian stacking
            if stacking_mode == "laplacian":
                # Laplacians progress bar
                laplacian_progress = self.create_progress_bar()
                laplacian_progress.setWindowTitle(
                    "Calculating laplacians of "
                    + str(len(self.loaded_image_files))
                    + " images."
                )
                laplacian_progress.setLabelText(
                    "Preparing to calculate laplacian gradients of your images. This shouldn't take long. Please wait."
                )

                self.laplacian_calc = QThreads.CalculateLaplacians(
                    self.loaded_image_files, parameters, self.ImageHandler
                )

                counter = 0

                def laplacian_progress_update(image_path):
                    nonlocal counter
                    laplacian_progress.setLabelText(
                        "Just computed laplacian gradient for: "
                        + self.Utilities.get_file_name(image_path)
                    )
                    counter += 1
                    laplacian_progress.setValue(counter)

                def laplacian_finished(returned):
                    # Add gaussian blurred and laplacian images to processing list widget
                    blurred_widget = (
                        self.main_layout.list_widget.gaussian_blurred_images_list
                    )
                    laplacian_widget = (
                        self.main_layout.list_widget.laplacian_images_list
                    )
                    self.main_layout.set_image_list(
                        returned["image_table"], blurred_widget
                    )
                    self.main_layout.set_image_list(
                        returned["image_table"], laplacian_widget
                    )

                    props = {}
                    props["progress_bar"] = laplacian_progress
                    # Success message
                    props["success_message"] = qtw.QMessageBox(self)
                    props["success_message"].setIcon(qtw.QMessageBox.Information)
                    props["success_message"].setWindowTitle(
                        "Laplacian gradients calculation success!"
                    )
                    props["success_message"].setText(
                        "Laplacian gradients have been calculated for "
                        + str(len(self.loaded_image_files))
                        + " images."
                    )
                    props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
                    self.result_message(
                        returned, props
                    )  # Display message about operation

                    """
                        Start final stacking process
                    """
                    # Progress bar setup
                    stack_progress = qtw.QProgressDialog(
                        "Loading... ",
                        "Cancel",
                        0,
                        self.ImageHandler.getImageShape()[0],
                        self,
                    )
                    stack_progress.setWindowModality(qtc.Qt.WindowModal)
                    stack_progress.setValue(0)
                    stack_progress.setWindowTitle(
                        "Final stacking of image: "
                        + str(self.ImageHandler.getImageShape()[0])
                        + " rows tall, "
                        + str(self.ImageHandler.getImageShape()[1])
                        + " columns wide."
                    )
                    stack_progress.setLabelText(
                        "Preparing to calculate the final focus stack of your images. This shouldn't take long. Please wait."
                    )

                    self.final_stacking_thread = QThreads.FinalStacking_Laplacian(
                        self.loaded_image_files, self.ImageHandler
                    )

                    def row_progress_update(current_row):
                        stack_progress.setLabelText(
                            "Just calculated row "
                            + str(current_row)
                            + " of "
                            + str(self.ImageHandler.getImageShape()[0])
                            + "."
                        )
                        stack_progress.setValue(current_row)

                    def stack_finished(returned):
                        # Add stacked image to processing list widget
                        stacked_widget = self.main_layout.list_widget.stacked_image_list
                        self.main_layout.set_image_list(
                            returned["image_table"], stacked_widget
                        )

                        props = {}
                        props["progress_bar"] = stack_progress
                        # Success message
                        props["success_message"] = qtw.QMessageBox(self)
                        props["success_message"].setIcon(qtw.QMessageBox.Information)
                        props["success_message"].setWindowTitle(
                            "Final stack calculation success!"
                        )
                        props["success_message"].setText(
                            "The final stack has successfully been calculated."
                        )
                        props["success_message"].setStandardButtons(qtw.QMessageBox.Ok)
                        self.result_message(returned, props)

                    self.final_stacking_thread.row_finished.connect(row_progress_update)
                    self.final_stacking_thread.finished.connect(stack_finished)

                    self.final_stacking_thread.start()
                    stack_progress.exec_()

                self.laplacian_calc.imageFinished.connect(laplacian_progress_update)
                self.laplacian_calc.finished.connect(laplacian_finished)
                laplacian_progress.canceled.connect(self.laplacian_calc.kill)

                self.laplacian_calc.start()
                laplacian_progress.exec_()
            elif stacking_mode == "pyramid":
                self.pyramid_calc = QThreads.PyramidStacking(
                    self.loaded_image_files, parameters, self.ImageHandler
                )

                def pyramid_finished(returned):
                    # Add stacked image to processing list widget
                    stacked_widget = self.main_layout.list_widget.stacked_image_list
                    self.main_layout.set_image_list(
                        returned["image_table"], stacked_widget
                    )

                self.pyramid_calc.finished.connect(pyramid_finished)
                self.pyramid_calc.start()

        popup = ParametersPopUp.StackImagesPopUp(proceedToStacking)
        popup.exec_()

    def exportImage(self):
        dir = None
        if self.current_directory:
            dir = self.current_directory
        else:
            dir = str(Path.home)
        file_path, _ = qtw.QFileDialog.getSaveFileName(
            self, "Export stacked image", dir, SUPPORTED_IMAGE_FORMATS
        )

        if file_path:
            file_path = os.path.abspath(file_path)

            self.current_directory = file_path
            # Export to path
            success = self.ImageHandler.exportImage(file_path)

            if success:
                # Display success message
                qtw.QMessageBox.information(
                    self,
                    "Exported image successfully!",
                    "Output image was successfully exported.",
                    qtw.QMessageBox.Ok,
                )
            else:
                # Display error message
                qtw.QMessageBox.critical(
                    self,
                    "Image export failed!",
                    "Failed to export output image. Have you completed the stacking process?",
                    qtw.QMessageBox.Ok,
                )

    def align_and_stack_images(self):
        self.align_images()
        self.stackImages_Laplacian()

    def clear_loaded_images(self):
        # Display confirmation dialog
        reply = qtw.QMessageBox.question(
            self,
            "Clear all loaded images?",
            "Are you sure you want to clear <b>all</b> loaded images?",
            qtw.QMessageBox.Yes | qtw.QMessageBox.No,
        )
        if reply == qtw.QMessageBox.Yes:
            # Clear (user confirmed)
            self.loaded_image_files = []
            self.main_layout.set_image_list(
                self.loaded_image_files
            )  # Clear all images lists

            self.toggle_actions(
                "processing", False
            )  # Toggle image processing actions off (no images loaded)
            self.main_layout.image_preview.setImage(None)  # Remove image from preview
            self.ImageHandler.image_storage = {}  # Clear all image memmaps

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

        progress_bar = None
        if "progress_bar" in props:
            progress_bar = props["progress_bar"]

        success_message = props["success_message"]

        if progress_bar:
            progress_bar.close()  # Hide progress bar

        # Get operation success
        success = None
        if image_table and not operation_success:
            # Are all loaded images inside of returned table?
            success = collections.Counter(image_table) == collections.Counter(
                self.loaded_image_files
            )
        elif operation_success:
            success = operation_success

        if success and not killed_by_user:  # Display success message
            # Get names of files
            file_names = []
            for image_path in self.loaded_image_files:
                file_names.append(self.Utilities.get_file_name(image_path) + ", ")

            success_message.setDetailedText(
                "".join(str(file) for file in file_names)
            )  # Display names inside "Detailed text"
            success_message.setInformativeText(
                "Execution time: " + self.Utilities.format_seconds(execution_time)
            )  # Display execution time
            success_message.exec_()

        elif not killed_by_user:  # Display error message (error occured)
            error_message = qtw.QMessageBox(self)
            error_message.setIcon(qtw.QMessageBox.Critical)
            error_message.setWindowTitle("Something went wrong!")
            error_message.setText("Operation failed. Please retry.")
            error_message.exec_()

        else:  # User has stopped process. Show confirmation
            user_killed_message = qtw.QMessageBox(self)
            user_killed_message.setIcon(qtw.QMessageBox.Information)
            user_killed_message.setWindowTitle("Operation canceled by user.")
            user_killed_message.setText("Operation successfully canceled by user.")
            user_killed_message.exec_()

    # Progress bar shorthand
    def create_progress_bar(self):
        progress = None
        progress = qtw.QProgressDialog(
            "Loading...", "Cancel", 0, len(self.loaded_image_files), self
        )
        progress.setWindowModality(qtc.Qt.WindowModal)
        progress.setValue(0)
        return progress

    def about_application(self):
        qtw.QMessageBox.about(
            self,
            "About PyStacker",
            "<b>PyStacker</b> is an application written in <b>Python</b> and <b>Qt</b> that is completely open-source: "
            "<a href='https://github.com/noah-peeters/python-focus-stacking'>PyStacker on github</a>"
            " and aims to provide an advanced focus stacking application for free.",
        )

    # Enable or disable processing actions (only enable when images are loaded)
    def toggle_actions(self, menu_name, bool):
        if menu_name == "processing":
            self.align_images_action.setEnabled(bool)
            self.stack_images_action.setEnabled(bool)
            self.align_and_stack_images_action.setEnabled(bool)
        elif menu_name == "image_preview":
            self.image_preview_reset_zoom.setEnabled(bool)


# Main layout for MainWindow (splitter window)
class MainLayout(qtw.QWidget):
    image_paths = []
    downscaling_list = {}
    image_scale_factor = 1
    scale_step = 0.25
    image_scale_factor_range = [1, 5]

    def __init__(self, parent):
        self.Parent = parent
        self.Utilities = parent.Utilities

        super().__init__()

        self.list_widget = ImageListWidget(self)
        self.image_preview = ImageViewer(self)

        self.parent_status_bar = parent.statusBar()
        self.toggle_actions = parent.toggle_actions

        # List widget clicked connections
        self.list_widget.loaded_images_list.itemClicked.connect(
            lambda item: self.setLoadedImage(item, "rgb_source")
        )  # Loaded images --> display RGB preview
        self.list_widget.aligned_images_list.itemClicked.connect(
            lambda item: self.setLoadedImage(item, "rgb_aligned")
        )  # Aligned images --> display RGB preview
        self.list_widget.gaussian_blurred_images_list.itemClicked.connect(
            lambda item: self.setLoadedImage(item, "grayscale_gaussian")
        )  # Gaussian blurred images --> display RGB preview
        self.list_widget.laplacian_images_list.itemClicked.connect(
            lambda item: self.setLoadedImage(item, "grayscale_laplacian")
        )  # Laplacian images --> display grayscale preview (float 64)
        self.list_widget.stacked_image_list.itemClicked.connect(
            lambda item: self.setLoadedImage(item, "stacked")
        )  # Stacked image --> display RGB preview

        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.image_preview)

        screen_width = qtw.QDesktopWidget().availableGeometry(0).width()
        splitter.setSizes([round(screen_width / 5), screen_width])

        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

        parent.image_preview_reset_zoom.triggered.connect(
            self.setZoomReset
        )  # Reset zoom on new image lad, file menu connection

    # Update list of loaded images
    def set_image_list(self, image_paths, widget=None):
        self.image_paths = image_paths

        if len(self.image_paths) > 0 and widget:  # Images loaded, display their names.
            widget.clear()
            items = {}
            # Display list (without icons)
            for path in image_paths:
                item = qtw.QListWidgetItem()
                item.setText(self.Utilities.get_file_name(path))  # Display name
                item.setData(qtc.Qt.UserRole, path)  # Add full path to data (hidden)
                widget.addItem(item)  # Add item to list
                items[path] = item
        else:
            # Loaded images have been removed --> clear all lists (add default info)
            widget = self.list_widget

            # Clear all
            widget.loaded_images_list.clear()
            widget.aligned_images_list.clear()
            widget.gaussian_blurred_images_list.clear()
            widget.laplacian_images_list.clear()
            widget.stacked_image_list.clear()

            # Add default info to every list
            widget.loaded_images_list.addItems(widget.loaded_images_default)
            widget.aligned_images_list.addItems(widget.aligned_images_list_default)
            widget.gaussian_blurred_images_list.addItems(
                widget.gaussian_blurred_images_list_default
            )
            widget.laplacian_images_list.addItems(widget.laplacian_images_list_default)
            widget.stacked_image_list.addItems(widget.stacked_image_list_default)

    # Update image of QGraphicsView
    def setLoadedImage(self, item, im_type):
        if len(self.image_paths) > 0:  # Check if images have been loaded
            np_array = self.Parent.ImageHandler.getImageFromPath(
                item.data(qtc.Qt.UserRole), im_type
            )

            if np_array.any():
                qPixMap = self.Utilities.numpyArrayToQPixMap(np_array)
                if qPixMap:
                    # Set image inside QGraphicsView
                    self.image_preview.setImage(qPixMap)
                    # Enable Image preview menu
                    self.toggle_actions("image_preview", True)

    # Set zoom reset on new image load of ImageViewer
    def setZoomReset(self, value):
        self.image_preview.reset_zoom = value


# Loaded images list
class ImageListWidget(qtw.QWidget):
    loaded_images_default = [
        "Loaded images will appear here.",
        "Please load them in from the 'file' menu.",
    ]
    aligned_images_list_default = [
        "Aligned images will appear here.",
        "Please load your images first,",
        "and align them from the 'processing' menu.",
    ]
    gaussian_blurred_images_list_default = [
        "Gaussian blurred images will apear here.",
        "They are used to decrease noise, ",
        "and improve laplacian gradient quality.",
    ]
    laplacian_images_list_default = [
        "Laplacian gradients (edge detection),",
        "of your images will appear here.",
        "Please load images first,",
        "and calulate their edges from the 'processing' menu.",
    ]
    stacked_image_list_default = [
        "The final stacked image will appear here.",
        "Please stack your images from the 'processing' menu.",
    ]

    def __init__(self, parent):
        super().__init__()

        # Loaded images list
        self.loaded_images_list = qtw.QListWidget()
        self.loaded_images_list.setAlternatingRowColors(True)
        self.loaded_images_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.loaded_images_list.addItems(self.loaded_images_default)

        """
            Processed images lists
        """
        # Aligned images
        self.aligned_images_list = qtw.QListWidget()
        self.aligned_images_list.setAlternatingRowColors(True)
        self.aligned_images_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.aligned_images_list.addItems(self.aligned_images_list_default)
        # Gaussian blurred images
        self.gaussian_blurred_images_list = qtw.QListWidget()
        self.gaussian_blurred_images_list.setAlternatingRowColors(True)
        self.gaussian_blurred_images_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.gaussian_blurred_images_list.addItems(
            self.gaussian_blurred_images_list_default
        )
        # Laplacian images
        self.laplacian_images_list = qtw.QListWidget()
        self.laplacian_images_list.setAlternatingRowColors(True)
        self.laplacian_images_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.laplacian_images_list.addItems(self.laplacian_images_list_default)
        # Stacked image
        self.stacked_image_list = qtw.QListWidget()
        self.stacked_image_list.setAlternatingRowColors(True)
        self.stacked_image_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.stacked_image_list.addItems(self.stacked_image_list_default)
        """
            Tab widget (For sorting processing output images)
        """
        self.processing_lists_tab = qtw.QTabWidget()
        self.processing_lists_tab.addTab(self.aligned_images_list, "Aligned images")
        self.processing_lists_tab.addTab(
            self.gaussian_blurred_images_list, "Gaussian blurred images"
        )
        self.processing_lists_tab.addTab(self.laplacian_images_list, "Laplacian images")
        self.processing_lists_tab.addTab(self.stacked_image_list, "Stacked image")

        # Vertical splitter in between lists
        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.setOrientation(qtc.Qt.Vertical)
        splitter.addWidget(self.loaded_images_list)
        splitter.addWidget(self.processing_lists_tab)
        screen_width = qtw.QDesktopWidget().availableGeometry(0).height()
        splitter.setSizes([screen_width, round(screen_width / 2)])

        layout = qtw.QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)


# Image viewer usinng QGraphicsView
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
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
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
        if self.reset_zoom:
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
        self.load_settings()

    # Load saved settings
    def load_settings(self):
        # Load saved theme
        applied_theme = self.settings.value("Theme", "Dark theme")
        self.set_color_theme(applied_theme)

        index = self.theme_setting.findText(applied_theme)
        if index != -1:  # -1 is not found
            self.theme_setting.setCurrentIndex(index)

    # Apply all settings
    def apply_settings(self):
        # Set theme
        applied_theme = self.theme_setting.currentText()
        self.set_color_theme(applied_theme)
        self.settings.setValue("Theme", applied_theme)

        index = self.theme_setting.findText(applied_theme)
        if index != -1:  # -1 is not found
            self.theme_setting.setCurrentIndex(index)

        self.close()  # Close preferences window

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


def main():
    app = qtw.QApplication(sys.argv)
    # window = MainWindow()
    MainWindow()

    # Force fusion style
    app.setStyle("Fusion")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
