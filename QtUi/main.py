import sys
import collections
from pathlib import Path
import time
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg

from utilities import Utilities
Utilities = Utilities()

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
        help_menu = menu_bar.addMenu("&Help")
        edit_menu = menu_bar.addMenu("&Edit")

        # Create actions
        new_action = qtw.QAction("&New file", self, shortcut="Ctrl+N", triggered=self.create_new_file)
        open_action = qtw.QAction("&Open file", self, shortcut="Ctrl+O", triggered=self.open_file)
        save_file = qtw.QAction("&Save file", self, shortcut="Ctrl+S", triggered=self.save_file)
        load_images_action = qtw.QAction("&Load images", self, shortcut="Ctrl+L", triggered=self.load_images)
        clear_loaded_images_action = qtw.QAction("&Clear loaded images", self, shortcut="Ctrl+Alt+C", triggered=self.clear_loaded_images)
        export_action = qtw.QAction("&Export", self, shortcut="Ctrl+E", triggered=self.export_image)
        quit_action = qtw.QAction("&Quit", self, shortcut="Ctrl+W", triggered=lambda: self.close())
        # Processing functions disabled by default (no images loaded)
        self.align_images_action = qtw.QAction("&Align images", self, shortcut="Ctrl+Shift+A", triggered=self.align_images, enabled=False)
        self.stack_images_action = qtw.QAction("&Stack Images", self, shortcut="Ctrl+Shift+S", triggered=self.stack_images, enabled=False)
        self.align_and_stack_images_action = qtw.QAction("&Align and stack images", self, shortcut="Ctrl+Shift+P", triggered=self.align_and_stack_images, enabled=False)

        about_app_action = qtw.QAction("&About PyStacker", self, triggered=self.about_application)
        about_qt_action = qtw.QAction("&About Qt", self, triggered=qtw.qApp.aboutQt)

        preferences_action = qtw.QAction("&Preferences", self, shortcut="Ctrl+P", triggered=self.open_preferences)

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

        about_app_action.setStatusTip("About this application.")
        about_qt_action.setStatusTip("About Qt, the framework that was used to design this ui.")

        preferences_action.setStatusTip("Preferences: themes and other settings.")

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

        help_menu.addAction(about_app_action)
        help_menu.addAction(about_qt_action)

        edit_menu.addAction(preferences_action)

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
            self.toggle_processing_actions(False)   # Disable processing actions
            return
        
        self.toggle_processing_actions(True)                # Enable processing actions
        self.current_directory = self.loaded_image_files[0] # Set current directory

        # Get total size of all images to import
        total_size = 0
        for image_path in self.loaded_image_files:
            file_size = Utilities.get_file_size_MB(image_path)
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
            image_progress.setLabelText("Just loaded: " + Utilities.get_file_name(image_path))

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

    def export_image(self):
        print("Export image")
    
    def align_images(self):
        print("Align images")

    def stack_images(self):
        print("Stack images")

    def align_and_stack_images(self):
        print("Align and stack images")

    def clear_loaded_images(self):
        # Display confirmation dialog
        reply = qtw.QMessageBox.question(self, "Clear loaded images", "Are you sure you want to clear all loaded images?", qtw.QMessageBox.Yes|qtw.QMessageBox.No)
        if reply == qtw.QMessageBox.Yes:
            # Clear (user confirmed)
            self.loaded_image_files = []
            self.main_layout.set_image_list(self.loaded_image_files)
            self.toggle_processing_actions(False)


    # Display result message after operation finished
    def result_message(self, returned_table, props):
        # "Unpack" values
        execution_time = returned_table["execution_time"]
        image_table = returned_table["image_table"]
        killed_by_user = returned_table["killed_by_user"]

        progress_bar = props["progress_bar"]
        success_message = props["success_message"]

        if progress_bar:
            progress_bar.close() # Hide progress bar

        # Are all loaded images inside of returned table?
        success = collections.Counter(image_table) == collections.Counter(self.loaded_image_files)
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

    def about_application(self):
        qtw.QMessageBox.about(self, "About PyStacker",
        "<b>PyStacker</b> is an application written in <b>Python</b> and <b>Qt</b> that is completely open-source: "
        "<a href='https://github.com/noah-peeters/python-focus-stacking'>PyStacker on github</a>"
        " and aims to provide an advanced focus stacking application for free."
        )

    # Enable or disable processing actions (only enable when images are loaded)
    def toggle_processing_actions(self, bool):
        self.align_images_action.setEnabled(bool)
        self.stack_images_action.setEnabled(bool)
        self.align_and_stack_images_action.setEnabled(bool)

    def open_preferences(self):
        self.Preferences.exec_()

# Main layout for MainWindow (splitter window)
class MainLayout(qtw.QWidget):
    image_names = []
    def __init__(self, parent):        
        super().__init__(parent)

        self.list_widget = LoadedImagesWidget(self)
        self.image_widget = ImageWidget(self)

        self.list_widget.image_list.itemClicked.connect(self.update_image)

        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.image_widget)
        screen_width = qtw.QDesktopWidget().availableGeometry(0).width()
        splitter.setSizes([round(screen_width / 5), screen_width])
        
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    # Update list of loaded images
    def set_image_list(self, image_paths):
        self.list_widget.image_list.clear()
        self.image_paths = image_paths

        # Get names from paths
        self.image_names = []
        for path in image_paths:
            self.image_names.append(Utilities.get_file_name(path))

        if len(self.image_names) > 0:
            self.list_widget.image_list.addItems(self.image_names)              # Add image names
        else:
            self.list_widget.image_list.addItems(self.list_widget.default_list) # Add default

    def update_image(self, item):
        # Try getting image index
        try:
            index = self.image_names.index(item.text())
        except ValueError:
            return

        image = self.image_paths[index]
        if image:
            self.image_widget.image.setPixmap(qtg.QPixmap(image))


# Loaded images list
class LoadedImagesWidget(qtw.QWidget):
    default_list = ["Loaded images will appear here.", "Please load them in from the 'file' menu."]
    def __init__(self, parent):
        super().__init__(parent)

        self.image_list = qtw.QListWidget()
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.image_list.addItems(self.default_list)

        layout = qtw.QVBoxLayout()
        layout.addWidget(self.image_list)
        self.setLayout(layout)


# Image preview 
class ImageWidget(qtw.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = qtw.QLabel("Please select an image to preview.", self)
        self.image.setFont(qtg.QFont("Times", 14))
        self.image.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.image.setAlignment(qtc.Qt.AlignCenter)

        layout = qtw.QHBoxLayout()
        layout.addWidget(self.image)
        self.setLayout(layout)

# preferences class
class Preferences(qtw.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.Application = parent

        self.setWindowTitle("Preferences")
        self.setWindowModality(qtc.Qt.ApplicationModal)
        self.theme_setting = qtw.QComboBox()

        apply_button = qtw.QPushButton()
        apply_button.setText("Apply")
        apply_button.clicked.connect(self.apply_settings)

        layout = qtw.QGridLayout()
        layout.addWidget(self.theme_setting, 0, 0, 1, 2)
        layout.addWidget(apply_button, 1, 1)
        self.setLayout(layout)

        # Setup settings
        self.settings = qtc.QSettings("PyStacker", "Preferences")
        self.load_settings()

    # Apply all settings
    def apply_settings(self):
        # Set theme
        applied_theme = self.theme_setting.currentText()
        self.set_color_theme(applied_theme)
        ls = []
        if applied_theme == "Dark theme":
            ls = ["Dark theme", "Light theme"]
        else:
            ls = ["Light theme", "Dark theme"]
        
        self.settings.setValue("Theme", ls)

        self.theme_setting.clear()
        self.theme_setting.insertItems(0, ls)
    
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

    # Load saved settings
    def load_settings(self):
        # Load saved theme
        theme = self.settings.value("Theme", ["Dark theme", "Light theme"])[0]
        self.set_color_theme(theme)
        ls = []
        if theme == "Dark theme":
            ls = ["Dark theme", "Light theme"]
        else:
            ls = ["Light theme", "Dark theme"]
        self.theme_setting.clear()
        self.theme_setting.insertItems(0, ls)

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    
    # Fore fusion style
    app.setStyle("Fusion")

    sys.exit(app.exec_())