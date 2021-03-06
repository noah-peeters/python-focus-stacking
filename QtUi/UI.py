import sys
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyStacker")
        self.setStatusBar(qtw.QStatusBar()) # Create status bar

        self.setup_file_menu()  # Setup file menu (top bar)

        self.setCentralWidget(MainLayout(self))
        self.showMaximized()

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

        # Create actions
        new_action = qtw.QAction("&New file", self)
        open_action = qtw.QAction("&Open file", self)
        save_file = qtw.QAction("&Save file", self)
        load_images_action = qtw.QAction("&Load images", self)
        export_action = qtw.QAction("&Export", self)
        quit_action = qtw.QAction("&Quit", self)

        align_images_action = qtw.QAction("&Align images", self)
        stack_images_action = qtw.QAction("&Stack Images", self)
        align_and_stack_images_action = qtw.QAction("&Align and stack images", self)

        # Setup keyboard shortcuts
        new_action.setShortcut("Ctrl+N")
        open_action.setShortcut("Ctrl+O")
        save_file.setShortcut("Ctrl+S")
        load_images_action.setShortcut("Ctrl+L")
        export_action.setShortcut("Ctrl+E")
        quit_action.setShortcut("Ctrl+W")

        align_images_action.setShortcut("Ctrl+Shift+A")
        stack_images_action.setShortcut("Ctrl+Shift+S")
        align_and_stack_images_action.setShortcut("Ctrl+Shift+P")

        # Setup help tips for actions
        new_action.setStatusTip("Create a new file. Unsaved progress will be lost!")
        open_action.setStatusTip("Open a file on disk. Unsaved progress will be lost!")
        save_file.setStatusTip("Save file to disk.")
        load_images_action.setStatusTip("Load images from disk.")
        export_action.setStatusTip("Export output image.")
        quit_action.setStatusTip("Exit the application. Unsaved progress will be lost!")

        align_images_action.setStatusTip("Align images.")
        stack_images_action.setStatusTip("Focus stack images.")
        align_and_stack_images_action.setStatusTip("Align images and focus stack them.")

        # Setup connections for actions
        new_action.triggered.connect(self.create_new_file)
        open_action.triggered.connect(self.open_file)
        save_file.triggered.connect(self.save_file)
        load_images_action.triggered.connect(self.load_images)
        export_action.triggered.connect(self.export_image)
        quit_action.triggered.connect(lambda: self.close())

        align_images_action.triggered.connect(self.align_images)
        stack_images_action.triggered.connect(self.stack_images)
        align_and_stack_images_action.triggered.connect(self.align_and_stack_images)

        # Icons for actions
        new_action.setIcon(get_icon("SP_FileIcon"))
        save_file.setIcon(get_icon("SP_DialogSaveButton"))
        open_action.setIcon(get_icon("SP_DialogOpenButton"))
        load_images_action.setIcon(get_icon("SP_FileDialogNewFolder"))
        export_action.setIcon(get_icon("SP_DialogYesButton"))
        quit_action.setIcon(get_icon("SP_TitleBarCloseButton"))

        align_images_action.setIcon(get_icon("SP_FileDialogContentsView"))
        stack_images_action.setIcon(get_icon("SP_TitleBarNormalButton"))
        align_and_stack_images_action.setIcon(get_icon("SP_BrowserReload"))

        # Add actions to menu
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_file)
        file_menu.addAction(load_images_action)
        file_menu.addAction(export_action)
        file_menu.addAction(quit_action)

        processing_menu.addAction(align_images_action)
        processing_menu.addAction(stack_images_action)
        processing_menu.addAction(align_and_stack_images_action)

    def create_new_file(self):
        print("Create new file")

    def open_file(self):
        print("Open file")

    def save_file(self):
        print("Save file")

    def load_images(self):
        print("Load images")

    def export_image(self):
        print("Export image")
    
    def align_images(self):
        print("Align images")

    def stack_images(self):
        print("Stack images")

    def align_and_stack_images(self):
        print("Align and stack images")

class MainLayout(qtw.QWidget):
    def __init__(self, parent):        
        super().__init__(parent)

        self.list_widget = qtw.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.list_widget.addItems(["Loaded images will appear here."])

        image_widget = ImageWidget(self)

        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(image_widget)
        splitter.setSizes([250, 1200])
        
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

class ImageWidget(qtw.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        image = qtw.QLabel("Please select an image to preview.", self)
        image.setFont(qtg.QFont("Times", 14))
        image.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        image.setStyleSheet("QLabel {background-color: rgb(200, 200, 200);}")
        image.setAlignment(qtc.Qt.AlignCenter)

        layout = qtw.QHBoxLayout()
        layout.addWidget(image)
        self.setLayout(layout)


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())