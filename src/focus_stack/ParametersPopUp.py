"""
    Pop-ups for changing parameters before some processing actions (Aligning, Laplcian gradient calculation)
"""
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc


class AlignImagesPopUp(qtw.QDialog):
    def __init__(self, func_to_run):
        super().__init__()
        self.func_to_run = func_to_run
        self.setWindowTitle("Settings for image alignment")
        self.setWindowModality(qtc.Qt.ApplicationModal)

        # Warp mode combobox
        self.warp_mode_input = qtw.QComboBox()
        self.warp_mode_input.insertItems(
            0, ["Translation", "Affine", "Euclidean", "Homography"]
        )

        # Number of iterations slider
        self.number_of_iterations_input = Slider(100, 15000, 5000)

        # Termination epsilon slider
        self.termination_epsilon_input = Slider(2, 30, 8)

        # Gaussian blur size
        self.gaussian_blur_size_input = Slider(0, 20, 5)

        apply_button = qtw.QPushButton()
        apply_button.setText("Apply")
        apply_button.clicked.connect(self.applySettings)

        form_layout = qtw.QFormLayout()
        form_layout.addRow("Warp mode:", self.warp_mode_input)
        form_layout.addRow(
            "Number of iterations until termination:", self.number_of_iterations_input
        )
        form_layout.addRow("Termination epsilon:", self.termination_epsilon_input)
        form_layout.addRow("Gaussian blur size:", self.gaussian_blur_size_input)
        form_layout.addWidget(apply_button)

        self.setLayout(form_layout)

    def applySettings(self):
        val = self.gaussian_blur_size_input.slider.value()
        if val % 2 == 0 and val != 0:
            # Gaussian blur has even value --> show error message and retry
            qtw.QMessageBox.critical(
                self,
                "Gaussian blur error",
                "Gaussian blur size must be an odd number (e.g.: 1, 3, 5 etc.)",
                qtw.QMessageBox.Ok,
            )
        else:
            # Gaussian blur has uneven value --> proceed to image alignment
            self.func_to_run(
                {
                    "WarpMode": self.warp_mode_input.currentText(),
                    "NumberOfIterations": self.number_of_iterations_input.slider.value(),
                    "TerminationEpsilon": self.termination_epsilon_input.slider.value(),
                    "GaussianBlur": self.gaussian_blur_size_input.slider.value(),
                },
                self,
            )


class StackImagesPopUp(qtw.QDialog):
    chosen_stacking_algorithm = ""

    def __init__(self, func_to_run):
        super().__init__()
        self.func_to_run = func_to_run

        self.setWindowTitle("Settings for image stacking")
        self.setWindowModality(qtc.Qt.ApplicationModal)

        # Stacking mode selection box
        self.stacking_mode = qtw.QComboBox()
        self.stacking_mode.insertItems(
            0, ["Laplacian gradient, pixel-based fusion", "Laplacian pyramid"]
        )
        self.stacking_mode.setItemData(0, "laplacian")
        self.stacking_mode.setItemData(1, "pyramid")
        self.stacking_mode.currentIndexChanged.connect(
            lambda i: self.stacked_widget.setCurrentIndex(i)
        )
        # Apply settings button
        apply_button = qtw.QPushButton()
        apply_button.setText("Apply")
        apply_button.clicked.connect(self.applySettings)

        """
            Laplacian gradient stacking widget
        """
        self.laplacian_widget = qtw.QWidget()
        # Gaussian blur size
        self.gaussian_blur_size_laplacian = Slider(0, 40, 5)
        # laplacian kernel size
        self.laplacian_kernel_size_laplacian = Slider(1, 31, 3)

        layout1 = qtw.QFormLayout()
        layout1.addRow("Gaussian blur kernel size:", self.gaussian_blur_size_laplacian)
        layout1.addRow("Laplacian kernel size:", self.laplacian_kernel_size_laplacian)

        self.laplacian_widget.setLayout(layout1)

        """
            Laplacian pyramid stacking widget
        """
        self.pyramid_widget = qtw.QWidget()
        # Pyramid minimum size
        self.pyramid_min_size = Slider(0, 128, 32)

        layout2 = qtw.QFormLayout()
        layout2.addRow("Pyramid minimum size:", self.pyramid_min_size)

        self.pyramid_widget.setLayout(layout2)

        # Stacked widget
        self.stacked_widget = qtw.QStackedWidget()
        self.stacked_widget.addWidget(self.laplacian_widget)
        self.stacked_widget.addWidget(self.pyramid_widget)

        # Main layout
        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.stacking_mode)
        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(apply_button)

        self.setLayout(main_layout)

    def applySettings(self):
        data = self.stacking_mode.currentData()
        if data == "laplacian":
            gaussian_val = self.gaussian_blur_size_laplacian.slider.value()
            laplacian_val = self.laplacian_kernel_size_laplacian.slider.value()
            continue_bool = True
            if gaussian_val % 2 == 0 and gaussian_val != 0:
                continue_bool = False
                # Gaussian blur has even value --> show error message and retry
                qtw.QMessageBox.critical(
                    self,
                    "Gaussian blur error",
                    "Gaussian blur size must be an odd number (e.g.: 1, 3, 5 etc.)",
                    qtw.QMessageBox.Ok,
                )
            elif laplacian_val % 2 == 0:
                continue_bool = False
                # laplacian has even value --> show error message and retry
                qtw.QMessageBox.critical(
                    self,
                    "Laplacian kernel size error",
                    "Laplacian kernel size must be an odd number (e.g.: 1, 3, 5 etc.)",
                    qtw.QMessageBox.Ok,
                )

            if continue_bool:
                self.func_to_run(
                    "laplacian",
                    {
                        "GaussianBlur": gaussian_val,
                        "LaplacianKernel": laplacian_val,
                    },
                    self,
                )
        elif data == "pyramid":
            self.func_to_run(
                "pyramid",
                {"MinLaplacianSize": self.pyramid_min_size.slider.value()},
                self,
            )


class Slider(qtw.QWidget):
    def __init__(self, min_range, max_range, default_value):
        super().__init__()
        # Slider
        self.slider = qtw.QSlider()
        self.slider.setRange(min_range, max_range)
        self.slider.setValue(default_value)
        self.slider.setFocusPolicy(qtc.Qt.NoFocus)
        self.slider.setOrientation(qtc.Qt.Horizontal)
        self.slider.valueChanged.connect(self.updateSliderText)

        # Slider value text
        self.slider_text = qtw.QLabel(str(default_value))
        self.slider_text.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)

        slider_layout = qtw.QVBoxLayout()
        slider_layout.addWidget(self.slider_text)
        slider_layout.addWidget(self.slider)
        self.setLayout(slider_layout)

    def updateSliderText(self, value):
        self.slider_text.setText(str(value))
