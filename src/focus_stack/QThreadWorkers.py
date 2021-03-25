"""
   Classes that manage long-running actions on a separate (Q)Thread.
   To prevent UI freezing.
"""

import time, collections
import PyQt5.QtCore as qtc
import ray

from src.focus_stack.utilities import Utilities

# Load images on separate thread
class LoadImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(str)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, image_handler):
        super().__init__()

        self.images = files
        self.is_killed = False

        # Initialize ImageHandler
        self.ImageHandler = image_handler

    def run(self):
        start_time = time.time()

        def update_func(path):
            self.finishedImage.emit(path)

        loaded_images = self.ImageHandler.loadImages(self.images, update_func)

        loaded_images.sort()  # Sort list (parallel operation can shuffle list)

        self.finished.emit(
            {
                "execution_time": round(time.time() - start_time, 4),
                "image_table": loaded_images,
                "killed_by_user": self.is_killed,
            }
        )

    def kill(self):
        self.is_killed = True


# Align images on separate thread
class AlignImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(list)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, parameters, image_handler):
        super().__init__()

        self.files = files
        self.parameters = parameters
        self.ImageHandler = image_handler
        self.is_killed = False

    def run(self):
        start_time = time.time()
        image0 = self.files[round(len(self.files) / 2)]  # Get middle image

        def update_func(path):
            self.finishedImage.emit([path, image0])

        self.parameters["image0"] = image0
        aligned_images = self.ImageHandler.alignImages(
            self.files, self.parameters, update_func
        )
        aligned_images.sort()  # Sort list (parallel operation can shuffle list)

        # Operation ended
        self.finished.emit(
            {
                "execution_time": round(time.time() - start_time, 4),
                "image_table": aligned_images,
                "killed_by_user": self.is_killed,
            }
        )

    def kill(self):
        self.is_killed = True


# Gaussian blur images and calculate their laplacian gradients on a separate thread
class CalculateLaplacians(qtc.QThread):
    imageFinished = qtc.pyqtSignal(str)
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, parameters, algorithm):
        super().__init__()

        self.start_time = 0
        self.laplacian_success = False
        self.is_killed = False

        self.files = files
        self.parameters = parameters

        # Initialize algorithm
        self.Algorithm = algorithm

    def run(self):
        self.start_time = time.time()
        """
            Compute laplacian edges
        """
        laplacian_images = []  # Table for processed laplacians
        for image_path in self.files:
            success = self.Algorithm.LaplacianPixelAlgorithm.computeLaplacianEdges(
                image_path, self.parameters
            )

            if not success or self.is_killed:  # Operation failed or stopped from UI
                break

            laplacian_images.append(image_path)  # Append aligned image
            # Send progress back to UI
            self.imageFinished.emit(image_path)

        if collections.Counter(laplacian_images) == collections.Counter(
            self.files
        ):  # All laplacian images computed?
            self.laplacian_success = True  # Laplacian edges computation success

        # Operation ended
        self.finished.emit(
            {
                "image_table": laplacian_images,
                "execution_time": round(time.time() - self.start_time, 4),
                "operation_success": self.laplacian_success,
                "killed_by_user": self.is_killed,
            }
        )

    # Kill operation
    def kill(self):
        self.is_killed = True


# Do final stacking on a separate thread
class FinalStacking_Laplacian(qtc.QThread):
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
        for current_row in self.Algorithm.LaplacianPixelAlgorithm.stackImages(
            self.files
        ):
            if (
                type(current_row) != int or self.is_killed
            ):  # Operation failed or stopped from UI
                break

            self.row_finished.emit(current_row)
            row_reference = current_row

        if (
            row_reference == self.Algorithm.getImageShape()[0]
        ):  # have all rows been processed?
            self.stack_success = True

        # Operation ended
        self.finished.emit(
            {
                "image_table": [self.Algorithm.stacked_image_temp_file.name],
                "execution_time": round(time.time() - self.start_time, 4),
                "operation_success": self.stack_success,
                "killed_by_user": self.is_killed,
            }
        )

    def kill(self):
        self.is_killed = True


class PyramidStacking(qtc.QThread):
    finished = qtc.pyqtSignal(dict)

    def __init__(self, files, parameters, algorithm):
        super().__init__()
        self.files = files
        self.Parameters = parameters
        self.Algorithm = algorithm
        self.is_killed = False
        self.stack_success = False
        self.start_time = 0

    def run(self):
        """
        Stack images using gaussian + laplacian pyramids.
        """
        self.start_time = time.time()
        stacked_file_name = self.Algorithm.PyramidAlgorithm.fusePyramid(
            self.files, self.Parameters
        )

        # Operation ended
        self.finished.emit(
            {
                "image_table": [stacked_file_name],
                "execution_time": round(time.time() - self.start_time, 4),
                "operation_success": self.stack_success,
                "killed_by_user": self.is_killed,
            }
        )

    def kill(self):
        self.is_killed = True


# (Down)scale images on a separate thread.
class ScaleImages(qtc.QThread):
    finishedImage = qtc.pyqtSignal(list)
    finished = qtc.pyqtSignal(bool)

    def __init__(self, image_paths, scale_factor, image_handler, im_type):
        super().__init__()
        self.image_paths = image_paths
        self.scale_factor = scale_factor
        self.ImageHandler = image_handler
        self.im_type = im_type

        self.Utilities = Utilities()

    def run(self):
        for path in self.image_paths:
            # Get image memmap
            np_array = self.ImageHandler.getImageFromPath(path, self.im_type)

            if np_array.any():
                # Downscale image to scale_factor (percentage) of original
                scaled = ray.get(
                    self.ImageHandler.downscaleImage.remote(np_array, self.scale_factor)
                )
                # Convert image to QPixmap and send to UI
                self.finishedImage.emit(
                    [path, self.Utilities.numpyArrayToQPixMap(scaled)]
                )
        self.finished.emit(True)
