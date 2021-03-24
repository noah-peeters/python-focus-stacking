import psutil, os, ntpath
import PyQt5.QtGui as qtg
from datetime import timedelta
import qimage2ndarray


class Utilities:
    # Correctly format seconds to HH:MM:SS
    def format_seconds(self, s):
        return str(timedelta(seconds=s))

    # Get only filename from file's path
    def get_file_name(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    # Get file size in MB from path
    def get_file_size_MB(self, path):
        if os.path.exists(path):
            size = os.path.getsize(path)
            return size / 1e6  # Bytes to MegaBytes
        return 0

    # Return memory usage of python in MB
    def python_memory_usage(self):
        mem = psutil.Process(os.getpid()).memory_info().rss / float(2 ** 20)
        return mem

    # Convert numpy array image to QPixmap
    def numpyArrayToQPixMap(self, array):
        height = None
        width = None
        color_image = False
        if len(array.shape) == 3:
            height, width, _ = array.shape
            color_image = True
        elif len(array.shape) == 2:
            height, width = array.shape

        if height and width:
            bytes_per_line = 3 * width
            qImage = None
            if array.dtype == "uint8":
                if color_image:
                    # Color image (uint8)
                    qImage = qtg.QImage(
                        array.data,
                        width,
                        height,
                        bytes_per_line,
                        qtg.QImage.Format_RGB888,
                    )
                else:
                    qImage = qimage2ndarray.array2qimage(array)
            elif array.dtype == "float64":
                qImage = qimage2ndarray.array2qimage(array)

            if qImage:
                return qtg.QPixmap(qImage)
