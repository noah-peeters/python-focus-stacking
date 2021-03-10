import PyQt5.QtGui as qtg
from datetime import timedelta
import ntpath
import os
import numpy as np
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
            size = os .path.getsize(path)
            return size / 1e+6 # Bytes to MegaBytes
        return 0
    
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
                    qImage = qtg.QImage(array.data, width, height, bytes_per_line, qtg.QImage.Format_RGB888)
                else:
                    qImage = qimage2ndarray.array2qimage(array)
            elif array.dtype == "float64":
                # arr = array.astype(np.int64)
                # # pack the 16 bit values of arr into the red, green, and blue channels
                # rgb = arr << 48 | arr << 32 | arr << 16 | 0xffff
                # im = qtg.QImage(rgb, rgb.shape[0], rgb.shape[1], qtg.QImage.Format_RGBA64)
                # return qtg.QPixmap.fromImage(im)

                qImage = qimage2ndarray.array2qimage(array)
            if qImage:
                return qtg.QPixmap(qImage)