import PyQt5.QtGui as qtg
from datetime import timedelta
import ntpath
import os

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
        height, width, _ = array.shape
        bytes_per_line = 3 * width
        qImage = qtg.QImage(array.data, width, height, bytes_per_line, qtg.QImage.Format_RGB888)
        return qtg.QPixmap(qImage)