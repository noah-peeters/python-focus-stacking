import napari
from napari.qt.threading import thread_worker
import numpy
import glob

IMAGE_DIR = "images/*.jpg"

with napari.gui_qt():
    viewer = napari.Viewer()

    @thread_worker(connect={"returned": viewer.add_image})
    def loadStackedImage():
        return numpy.memmap("stacked.img", mode="r", shape=(4000, 6000))

    @thread_worker(connect={"returned": viewer.add_image})
    def loadImages():
        images = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            images.append(numpy.memmap(imPath + ".img", mode="r", shape=(4000, 6000, 3)))

        return numpy.asarray(images)

    @thread_worker(connect={"returned": viewer.add_image})
    def loadGrayscaleImages():
        grayscales = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            grayscales.append(numpy.memmap(imPath + ".grayscale", mode="r", shape=(4000, 6000)))

        return numpy.asarray(grayscales)

    # Run threaded functions
    loadStackedImage()
    loadImages()
    loadGrayscaleImages()