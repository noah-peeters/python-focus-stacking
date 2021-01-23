import napari
from napari.qt.threading import thread_worker
import numpy
import glob
from PIL import Image


IMAGE_DIR = "HighResImages/*.jpg"

imageHeight = None
imageWidth = None

imgFileList = glob.glob(IMAGE_DIR)
for index, imPath in enumerate(imgFileList):
    if index == 0:
        image = Image.open(imPath)
        imageWidth, imageHeight = image.size



with napari.gui_qt():
    viewer = napari.Viewer()

    def stackedImghandler(imgArray):
        viewer.add_image(imgArray, name="Stacked Image", rgb=True)
        print("Stacked image has been loaded.")

    def origImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Original Images")
        print("Original images have loaded.")

    def grayscaleImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Grayscale Images")
        print("Grayscale images have loaded.")

    @thread_worker(connect={"returned": stackedImghandler})
    def loadStackedImage():
        return numpy.memmap("stacked.img", mode="r", shape=(imageHeight, imageWidth, 3))

    @thread_worker(connect={"returned": origImagesHandler})
    def loadImages():
        images = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            images.append(numpy.memmap(imPath + ".img", mode="r", shape=(imageHeight, imageWidth, 3)))

        return numpy.asarray(images)

    @thread_worker(connect={"returned": grayscaleImagesHandler})
    def loadGrayscaleImages():
        grayscales = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            grayscales.append(numpy.memmap(imPath + ".grayscale", mode="r", shape=(imageHeight, imageWidth)))

        return numpy.asarray(grayscales)

    # Run threaded functions
    loadStackedImage()
    loadImages()
    loadGrayscaleImages()