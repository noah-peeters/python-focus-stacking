import napari
from napari.qt.threading import thread_worker
import numpy
import glob
import os, os.path


IMAGE_DIR = "test/*.jpg"


with napari.gui_qt():
    viewer = napari.Viewer()

    def stackedImghandler(imgArray):
        viewer.add_image(imgArray, name="Stacked Image", rgb=True)
        print("Stacked image has been loaded.")

    def maskImghandler(imgArray):
        viewer.add_image(imgArray, name="Mask")
        print("Mask has been loaded.")

    def origImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Original Images")
        print("Original images have loaded.")

    def grayscaleImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Grayscale Images")
        print("Grayscale images have loaded.")

    @thread_worker(connect={"returned": stackedImghandler})
    def loadStackedImage():
        return numpy.memmap("stacked.img", mode="r", shape=(4000, 6000, 3))
    
    @thread_worker(connect={"returned": maskImghandler})
    def loadMask():
        return numpy.memmap("mask.img", mode="r", shape=(54, 4000, 6000)) # First dim is number of images in stack

    @thread_worker(connect={"returned": origImagesHandler})
    def loadImages():
        images = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            images.append(numpy.memmap(imPath + ".img", mode="r", shape=(4000, 6000, 3)))

        return numpy.asarray(images)

    @thread_worker(connect={"returned": grayscaleImagesHandler})
    def loadGrayscaleImages():
        grayscales = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            grayscales.append(numpy.memmap(imPath + ".grayscale", mode="r", shape=(4000, 6000)))

        return numpy.asarray(grayscales)

    # Run threaded functions
    loadStackedImage()
    loadMask()
    loadImages()
    loadGrayscaleImages()