import napari
from napari.qt.threading import thread_worker
import numpy
import glob
from PIL import Image
import os


IMAGE_DIR = "HighResImages/*.jpg"

imageHeight = None
imageWidth = None

imgFileList = glob.glob(IMAGE_DIR)
for index, imPath in enumerate(imgFileList):
    if index == 0:
        image = Image.open(imPath)
        imageWidth, imageHeight = image.size

def temp_filenames(imPath):
    filename, file_extension = os.path.splitext(imPath)
    return filename + ".raw", filename + ".raw.grayscale", filename + ".raw.aligned", filename + ".raw.grayscale.laplacian"


with napari.gui_qt():
    viewer = napari.Viewer()

    def stackedImghandler(imgArray):
        viewer.add_image(imgArray, name="Stacked Image", rgb=True)

    def origImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Original Images")

    def grayscaleImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Grayscale Images")

    def allignedImagesHandler(imgArray):
        viewer.add_image(imgArray, name="Aligned Images")

    def laplacianGradientsHandler(imgArray):
        viewer.add_image(imgArray, name="Laplacian Gradients")

    @thread_worker(connect={"returned": stackedImghandler})
    def loadStackedImage():
        return numpy.memmap("stacked.raw", mode="r", shape=(imageHeight, imageWidth, 3))

    @thread_worker(connect={"returned": origImagesHandler})
    def load_source_images():
        images = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            raw_fn, _, _, _ = temp_filenames(imPath)
            images.append(numpy.memmap(raw_fn, mode="r", shape=(imageHeight, imageWidth, 3)))

        return numpy.asarray(images)

    @thread_worker(connect={"returned": grayscaleImagesHandler})
    def load_grayscale_images():
        grayscales = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            _, grayscale_fn, _, _ = temp_filenames(imPath)
            grayscales.append(numpy.memmap(grayscale_fn, mode="r", shape=(imageHeight, imageWidth)))

        return numpy.asarray(grayscales)

    @thread_worker(connect={"returned": allignedImagesHandler})
    def load_aligned_images():
        grayscales = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            _raw_fn, _, _raw_aligned, _ = temp_filenames(imPath)
            if os.path.isfile(_raw_aligned):
                grayscales.append(numpy.memmap(_raw_aligned, mode="r", shape=(imageHeight, imageWidth, 3)))
            else:
                grayscales.append(numpy.memmap(_raw_fn, mode="r", shape=(imageHeight, imageWidth, 3)))

        return numpy.asarray(grayscales)

    @thread_worker(connect={"returned": laplacianGradientsHandler})
    def load_laplacian_gradients():
        laplacians = []

        imgFileList = glob.glob(IMAGE_DIR)
        for imPath in sorted(imgFileList):
            _, _, _, _laplacian_fn = temp_filenames(imPath)
            laplacians.append(numpy.memmap(_laplacian_fn, mode="r", shape=(imageHeight, imageWidth)))

        return numpy.asarray(laplacians)

    # Run threaded functions
    #loadStackedImage()
    load_source_images()
    load_grayscale_images()
    load_aligned_images()
    load_laplacian_gradients()