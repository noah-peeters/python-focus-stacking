"""
This code and algorithm was inspired and adapted from the following sources:
http://stackoverflow.com/questions/15911783/what-are-some-common-focus-stacking-algorithms
https://github.com/cmcguinness/focusstack

"""
from typing import List
import numpy
import dask
import glob
import time
import cv2
import os
from PIL import Image
import logging

DEBUG = False

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

class FocusStacker(object):
    def __init__(
        self, laplacian_kernel_size: int = 5, gaussian_blur_kernel_size: int = 5,
    ) -> None:
        """Focus stacking class.
        Args:
            laplacian_kernel_size: Size of the laplacian window. Must be odd.
            gaussian_blur_kernel_size: How big of a kernel to use for the gaussian blur. Must be odd.
        """
        self._laplacian_kernel_size = laplacian_kernel_size
        self._gaussian_blur_kernel_size = gaussian_blur_kernel_size

    def focus_stack(self, image_files: List[str]) -> numpy.ndarray:
        """Pipeline to focus stack a list of images."""

        stacked_image = self._process_images(image_files)
        return stacked_image

    def _process_images(self, image_files: List[str]) -> List[numpy.ndarray]:
        logger.info("concurrent processing images")
    #   processes = []
        image_objects = []
        for img_fn in image_files:
            _image = ImageToStack(img_fn, self._gaussian_blur_kernel_size, self._laplacian_kernel_size)
            _image.read_and_analyze()
            image_objects.append(_image)
        #    processes.append(dask.delayed(_image.read_and_analyze)(_image))
        # Load all images in parallel
        dask.compute()

        logger.info("stacking images")    
        # now do the final stacking
        images = []
        laplacianGradients = []
        for image in image_objects:
            raw_image, raw_grayscale = image.read_temp_files_as_mmap()
            images.append(raw_image)
            laplacianGradients.append(raw_grayscale)

        laplacianGradients = numpy.asarray(laplacianGradients)

        output = numpy.zeros_like(images[0])                    # Create new array filled with zeros
        absLaplacian = numpy.absolute(laplacianGradients)       # Get absolute values of Laplacian gradients
        maxLaplacian = absLaplacian.max(axis=0)                 # Get max value of Laplacian gradients
        boolMask = numpy.array(absLaplacian == maxLaplacian)    # Create bool mask (true or false values)
        mask = boolMask.astype(numpy.uint8)                     # Convert true/false into 1/0

        for i, img_fn in enumerate(images):
            output = cv2.bitwise_not(img_fn, output, mask=mask[i])
        

        outputImage = 255 - output

        stacked_image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB) #* PILLOW (PIL) is in RGB and cv2 is in BGR! Image must be converted from BGR to RGB.

        #cleanup temp files
        [image.cleanup_temp_files for image in image_objects]

        return Image.fromarray(stacked_image)


class ImageToStack(object):
    def __init__(
        self, image_filename, laplacian_kernel_size: int = 5, gaussian_blur_kernel_size: int = 5
    ) -> None:
        """Single Image Class, groups all processing
        """
        self._image_filename = image_filename
        self._laplacian_kernel_size = laplacian_kernel_size
        self._gaussian_blur_kernel_size = gaussian_blur_kernel_size


    def dimensions(self):
        return self._height, self._width

    def _temp_filenames(self):
        filename, file_extension = os.path.splitext(self._image_filename)
        return filename + ".raw", filename+".grayscale.raw"

    #@dask.delayed
    def read_and_analyze(self):
        _raw_fn, _grayscale_fn = self._temp_filenames()
        # step 1: read image / prepare memmap 
        image = cv2.imread(self._image_filename)
        self._height, self._width, channels = image.shape
        memMappedImg = numpy.memmap(_raw_fn, mode="w+", shape=(self._height, self._width, 3)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedImg[:] = numpy.asarray(image)
        del memMappedImg 

        # step 2: convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        memMappedGrayscale = numpy.memmap(_grayscale_fn, mode="w+", shape=(self._height, self._width)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedGrayscale[:] = numpy.asarray(grayscale) 

        # step 3: on grayscale: Gaussian Blur + Laplacian
        blurredImg = cv2.GaussianBlur(memMappedGrayscale, (self._gaussian_blur_kernel_size, self._gaussian_blur_kernel_size), 0)
        laplacianGradient = cv2.Laplacian(blurredImg, -1, ksize=3) # ddepth -1 for same as src image (cv2.CV_64F)
        memMappedGrayscale[:] = laplacianGradient   # Store processed (grayscale) image
        del memMappedGrayscale

    
    def read_temp_files_as_mmap(self):
        _raw_fn, _grayscale_fn = self._temp_filenames()
        _img_raw = numpy.memmap(_raw_fn, mode="r", shape=(self._height, self._width, 3))
        _grayscale_raw = numpy.memmap(_grayscale_fn, mode="r", shape=(self._height, self._width))
        return _img_raw, _grayscale_raw 

    def cleanup_temp_files(self):
         os.remove(self._temp_filenames)
