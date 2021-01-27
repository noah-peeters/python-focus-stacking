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
        processes = []
        image_objects = []
        for i, img_fn in enumerate(image_files):
            if i == 0:
                # Don't align first image
                _image = IndividualImage(img_fn, None, self._gaussian_blur_kernel_size, self._laplacian_kernel_size)
            else:
                # Align next images to first image (image_files[0])
                _image = IndividualImage(img_fn, image_files[0], self._gaussian_blur_kernel_size, self._laplacian_kernel_size)

            processes.append(dask.delayed(_image.read_and_analyze)())
            image_objects.append(_image)

        # Load all images in parallel
        dask.compute(*processes)

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
        for image in image_objects:
            image.cleanup_temp_files() 
        return Image.fromarray(stacked_image)


class IndividualImage(object):
    def __init__(
        self, image_filename, first_image_path, laplacian_kernel_size: int = 5, gaussian_blur_kernel_size: int = 5) -> None:
        """Single Image Class, groups all processing
        """
        self._image_filename = image_filename
        self._laplacian_kernel_size = laplacian_kernel_size
        self._gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self._first_image_path = first_image_path


    def dimensions(self):
        return self._height, self._width

    def _temp_filenames(self, imPath):
        filename, file_extension = os.path.splitext(imPath)
        return filename + ".raw", filename + ".grayscale.raw"

    def read_and_analyze(self):
        _raw_fn, _grayscale_fn = self._temp_filenames(self._image_filename)
        # step 1: read image / prepare memmap 
        image = cv2.imread(self._image_filename)
        self._height, self._width, channels = image.shape
        memMappedImg = numpy.memmap(_raw_fn, mode="w+", shape=(self._height, self._width, 3)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedImg[:] = numpy.asarray(image)

        # step 2: align image to first one
        # if self._first_image_path != None:
        #     detector = cv2.ORB_create(100000)

        #     def _find_homography(_img1_key_points: numpy.ndarray, _image_2_kp: numpy.ndarray, _matches: List):
        #         image_1_points = numpy.zeros((len(_matches), 1, 2), dtype=numpy.float32)
        #         image_2_points = numpy.zeros((len(_matches), 1, 2), dtype=numpy.float32)

        #         for j in range(0, len(_matches)):
        #             image_1_points[j] = _img1_key_points[_matches[j].queryIdx].pt
        #             image_2_points[j] = _image_2_kp[_matches[j].trainIdx].pt

        #         homography, mask = cv2.findHomography(
        #             image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0
        #         )

        #         return homography

        #     first_image = numpy.memmap(_raw_fn, mode="r", shape=(self._height, self._width))
        #     img_i_key_points, image_i_desc = detector.detectAndCompute(memMappedImg, None)
        #     img1_key_points, image1_desc = detector.detectAndCompute(first_image, None)

        #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #     raw_matches = bf.match(image_i_desc, image1_desc)

        #     sort_matches = sorted(raw_matches, key=lambda x: x.distance)
        #     matches = sort_matches[0:128]

        #     homography_matrix = _find_homography(
        #         img_i_key_points, img1_key_points, matches
        #     )
        #     aligned_img = cv2.warpPerspective(
        #         memMappedImg,
        #         homography_matrix,
        #         (memMappedImg.shape[1], memMappedImg.shape[0]),
        #         flags=cv2.INTER_LINEAR,
        #     )
            
        #     memMappedImg[:] = numpy.asarray(aligned_img)
        #     image = numpy.asarray(aligned_img)

        # del memMappedImg

        # step 3: convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        memMappedGrayscale = numpy.memmap(_grayscale_fn, mode="w+", shape=(self._height, self._width)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedGrayscale[:] = numpy.asarray(grayscale)

        
        # step 4: on aligned grayscale: Gaussian Blur + Laplacian
        memMappedGrayscale = numpy.memmap(_grayscale_fn, mode="r+", shape=(self._height, self._width))
        blurredImg = cv2.GaussianBlur(memMappedGrayscale, (self._gaussian_blur_kernel_size, self._gaussian_blur_kernel_size), 0)
        laplacianGradient = cv2.Laplacian(blurredImg, -1, ksize=self._laplacian_kernel_size) # ddepth -1 for same as src image (cv2.CV_64F)
        memMappedGrayscale[:] = laplacianGradient   # Store processed (grayscale) image
        del memMappedGrayscale

    
    def read_temp_files_as_mmap(self):
        _raw_fn, _grayscale_fn = self._temp_filenames(self._image_filename)
        _img_raw = numpy.memmap(_raw_fn, mode="r", shape=(self._height, self._width, 3))
        _grayscale_raw = numpy.memmap(_grayscale_fn, mode="r", shape=(self._height, self._width))
        return _img_raw, _grayscale_raw 

    def cleanup_temp_files(self):
        _raw_fn, _grayscale_fn = self._temp_filenames(self._image_filename)
        os.remove(_raw_fn)
        os.remove(_grayscale_fn)
