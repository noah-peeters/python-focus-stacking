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
        sorted_image_paths = sorted(image_files) # Sort paths by name

        self._load_images(sorted_image_paths)
        self._align_images(sorted_image_paths)
        self._gaussian_blur_and_laplacian_images(sorted_image_paths)
        stacked_image = self._stack_images(sorted_image_paths)
        return stacked_image

    def _load_images(self, image_files):
        logger.info("Reading images.")
        image_load = []
        for _, img_fn in enumerate(image_files):
            _image = ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size)
            image_load.append(dask.delayed(_image.load_image)(img_fn))

        # Load all images in parallel
        dask.compute(*image_load)

    def _align_images(self, image_files):
        logger.info("Aligning images.")
        #align_images = []
        for i, img_fn in enumerate(image_files):
            if i != 0:
                #_image = ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size)
                #align_images.append(dask.delayed(_image.align_image)(image_files[i - 1], img_fn))
                ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size).align_image(image_files[i - 1], img_fn)

        # Align all images in parallel
        #dask.compute(*align_images)
    
    def _gaussian_blur_and_laplacian_images(self, image_files):
        logger.info("Gaussian blurring images, and calculating their laplacian edges.")
        process_images = []
        for i, img_fn in enumerate(image_files):
            #_image = ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size)
            #process_images.append(dask.delayed(_image.gaussian_and_laplacian)(img_fn))
            ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size).gaussian_and_laplacian(img_fn)

        # Process all images in parallel
        #dask.compute(*process_images)

    def _stack_images(self, image_files):
        logger.info("Stacking images.")
        return ParallelCompute(self._gaussian_blur_kernel_size, self._laplacian_kernel_size).stack_images(image_files)

class ParallelCompute(object):
    def __init__(
        self, laplacian_kernel_size: int = 5, gaussian_blur_kernel_size: int = 5) -> None:
        """
            Single Image Class, groups all processing
        """
        self._laplacian_kernel_size = laplacian_kernel_size
        self._gaussian_blur_kernel_size = gaussian_blur_kernel_size

    def _temp_filenames(self, imPath):
        filename, file_extension = os.path.splitext(imPath)
        return filename + ".raw", filename + ".raw.grayscale", filename + ".raw.aligned", filename + ".raw.grayscale.laplacian"

    # Read image and save to disk (memmap)
    def load_image(self, img_path):
        _raw_fn, _grayscale_fn, _, _ = self._temp_filenames(img_path)
        # RGB image
        image = cv2.imread(img_path)
        
        # Set image size
        global IMAGE_HEIGHT, IMAGE_WIDTH
        IMAGE_HEIGHT, IMAGE_WIDTH, channels = image.shape

        memMappedImg = numpy.memmap(_raw_fn, mode="w+", shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedImg[:] = numpy.asarray(image)
        del memMappedImg

        # Grayscale image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        memMappedGrayscale = numpy.memmap(_grayscale_fn, mode="w+", shape=(IMAGE_HEIGHT, IMAGE_WIDTH)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedGrayscale[:] = numpy.asarray(grayscale)
        del memMappedGrayscale

    # Align an image to a source image
    def align_image(self, src_img_path, img_to_align_path):
        logger.info("Aligning " + img_to_align_path + " to: " + src_img_path)

        # Read previously aligned image
        _raw_fn, _, _raw_aligned, _ = self._temp_filenames(src_img_path)
        if os.path.isfile(_raw_aligned):
            img1 = numpy.memmap(_raw_aligned, mode="r", shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        else:
            img1 = numpy.memmap(_raw_fn, mode="r", shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        _raw_fn, _, _, _ = self._temp_filenames(img_to_align_path)
        im2 =  numpy.memmap(_raw_fn, mode="r+", shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = numpy.eye(3, 3, dtype=numpy.float32)
        else:
            warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        _, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography 
            aligned_image = cv2.warpPerspective(im2, warp_matrix, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            aligned_image = cv2.warpAffine(im2, warp_matrix, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Write aligned RGB image
        _, _, _raw_aligned, _ = self._temp_filenames(img_to_align_path)
        im2_aligned = numpy.memmap(_raw_aligned, mode="w+", shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        im2_aligned[:] = aligned_image

        del img1
        del im2
        del im2_aligned

    # Gaussian blur images and get their edges using a Laplacian gradient
    def gaussian_and_laplacian(self, img_path):
        _, _grayscale_fn, _, _laplacian_fn = self._temp_filenames(img_path)

        memMappedGrayscale = numpy.memmap(_grayscale_fn, mode="r+", shape=(IMAGE_HEIGHT, IMAGE_WIDTH))
        blurredImg = cv2.GaussianBlur(memMappedGrayscale, (self._gaussian_blur_kernel_size, self._gaussian_blur_kernel_size), 0)
        laplacianGradient = cv2.Laplacian(blurredImg, -1, ksize=self._laplacian_kernel_size) # ddepth -1 for same as src image (cv2.CV_64F)

        laplacianGradientMemmap = numpy.memmap(_laplacian_fn, mode="w+", shape=(IMAGE_HEIGHT, IMAGE_WIDTH))
        laplacianGradientMemmap[:] = laplacianGradient # Write to memmap

        del memMappedGrayscale
        del laplacianGradientMemmap


    # Stack images and cleanup temporary files  
    def stack_images(self, image_paths):
        # Get all RGB images and laplacian gradients
        rgb_images = []
        laplacian_gradients = []
        for i, img_fn in enumerate(image_paths):
            _raw_fn, _, _raw_aligned_fn, _laplacian_fn = self._temp_filenames(img_fn)
            
            if os.path.isfile(_raw_aligned_fn): # First image is never aligned
                rgb_images.append(numpy.memmap(_raw_aligned_fn, mode="r", shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
            else:
                rgb_images.append(numpy.memmap(_raw_fn, mode="r", shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
            laplacian_gradients.append(numpy.memmap(_laplacian_fn, mode="r", shape=(IMAGE_HEIGHT, IMAGE_WIDTH)))

        laplacian_gradients = numpy.asarray(laplacian_gradients)

        output = numpy.zeros_like(rgb_images[0])              # Create new array filled with zeros
        absLaplacian = numpy.absolute(laplacian_gradients)    # Get absolute values of Laplacian gradients
        maxLaplacian = absLaplacian.max(axis=0)               # Get max value of Laplacian gradients
        boolMask = numpy.array(absLaplacian == maxLaplacian)  # Create bool mask (true or false values)
        mask = boolMask.astype(numpy.uint8)                   # Convert true/false into 1/0

        for i, img in enumerate(rgb_images):
            output = cv2.bitwise_not(img, output, mask=mask[i])

        outputImage = 255 - output

        stacked_image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB) # PILLOW (PIL) is in RGB and cv2 is in BGR! Image must be converted from BGR to RGB.

        # Cleanup temp files
        for i, img_fn in enumerate(image_paths):
            _raw_fn, _grayscale_fn, _raw_aligned_fn, _laplacian_fn = self._temp_filenames(img_fn)
            os.remove(_raw_fn)
            os.remove(_grayscale_fn)
            os.remove(_raw_aligned_fn)
            os.remove(_laplacian_fn)

        return Image.fromarray(stacked_image)
