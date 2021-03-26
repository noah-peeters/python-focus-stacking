"""
Module containing helper functions that execute small tasks.
These can be parallellized with Ray.
"""

import tempfile, logging, psutil, os
import numpy as np
import cv2
import ray
from scipy import ndimage
from guppy import hpy

# Setup logging
log = logging.getLogger(__name__)

# Load a single image
@ray.remote
def loadImage(image_path, dir):
    """
    Load an image in RGB, convert to grayscale and get its shape.
    Write these operations inside of a memmap (tempfile) and return the tempfile path.
    """
    # Load in memory using cv2
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_shape = image_rgb.shape

    # Write to memmaps
    _, rgb_name = tempfile.mkstemp(suffix=".rgb_source", dir=dir)
    rgb_memmap = np.memmap(
        rgb_name,
        mode="w+",
        shape=image_shape,
        dtype=image_rgb.dtype,
    )
    rgb_memmap[:] = image_rgb

    _, grayscale_name = tempfile.mkstemp(suffix=".grayscale_source", dir=dir)
    grayscale_memmap = np.memmap(
        grayscale_name,
        mode="w+",
        shape=(image_shape[0], image_shape[1]),
        dtype=image_grayscale.dtype,
    )
    grayscale_memmap[:] = image_grayscale

    print("Loaded image: " + image_path)

    # Return info
    return [image_path, image_shape, rgb_name, grayscale_name]


# Align a single image
@ray.remote
def alignImage(im1_path, parameters, image_storage, dir):
    im0_path = parameters["image0"]
    # Checks
    if not im1_path in image_storage:
        return
    elif not im0_path in image_storage:
        return
    elif not "grayscale_source" in image_storage[im1_path]:
        return
    elif not "grayscale_source" in image_storage[im0_path]:
        return
    elif not "rgb_source" in image_storage[im0_path]:
        return
    elif not "image_shape" in image_storage[im0_path]:
        return

    # Get motion model from parameters
    mode = parameters["WarpMode"]
    if mode == "Translation":
        warp_mode = cv2.MOTION_TRANSLATION
    elif mode == "Affine":
        warp_mode = cv2.MOTION_AFFINE
    elif mode == "Euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif mode == "Homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY

    print("ONE: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # Define 2x3 or 3x3 warp matrix (1's on diagonal, 0's everywhere else)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    print("TWO: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # Get number of iterations
    number_of_iterations = 5000
    if parameters["NumberOfIterations"]:
        number_of_iterations = parameters["NumberOfIterations"]

    # Get termination epsilon
    termination_eps = 1e-8
    if parameters["TerminationEpsilon"]:
        termination_eps = 1 * 10 ** (-parameters["TerminationEpsilon"])

    # Define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    # Get gaussian blur size
    gaussian_blur_size = 5
    if parameters["GaussianBlur"]:
        gaussian_blur_size = parameters["GaussianBlur"]

    print("THREE: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    """
    Find transformation
    """
    # Get grayscale and RGB memmaps of sources
    t = image_storage[im0_path]
    shape = t["image_shape"]
    gray_memmap1 = np.memmap(
        t["grayscale_source"],
        mode="r",
        shape=(shape[0], shape[1]),
    )
    t = image_storage[im1_path]
    shape = t["image_shape"]
    gray_memmap2 = np.memmap(
        t["grayscale_source"],
        mode="r",
        shape=(shape[0], shape[1]),
    )
    rgb_memmap2 = np.memmap(
        t["rgb_source"],
        mode="r",
        shape=shape,
    )

    print("FOUR: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # Tempfile for warp_matrix
    _, temp_name = tempfile.mkstemp(suffix=".warp_matrix", dir=dir)
    warp_matrix_memmap = np.memmap(
        temp_name,
        mode="w+",
        shape=warp_matrix.shape,
        dtype=warp_matrix.dtype,
    )

    # Calculate warp_matrix using grayscale memmaps
    _, warp_matrix_memmap[:] = cv2.findTransformECC(
        gray_memmap1,
        gray_memmap2,
        warp_matrix,
        warp_mode,
        criteria,
        None,
        gaussian_blur_size,
    )

    print("FIVE: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # Create memaps for output images
    _, rgb_aligned_name = tempfile.mkstemp(suffix=".rgb_aligned", dir=dir)
    rgb_aligned = np.memmap(
        rgb_aligned_name,
        mode="w+",
        shape=rgb_memmap2.shape,
        dtype=rgb_memmap2.dtype,
    )
    _, grayscale_aligned_name = tempfile.mkstemp(suffix=".grayscale_aligned", dir=dir)
    grayscale_aligned = np.memmap(
        grayscale_aligned_name,
        mode="w+",
        shape=gray_memmap2.shape,
        dtype=gray_memmap2.dtype,
    )

    print("SIX: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # Calculate transformed images and write to memmaps
    shape = (rgb_memmap2.shape[1], rgb_memmap2.shape[0])
    if warp_mode == cv2.MOTION_HOMOGRAPHY:  # Use warpPerspective for Homography
        # Align RGB
        rgb_aligned[:] = cv2.warpPerspective(
            rgb_memmap2,
            warp_matrix_memmap,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        del rgb_aligned
        # Align grayscale
        grayscale_aligned[:] = cv2.warpPerspective(
            gray_memmap2,
            warp_matrix_memmap,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        del grayscale_aligned
    else:  # Use warpAffine for Translation, Euclidean and Affine
        # Align RGB
        rgb_aligned[:] = cv2.warpAffine(
            rgb_memmap2,
            warp_matrix_memmap,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        del rgb_aligned
        # Align grayscale
        grayscale_aligned[:] = cv2.warpAffine(
            gray_memmap2,
            warp_matrix_memmap,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        del grayscale_aligned

    print("SEVEN: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    print("Successfully aligned {} to {}".format(im1_path, im0_path))

    return [im1_path, rgb_aligned_name, grayscale_aligned_name]


# Reduce layer for gaussian pyramid
@ray.remote
def reduceLayer(layer):
    def generating_kernel(a):
        kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
        return np.outer(kernel, kernel)

    def convolve(image, kernel=generating_kernel(0.4)):
        return ndimage.convolve(image.astype(np.float64), kernel, mode="mirror")

    kernel = generating_kernel(0.4)
    if len(layer.shape) == 2:
        convolution = convolve(layer, kernel)
        return convolution[::2, ::2]

    ch_layer = ray.get(reduceLayer.remote(layer[:, :, 0]))
    next_layer = np.memmap(
        tempfile.NamedTemporaryFile(),
        mode="w+",
        shape=tuple(list(ch_layer.shape) + [layer.shape[2]]),
        dtype=ch_layer.dtype,
    )
    next_layer[:, :, 0] = ch_layer

    # Get data in parallel
    data = [
        reduceLayer.remote(layer[:, :, channel]) for channel in range(1, layer.shape[2])
    ]
    data = ray.get(data)
    # Write to arrays
    for index, value in enumerate(data):
        next_layer[:, :, index] = value

    return next_layer
