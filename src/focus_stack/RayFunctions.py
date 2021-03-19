"""
    Module containing helper functions that execute small tasks.
    These can be parallellized with Ray.
"""

import tempfile, logging
import numpy as np
import cv2
import ray

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

    # Return info
    return [image_path, image_shape, rgb_name, grayscale_name]


# Align a single image
@ray.remote
def alignImage(im1_path, parameters, image_storage, dir):
    im2_path = parameters["image0"]
    # Checks
    if not im1_path in image_storage:
        return
    elif not im2_path in image_storage:
        return
    elif not "grayscale_source" in image_storage[im1_path]:
        return
    elif not "grayscale_source" in image_storage[im2_path]:
        return
    elif not "rgb_source" in image_storage[im2_path]:
        return
    elif not "image_shape" in image_storage[im2_path]:
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

    # Define 2x3 or 3x3 warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

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

    """
    Find transformation
    """
    t = image_storage[im1_path]
    shape = t["image_shape"]
    gray_memmap1 = np.memmap(
        t["grayscale_source"],
        mode="r",
        shape=(shape[0], shape[1]),
    )
    t = image_storage[im2_path]
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

    _, warp_matrix = cv2.findTransformECC(
        gray_memmap1,
        gray_memmap2,
        warp_matrix,
        warp_mode,
        criteria,
        None,
        gaussian_blur_size,
    )

    shape = (rgb_memmap2.shape[1], rgb_memmap2.shape[0])
    if warp_mode == cv2.MOTION_HOMOGRAPHY:  # Use warpPerspective for Homography
        # Align RGB
        im2_aligned = cv2.warpPerspective(
            rgb_memmap2,
            warp_matrix,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        # Align grayscale
        im2_grayscale_aligned = cv2.warpPerspective(
            gray_memmap2,
            warp_matrix,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:  # Use warpAffine for Translation, Euclidean and Affine
        # Align RGB
        im2_aligned = cv2.warpAffine(
            rgb_memmap2,
            warp_matrix,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        # Align grayscale
        im2_grayscale_aligned = cv2.warpAffine(
            gray_memmap2,
            warp_matrix,
            shape,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

    # Store aligned grayscale image in new memmap files
    _, rgb_aligned_name = tempfile.mkstemp(suffix=".rgb_aligned", dir=dir)
    rgb_aligned = np.memmap(
        rgb_aligned_name,
        mode="w+",
        shape=im2_aligned.shape,
        dtype=im2_aligned.dtype,
    )
    rgb_aligned[:] = im2_aligned
    del im2_aligned

    _, grayscale_aligned_name = tempfile.mkstemp(suffix=".grayscale_aligned", dir=dir)
    grayscale_aligned = np.memmap(
        grayscale_aligned_name,
        mode="w+",
        shape=im2_grayscale_aligned.shape,
        dtype=im2_grayscale_aligned.dtype,
    )
    grayscale_aligned[:] = im2_grayscale_aligned

    log.info("Successfully aligned %s to %s".format(im1_path, im2_path))

    return [im2_path, rgb_aligned_name, grayscale_aligned_name]
