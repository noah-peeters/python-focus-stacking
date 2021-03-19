"""
    Module containing helper functions that execute small tasks.
    These can be parallellized with Ray.
"""

import tempfile
import numpy as np
import cv2
import ray

# Function to load an image. Returns memmaps of images and other useful information
@ray.remote
def loadImage(image_path, dir):
    """
    Load an image in RGB, convert to grayscale and get its shape.
    """
    # Load in memory using cv2
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_shape = image_rgb.shape

    # Write to memmaps
    rgb_handle, rgb_name = tempfile.mkstemp(suffix=".rgb_source", dir=dir)
    rgb_memmap = np.memmap(
        rgb_name,
        mode="w+",
        shape=image_shape,
        dtype=image_rgb.dtype,
    )
    rgb_memmap[:] = image_rgb

    grayscale_handle, grayscale_name = tempfile.mkstemp(
        suffix=".grayscale_source", dir=dir
    )
    grayscale_memmap = np.memmap(
        grayscale_name,
        mode="w+",
        shape=(image_shape[0], image_shape[1]),
        dtype=image_grayscale.dtype,
    )
    grayscale_memmap[:] = image_grayscale

    # Return info
    return [image_path, image_shape, rgb_name, grayscale_name]
