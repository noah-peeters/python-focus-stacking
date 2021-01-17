import os
from skimage import data, io
from dask_image.imread import imread
import dask_image.ndfilters
import dask_image.ndmeasure
import dask.array as da
import matplotlib.pyplot as plt
import napari

# Function to convert images to grayscale
def grayscale(rgb):
    # Calculate luminance
    return ((rgb[..., 0] * 0.2125) + (rgb[..., 1] * 0.7154) + (rgb[..., 2] * 0.0721))

# get images and load them with dask
filename_pattern = os.path.join("images", "*.jpg")
images = imread(filename_pattern)


# * Focus stack images
grayscaled_images = grayscale(images)   # Convert images to grayscale

#dask_image.ndfilters.gaussian_filter(images, 0, 0, "reflect", 0.0, 4.0) # Apply gaussian blur to images

with napari.gui_qt():
    viewer = napari.view_image(images)
    viewer.add_image(grayscaled_images)
