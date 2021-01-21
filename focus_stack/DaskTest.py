from dask_image.imread import imread
import napari
import dask

# Function to convert images to grayscale
def grayscale(rgb):
    # Calculate luminance
    return ((rgb[..., 0] * 0.2125) + (rgb[..., 1] * 0.7154) + (rgb[..., 2] * 0.0721))


# get images and load them with dask
# filename_pattern = os.path.join("images", "*.jpg")

images = imread("images/DSC_0374.jpg")

grayscaled_images = grayscale(images)   # Convert images to grayscale
print(images.shape)
print(grayscaled_images.shape)
#smoothed = ndfilters.gaussian_filter(grayscaled_images, sigma=[1, 1])

with napari.gui_qt():
    viewer = napari.view_image(dask.compute(images))
    #viewer.add_image(smoothed)