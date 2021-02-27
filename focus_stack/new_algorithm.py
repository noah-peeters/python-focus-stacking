import cv2
import numpy as np
import os
import glob
import dask

directory_name = "images"   # Directory to search in
file_name_pattern = '*.jpg' # Extension of images to search for
laplacian_kernel_size = 5   # SIze of laplacian kernel
gaussian_blur_size = 5      # Size of gaussian blur

# Extensions for memmap files
rgb_memmap_extension = ".rgb"
grayscale_memmap_extension = ".grayscale"
laplacian_memmap_extension = ".laplacian"

memmap_extensions = [rgb_memmap_extension, grayscale_memmap_extension, laplacian_memmap_extension]

SHAPE = None

# Load images (RGB and Grayscale) and write them to disk (seperately) parrallelized using Dask
def load_images():
    print("Loading Images")
    def load_single_image(image_path):
        print("Loading: " + image_path)
        global SHAPE
        # Load in memory using cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        SHAPE = image_bgr.shape

        # Write to disk (memmap)
        memmapped_rgb = np.memmap(image_path + rgb_memmap_extension, mode="w+", shape=SHAPE)
        memmapped_rgb[:] = image_rgb
        del memmapped_rgb

        memmapped_grayscale = np.memmap(image_path + grayscale_memmap_extension, mode="w+", shape=(SHAPE[0], SHAPE[1]))
        memmapped_grayscale[:] = image_grayscale
        del memmapped_grayscale

    # Load all images in parallel
    image_paths = glob.glob(os.path.join(directory_name, file_name_pattern))
    image_paths.sort()
    images = []
    for image_path in image_paths:
        images.append(dask.delayed(load_single_image)(image_path))

    dask.compute(*images)

    return image_paths  # Return listing of images

# Align images in parallel (to middle of stack)
def align_images(image_paths):
    print("Aligning Images")

    '''
    Takes two image_paths, and and aligns the second one to the first one (overwriting memmap of second RGB image).
    Using a warp_mode other than cv2.MOTION_TRANSLATION takes a very, very long time and does not significantly improve image quality.
    '''
    def align_single_image(im1_path, im2_path):
        print("Aligning " + im2_path + " to: " + im1_path)
        global SHAPE
        # Get memmap's
        im1_gray = np.memmap(im1_path + grayscale_memmap_extension, mode="r", shape=(SHAPE[0], SHAPE[1]))
        im2_gray = np.memmap(im2_path + grayscale_memmap_extension, mode="r", shape=(SHAPE[0], SHAPE[1]))
        im2_rgb = np.memmap(im2_path + rgb_memmap_extension, mode="r+", shape=SHAPE)

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
        # warp_mode = cv2.MOTION_HOMOGRAPHY
        # warp_mode = cv2.MOTION_AFFINE

        # Define 2x3 or 3x3 matrices
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (_, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective (im2_rgb, warp_matrix, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_rgb, warp_matrix, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # Overwrite RGB memmap of second image
        im2_rgb[:] = im2_aligned

        del im1_gray
        del im2_gray
        del im2_rgb

    # Compute in parallel
    im0_path = image_paths[round(len(image_paths)/2)] # Middle image
    aligning = []
    for im_path in image_paths:
        if im_path != im0_path:
            aligning.append(dask.delayed(align_single_image)(im0_path, im_path))
    
    dask.compute(*aligning)

# Calculate edges using laplacian filter, from grayscale
def calculate_edges_laplacian(image_paths):
    def calculate_single_laplacian(image_path):
        global SHAPE
        grayscale_image = np.memmap(image_path + grayscale_memmap_extension, mode="r", shape=(SHAPE[0], SHAPE[1]))
        blurred = cv2.GaussianBlur(grayscale_image, (gaussian_blur_size, gaussian_blur_size), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_kernel_size)

        # Write to disk
        memmapped_laplacian = np.memmap(image_path + laplacian_memmap_extension, mode="w+", shape=(SHAPE[0], SHAPE[1]), dtype="float64") # dtype="float64"!!
        memmapped_laplacian[:] = laplacian

        del grayscale_image
        del memmapped_laplacian

    # Compute in parallel
    laplacians = []
    for im_path in image_paths:
        laplacians.append(dask.delayed(calculate_single_laplacian)(im_path))
    
    dask.compute(*laplacians)

# Find points of highest focus and merge them into a new image
def focus_stack(image_paths):
    print("Focus stacking...")

    images = []
    laplacians = []
    for im_path in image_paths:
        global SHAPE
        images.append(np.memmap(im_path + rgb_memmap_extension, mode="r", shape=SHAPE))
        laplacians.append(np.memmap(im_path + laplacian_memmap_extension, mode="r", shape=(SHAPE[0], SHAPE[1])))

    laplacians = np.asarray(laplacians)
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    for y in range(0, images[0].shape[0]):                  # Loop through vertical pixels (columns)
        for x in range(0, images[0].shape[1]):              # Loop through horizontal pixels (rows)
            yxlaps = abs(laplacians[:, y, x])               # Absolute value of laplacian at this pixel
            index = (np.where(yxlaps == max(yxlaps)))[0][0]
            output[y, x] = images[index][y, x]              # Write focus pixel to output image
    
    return output

print('LOADING files in {}'.format(directory_name))
image_paths = load_images()             # Write all images to memmap's
align_images(image_paths)               # Align images
calculate_edges_laplacian(image_paths)  # Calculate edges with a laplacian gradient 
im = focus_stack(image_paths)

# Save image to disk
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cv2.imwrite(directory_name + "/focus_stacked.jpg", im) 

# Cleanup memmapped (binary) files
for im_path in image_paths:
    for ext in memmap_extensions:
        os.remove(im_path + ext)

print("Stacking completed")