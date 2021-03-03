import numpy as np
from cv2 import cv2
import dask

# Extensions for memmap files
rgb_memmap_extension = ".rgb"
grayscale_memmap_extension = ".grayscale"
laplacian_memmap_extension = ".laplacian"
stacked_memmap_filename = "stacked.rgb"

class MainAlgorithm:
    def __init__(self):
        self.image_shape = []

    # Load a single image
    def load_image(self, image_path):
        # Load in memory using cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.image_shape = image_bgr.shape

        # Write to disk (memmap)
        memmapped_rgb = np.memmap(image_path + rgb_memmap_extension, mode="w+", shape=self.image_shape)
        memmapped_rgb[:] = image_rgb
        del memmapped_rgb

        memmapped_grayscale = np.memmap(image_path + grayscale_memmap_extension, mode="w+", shape=(self.image_shape[0], self.image_shape[1]))
        memmapped_grayscale[:] = image_grayscale
        del memmapped_grayscale

        return True # Successfully loaded image to memmap
    
    # Align a single image
    def align_image(self, im1_path, im2_path):
        # Get memmap's
        im1_gray = np.memmap(im1_path + grayscale_memmap_extension, mode="r", shape=(self.image_shape[0], self.image_shape[1]))
        im2_gray = np.memmap(im2_path + grayscale_memmap_extension, mode="r", shape=(self.image_shape[0], self.image_shape[1]))
        im2_rgb = np.memmap(im2_path + rgb_memmap_extension, mode="r+", shape=self.image_shape)

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
        termination_eps = 1e-8

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (_, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective (im2_rgb, warp_matrix, (self.image_shape[1], self.image_shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_rgb, warp_matrix, (self.image_shape[1], self.image_shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # Overwrite RGB memmap of second image
        im2_rgb[:] = im2_aligned

        del im1_gray
        del im2_gray
        del im2_rgb

        return im1_path, im2_path, True # Operation success

    # Compute the laplacian edges of an image
    def compute_laplacian_image(self, image_path, gaussian_blur_size, laplacian_kernel_size):
        grayscale_image = np.memmap(image_path + grayscale_memmap_extension, mode="r", shape=(self.image_shape[0], self.image_shape[1]))

        blurred = cv2.GaussianBlur(grayscale_image, (gaussian_blur_size, gaussian_blur_size), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_kernel_size)

        # Write to disk
        memmapped_laplacian = np.memmap(image_path + laplacian_memmap_extension, mode="w+", shape=(self.image_shape[0], self.image_shape[1]), dtype="float64") # dtype="float64"!!
        memmapped_laplacian[:] = laplacian

        del grayscale_image
        del memmapped_laplacian

        return True

    def load_stack_images(self, image_paths):
        images = []
        laplacians = []
        for im_path in image_paths:
            global SHAPE
            images.append(np.memmap(im_path + rgb_memmap_extension, mode="r", shape=self.image_shape))
            laplacians.append(np.memmap(im_path + laplacian_memmap_extension, mode="r", shape=(self.image_shape[0], self.image_shape[1]), dtype="float64"))

        laplacians = np.asarray(laplacians)

        self.rgb_images = images
        self.laplacian_images = laplacians

    def stack_images(self):
        output = np.zeros(shape=self.rgb_images[0].shape, dtype=self.rgb_images[0].dtype)

        for y in range(0, self.rgb_images[0].shape[0]):             # Loop through vertical pixels (columns)
            for x in range(0, self.rgb_images[0].shape[1]):         # Loop through horizontal pixels (rows)
                yxlaps = abs(self.laplacian_images[:, y, x])        # Absolute value of laplacian at this pixel
                index = (np.where(yxlaps == max(yxlaps)))[0][0]
                output[y, x] = self.rgb_images[index][y, x]         # Write focus pixel to output image

            yield y # Send progress back to UI
        
        # Delete unused memmaps
        del self.rgb_images
        del self.laplacian_images

        # Write stacked image to memmap
        stacked_memmap = np.memmap(stacked_memmap_filename, mode="w+", shape=self.image_shape)
        stacked_memmap[:] = output


    def get_image_shape(self):
        return self.image_shape