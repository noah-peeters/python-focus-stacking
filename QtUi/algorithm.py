import numpy as np
from cv2 import cv2
import dask

# Extensions for memmap files
rgb_memmap_extension = ".rgb"
grayscale_memmap_extension = ".grayscale"
laplacian_memmap_extension = ".laplacian"

class MainAlgorithm:
    def __init__(self):
        self.loaded_image_paths = []
        self.image_shape = []

    # Load images
    def load_images(self, image_paths):
        def load_single_image(image_path):
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


        # Load all images in parallel
        image_paths.sort()
        processes = []
        for image_path in image_paths:
            processes.append(dask.delayed(load_single_image)(image_path))

        dask.compute(*processes)

        self.loaded_image_paths = image_paths

    @dask.delayed
    def load_images_generator(self, image_paths):
        for image_path in image_paths:
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

            yield image_path # Return image path that has loaded (for progress bar)
    
    # Align loaded images
    def align_images(self):
        '''
        Takes two image_paths, and and aligns the second one to the first one (overwriting memmap of second RGB image).
        Using a warp_mode other than cv2.MOTION_TRANSLATION takes a very, very long time and does not significantly improve image quality.
        '''
        def align_single_image(im1_path, im2_path):
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

        # Compute
        im0_path = self.loaded_image_paths[round(len(self.loaded_image_paths)/2)] # Middle image
        for im_path in self.loaded_image_paths:
            if im_path != im0_path:
                align_single_image(im0_path, im_path)
