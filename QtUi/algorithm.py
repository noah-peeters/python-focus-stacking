import numpy as np
import cv2
import dask

# Extensions for memmap files
rgb_memmap_extension = ".rgb"
grayscale_memmap_extension = ".grayscale"
laplacian_memmap_extension = ".laplacian"
stacked_memmap_filename = "stacked.rgb"

class MainAlgorithm:
    def __init__(self):
        self.image_shape = []
        self.final_stack_row_increment = 75

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

    def stack_images(self, image_paths):
        """
            Load rgb images and laplacian gradients
        """
        rgb_images = []
        laplacian_images = []
        for im_path in image_paths:
            global SHAPE
            rgb_images.append(np.memmap(im_path + rgb_memmap_extension, mode="r", shape=self.image_shape))
            laplacian_images.append(np.memmap(im_path + laplacian_memmap_extension, mode="r", shape=(self.image_shape[0], self.image_shape[1]), dtype="float64"))

        laplacian_images = np.asarray(laplacian_images)
        """
            Calculate output image
        """
        output = np.zeros(shape=rgb_images[0].shape, dtype=rgb_images[0].dtype)
        counter = 0
        for y in range(0, rgb_images[0].shape[0]):             # Loop through vertical pixels (rows)
            counter += 1
            for x in range(0, rgb_images[0].shape[1]):         # Loop through horizontal pixels (columns)
                yxlaps = abs(laplacian_images[:, y, x])        # Absolute value of laplacian at this pixel
                index = (np.where(yxlaps == max(yxlaps)))[0][0]
                output[y, x] = rgb_images[index][y, x]         # Write focus pixel to output image
            
            if counter >= self.final_stack_row_increment:
                counter = 0
                yield y # Send progress back to UI (every increment rows)

        yield rgb_images[0].shape[0] # Finished
        
        # Delete unused memmaps
        del rgb_images
        del laplacian_images

        # Write stacked image to memmap
        stacked_memmap = np.memmap(stacked_memmap_filename, mode="w+", shape=self.image_shape)
        stacked_memmap[:] = output

    # Export image to path
    def export_image(self, path):
        output = np.memmap(stacked_memmap_filename, mode="r", shape=self.image_shape)
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, rgb)
        del output

    def get_image_shape(self):
        return self.image_shape