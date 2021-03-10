import numpy as np
import cv2
import tempfile

class MainAlgorithm:
    def __init__(self):
        self.image_shape = []
        self.final_stack_row_increment = 75

        # Setup dictionaries for storing temporary matrices (loaded/processed images)
        self.clearTempFiles()

    # Load a single image
    def load_image(self, image_path):
        # Load in memory using cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.image_shape = image_bgr.shape

        # Write to disk (memmap + tempfile)
        temp_rgb_file = tempfile.NamedTemporaryFile()
        memmapped_rgb = np.memmap(temp_rgb_file, mode="w+", shape=self.image_shape)
        memmapped_rgb[:] = image_rgb
        self.rgb_images_temp_files[image_path] = temp_rgb_file  # Keep reference to temporary file

        temp_grayscale_file = tempfile.NamedTemporaryFile()
        memmapped_grayscale = np.memmap(temp_grayscale_file, mode="w+", shape=(self.image_shape[0], self.image_shape[1]))
        memmapped_grayscale[:] = image_grayscale
        self.grayscale_images_temp_files[image_path] = temp_grayscale_file

        del memmapped_rgb
        del memmapped_grayscale
        return True # Successfully loaded image to memmap
    
    # Align a single image
    def align_image(self, im1_path, im2_path):
        # Get memmap's
        im1_gray = np.memmap(self.grayscale_images_temp_files[im1_path], mode="r", shape=(self.image_shape[0], self.image_shape[1]))
        im2_gray = np.memmap(self.grayscale_images_temp_files[im2_path], mode="r", shape=(self.image_shape[0], self.image_shape[1]))
        im2_rgb = np.memmap(self.rgb_images_temp_files[im2_path], mode="r", shape=self.image_shape)

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
        del im1_gray
        del im2_gray

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2_rgb, warp_matrix, (self.image_shape[1], self.image_shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_rgb, warp_matrix, (self.image_shape[1], self.image_shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        del im2_rgb
        
        # Write aligned to new memmap (tempfile)
        temp_aligned_rgb = tempfile.NamedTemporaryFile()
        aligned_rgb = np.memmap(temp_aligned_rgb, mode="w+", shape=self.image_shape)
        aligned_rgb[:] = im2_aligned
        self.rgb_images_temp_files[im2_path] = temp_aligned_rgb  # Keep reference to temporary file

        return im1_path, im2_path, True # Operation success

    # Compute the laplacian edges of an image
    def compute_laplacian_image(self, image_path, gaussian_blur_size, laplacian_kernel_size):
        grayscale_image = np.memmap(self.grayscale_images_temp_files[image_path], mode="r", shape=(self.image_shape[0], self.image_shape[1]))

        blurred = cv2.GaussianBlur(grayscale_image, (gaussian_blur_size, gaussian_blur_size), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_kernel_size)
        del grayscale_image

        # Write gaussian blurred to disk
        temp_gaussian_blurred_file = tempfile.NamedTemporaryFile()
        memmapped_blur = np.memmap(temp_gaussian_blurred_file, mode="w+", shape=(self.image_shape[0], self.image_shape[1]), dtype=blurred.dtype)
        memmapped_blur[:] = blurred
        self.gaussian_blurred_images_temp_files[image_path] = temp_gaussian_blurred_file  # Store temporary file
        del memmapped_blur

        # Write laplacian to disk
        temp_laplacian_file = tempfile.NamedTemporaryFile()
        memmapped_laplacian = np.memmap(temp_laplacian_file, mode="w+", shape=(self.image_shape[0], self.image_shape[1]), dtype=laplacian.dtype) # dtype="float64" !!
        memmapped_laplacian[:] = laplacian
        self.laplacian_images_temp_files[image_path] = temp_laplacian_file  # Store temporary file
        del memmapped_laplacian

        return True

    # Calculate output image (final stacking)
    def stack_images(self, image_paths):
        """
            Load rgb images and laplacian gradients
            Try using aligned RGB images (if there), or use source RGB images
        """
        rgb_images = []
        laplacian_images = []
        for im_path in image_paths:
            rgb_images.append(np.memmap(self.useSource_or_Aligned()[im_path], mode="r", shape=self.image_shape))
            laplacian_images.append(np.memmap(self.laplacian_images_temp_files[im_path], mode="r", shape=(self.image_shape[0], self.image_shape[1]), dtype="float64"))

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
        stacked_temp_file = tempfile.NamedTemporaryFile()
        stacked_memmap = np.memmap(stacked_temp_file, mode="w+", shape=self.image_shape)
        stacked_memmap[:] = output
        self.stacked_image_temp_file = stacked_temp_file    # Store temp file

    # Export image to path
    def export_image(self, path):
        if self.stacked_image_temp_file:
            output = np.memmap(self.stacked_image_temp_file, mode="r", shape=self.image_shape)
            rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, rgb)
            del output
            return True
        else:
            return False    # No stacked image

    def get_image_shape(self):
        return self.image_shape
    
    # Clear all temp file references (Python will garbage-collect tempfiles automatically once they aren't referenced)
    def clearTempFiles(self):
        self.rgb_images_temp_files = {}
        self.grayscale_images_temp_files = {}
        self.aligned_images_temp_files = {}
        self.gaussian_blurred_images_temp_files = {}
        self.laplacian_images_temp_files = {}
        self.stacked_image_temp_file = None
    
    # Get image (using tempfile) from path to tempfile
    def getImageFromPath(self, path, im_type):
        if im_type == "rgb":
            image = self.rgb_images_temp_files[path]
            if image:
                return np.memmap(image, mode="r", shape=self.image_shape)
    
    def downscaleImage(self, image, scale_percent):
        new_dim = (round(image.shape[1] * scale_percent / 100), round(image.shape[0] * scale_percent / 100))  # New width and height
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    
    # Return aligned RGB images (if any) or return source RGB images 
    def useSource_or_Aligned(self):
        if len(self.aligned_images_temp_files) > 0:
            return self.aligned_images_temp_files   # Use aligned images
        else:
            return self.rgb_images_temp_files       # Use non-aligned source images