import numpy as np
import cv2
import tempfile


class MainAlgorithm:
    def __init__(self):
        self.image_shape = []

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
        self.rgb_images_temp_files[
            image_path
        ] = temp_rgb_file  # Keep reference to temporary file

        temp_grayscale_file = tempfile.NamedTemporaryFile()
        memmapped_grayscale = np.memmap(
            temp_grayscale_file,
            mode="w+",
            shape=(self.image_shape[0], self.image_shape[1]),
        )
        memmapped_grayscale[:] = image_grayscale
        self.grayscale_images_temp_files[image_path] = temp_grayscale_file

        del memmapped_rgb
        del memmapped_grayscale
        return True  # Successfully loaded image to memmap

    # Align a single image
    def align_image(self, im1_path, im2_path, parameters):
        # Get memmap's
        im1_gray = np.memmap(
            self.grayscale_images_temp_files[im1_path],
            mode="r",
            shape=(self.image_shape[0], self.image_shape[1]),
        )
        im2_gray = np.memmap(
            self.grayscale_images_temp_files[im2_path],
            mode="r",
            shape=(self.image_shape[0], self.image_shape[1]),
        )
        im2_rgb = np.memmap(
            self.rgb_images_temp_files[im2_path], mode="r", shape=self.image_shape
        )

        # Get motion model from parameters
        warp_mode = cv2.MOTION_TRANSLATION  # Default
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

        # Run the algorithm
        _, warp_matrix = cv2.findTransformECC(
            im1_gray,
            im2_gray,
            warp_matrix,
            warp_mode,
            criteria,
            None,
            gaussian_blur_size,
        )
        del im1_gray
        del im2_gray

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(
                im2_rgb,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(
                im2_rgb,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        del im2_rgb

        # Write aligned to new memmap (tempfile)
        temp_aligned_rgb = tempfile.NamedTemporaryFile()
        aligned_rgb = np.memmap(temp_aligned_rgb, mode="w+", shape=self.image_shape)
        aligned_rgb[:] = im2_aligned
        self.aligned_images_temp_files[
            im2_path
        ] = temp_aligned_rgb  # Keep reference to temporary file

        return im1_path, im2_path, True  # Operation success

    # Compute the laplacian edges of an image
    def compute_laplacian_image(self, image_path, parameters):
        grayscale_image = np.memmap(
            self.grayscale_images_temp_files[image_path],
            mode="r",
            shape=(self.image_shape[0], self.image_shape[1]),
        )

        blurred = grayscale_image
        if parameters["GaussianBlur"] != 0:
            # Blur image
            blurred = cv2.GaussianBlur(
                grayscale_image,
                (parameters["GaussianBlur"], parameters["GaussianBlur"]),
                0,
            )

        laplacian = cv2.Laplacian(
            blurred, cv2.CV_64F, ksize=parameters["LaplacianKernel"]
        )
        del grayscale_image

        # Write gaussian blurred to disk
        temp_gaussian_blurred_file = tempfile.NamedTemporaryFile()
        memmapped_blur = np.memmap(
            temp_gaussian_blurred_file,
            mode="w+",
            shape=(self.image_shape[0], self.image_shape[1]),
            dtype=blurred.dtype,
        )
        memmapped_blur[:] = blurred
        self.gaussian_blurred_images_temp_files[
            image_path
        ] = temp_gaussian_blurred_file  # Store temporary file
        del memmapped_blur

        # Write laplacian to disk
        temp_laplacian_file = tempfile.NamedTemporaryFile()
        memmapped_laplacian = np.memmap(
            temp_laplacian_file,
            mode="w+",
            shape=(self.image_shape[0], self.image_shape[1]),
            dtype=laplacian.dtype,
        )  # dtype="float64" !!
        memmapped_laplacian[:] = laplacian
        self.laplacian_images_temp_files[
            image_path
        ] = temp_laplacian_file  # Store temporary file
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
            rgb_images.append(
                np.memmap(
                    self.useSource_or_Aligned()[im_path],
                    mode="r",
                    shape=self.image_shape,
                )
            )
            laplacian_images.append(
                np.memmap(
                    self.laplacian_images_temp_files[im_path],
                    mode="r",
                    shape=(self.image_shape[0], self.image_shape[1]),
                    dtype="float64",
                )
            )

        """
            Calculate output image
        """
        # Create memmap (same size as rgb input)
        stacked_temp_file = tempfile.NamedTemporaryFile()
        stacked_memmap = np.memmap(
            stacked_temp_file,
            mode="w+",
            shape=self.image_shape,
            dtype=rgb_images[0].dtype,
        )

        for y in range(rgb_images[0].shape[0]):  # Loop through vertical pixels (rows)
            # Create holder for whole row
            holder = np.zeros(
                [1, stacked_memmap.shape[1], stacked_memmap.shape[2]],
                dtype=stacked_memmap.dtype,
            )

            for x in range(
                rgb_images[0].shape[1]
            ):  # Loop through horizontal pixels (columns)

                def get_abs():
                    values = []
                    for arr in laplacian_images:  # noqa: F821
                        values.append(
                            abs(arr[y, x])
                        )  # Insert (absolute) values of this pixel for each image
                    return np.asarray(values, dtype=np.uint8)

                abs_val = get_abs()  # Get absolute value of this pixel from every image
                index = (np.where(abs_val == max(abs_val)))[0][
                    0
                ]  # Get image that has highest value for this pixel

                holder[0, x] = rgb_images[index][
                    y, x
                ]  # Write pixel from "best image" to holder

            stacked_memmap[y] = holder[
                0
            ]  # Write entire focused row to output (holder has only one row)

            yield y  # Send progress back to UI (every row)

        yield rgb_images[0].shape[0]  # Finished

        # Delete unused memmaps
        del rgb_images
        del laplacian_images

        self.stacked_image_temp_file = stacked_temp_file  # Store temp file

    # Export image to path
    def export_image(self, path):
        if self.stacked_image_temp_file:
            output = np.memmap(
                self.stacked_image_temp_file, mode="r", shape=self.image_shape
            )
            rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, rgb)
            del output
            return True
        else:
            return False  # No stacked image

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
        # Get image from path
        image = None
        if im_type == "source":
            image = self.rgb_images_temp_files[path]
            return np.memmap(image, mode="r", shape=self.image_shape)
        elif im_type == "aligned":
            image = self.aligned_images_temp_files[path]
            return np.memmap(image, mode="r", shape=self.image_shape)
        elif im_type == "gaussian":
            image = self.gaussian_blurred_images_temp_files[path]
            return np.memmap(
                image, mode="r", shape=(self.image_shape[0], self.image_shape[1])
            )
        elif im_type == "laplacian":
            image = self.laplacian_images_temp_files[path]
            return np.memmap(
                image,
                mode="r",
                shape=(self.image_shape[0], self.image_shape[1]),
                dtype="float64",
            )
        elif im_type == "stacked":
            image = self.stacked_image_temp_file
            return np.memmap(image, mode="r", shape=self.image_shape)

    def downscaleImage(self, image, scale_percent):
        new_dim = (
            round(image.shape[1] * scale_percent / 100),
            round(image.shape[0] * scale_percent / 100),
        )  # New width and height
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    # Return aligned RGB images (if any) or return source RGB images
    def useSource_or_Aligned(self):
        if len(self.aligned_images_temp_files) > 0:
            return self.aligned_images_temp_files  # Use aligned images
        else:
            return self.rgb_images_temp_files  # Use non-aligned source images
