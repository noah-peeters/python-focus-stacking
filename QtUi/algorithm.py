import numpy as np
import cv2
import tempfile
from scipy import ndimage


class ImageHandler:
    image_shape = []
    # Tempfile setup
    rgb_images_temp_files = {}
    grayscale_images_temp_files = {}
    aligned_images_temp_files = {}
    gaussian_blurred_images_temp_files = {}
    laplacian_images_temp_files = {}
    stacked_image_temp_file = None

    def __init__(self):
        # Initialize algorithms
        self.LaplacianPixelAlgorithm = LaplacianPixelAlgorithm(self)
        self.PyramidAlgorithm = PyramidAlgorithm(self)

    # Load an image
    def loadImage(self, image_path):
        """
        Load a single image (RGB and grayscale) using a memmap inside a tempfile.
        Keep a reference to the tempfiles, so they don't get destroyed.
        """
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
    def alignImage(self, im1_path, im2_path, parameters):
        # Get memmap's
        im1_gray = np.memmap(
            self.grayscale_images_temp_files[im1_path],
            mode="r",
            shape=(self.image_shape[0], self.image_shape[1]),
        )
        im2_gray = np.memmap(
            self.grayscale_images_temp_files[im2_path],
            mode="r+",
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

        if warp_mode == cv2.MOTION_HOMOGRAPHY:  # Use warpPerspective for Homography
            # Align RGB
            im2_aligned = cv2.warpPerspective(
                im2_rgb,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
            # Align grayscale
            im2_grayscale_aligned = cv2.warpPerspective(
                im2_gray,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        else:  # Use warpAffine for Translation, Euclidean and Affine
            # Align RGB
            im2_aligned = cv2.warpAffine(
                im2_rgb,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
            # Align grayscale
            im2_grayscale_aligned = cv2.warpAffine(
                im2_gray,
                warp_matrix,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        del im2_rgb

        # Write aligned grayscale image
        im2_gray[:] = im2_grayscale_aligned
        del im2_gray

        # Write aligned to new memmap (tempfile)
        temp_aligned_rgb = tempfile.NamedTemporaryFile()
        aligned_rgb = np.memmap(temp_aligned_rgb, mode="w+", shape=self.image_shape)
        aligned_rgb[:] = im2_aligned
        self.aligned_images_temp_files[
            im2_path
        ] = temp_aligned_rgb  # Keep reference to temporary file

        return im1_path, im2_path, True  # Operation success

    # Return image shape
    def getImageShape(self):
        return self.image_shape

    # Export image to path
    def exportImage(self, path):
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

    # Return aligned RGB images (if any) or return source RGB images
    def useSource_or_Aligned(self):
        if len(self.aligned_images_temp_files) > 0:
            return self.aligned_images_temp_files  # Use aligned images
        else:
            return self.rgb_images_temp_files  # Use non-aligned source images

    # Downscale an image
    def downscaleImage(self, image, scale_percent):
        new_dim = (
            round(image.shape[1] * scale_percent / 100),
            round(image.shape[0] * scale_percent / 100),
        )  # New width and height
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    # Clear all temp file references
    def clearTempFiles(self):
        """
        Clear all temp file references (Python will garbage-collect tempfiles automatically once they aren't referenced)
        """
        self.rgb_images_temp_files = {}
        self.grayscale_images_temp_files = {}
        self.aligned_images_temp_files = {}
        self.gaussian_blurred_images_temp_files = {}
        self.laplacian_images_temp_files = {}
        self.stacked_image_temp_file = None


class LaplacianPixelAlgorithm:
    """
    Class that handles image stacking using a gaussian / laplacian pyramid.
    Uses images from the "ImageHandler" class.
    """

    def __init__(self, parent):
        self.Parent = parent

    # Compute the laplacian edges of an image
    def computeLaplacianEdges(self, image_path, parameters):
        grayscale_image = np.memmap(
            self.Parent.grayscale_images_temp_files[image_path],
            mode="r",
            shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
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
            shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
            dtype=blurred.dtype,
        )
        memmapped_blur[:] = blurred
        self.Parent.gaussian_blurred_images_temp_files[
            image_path
        ] = temp_gaussian_blurred_file  # Store temporary file
        del memmapped_blur

        # Write laplacian to disk
        temp_laplacian_file = tempfile.NamedTemporaryFile()
        memmapped_laplacian = np.memmap(
            temp_laplacian_file,
            mode="w+",
            shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
            dtype=laplacian.dtype,
        )  # dtype="float64" !!
        memmapped_laplacian[:] = laplacian
        self.Parent.laplacian_images_temp_files[
            image_path
        ] = temp_laplacian_file  # Store temporary file
        del memmapped_laplacian

        return True

    # Calculate output image (final stacking)
    def stackImages(self, image_paths):
        """
        Load rgb images and laplacian gradients
        Try using aligned RGB images (if there), or use source RGB images
        """
        rgb_images = []
        laplacian_images = []
        for im_path in image_paths:
            rgb_images.append(
                np.memmap(
                    self.Parent.useSource_or_Aligned()[im_path],
                    mode="r",
                    shape=self.Parent.image_shape,
                )
            )
            laplacian_images.append(
                np.memmap(
                    self.Parent.laplacian_images_temp_files[im_path],
                    mode="r",
                    shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
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
            shape=self.Parent.image_shape,
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

        self.Parent.stacked_image_temp_file = stacked_temp_file  # Store temp file


class PyramidAlgorithm:
    """
    Class that handles image stacking using a gaussian / laplacian pyramid.
    Uses inherited images from the "ImageHandler" class.
    """

    def __init__(self, parent):
        self.Parent = parent

    def generating_kernel(a):
        kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
        return np.outer(kernel, kernel)

    def reduce_layer(self, layer, kernel=generating_kernel(0.4)):
        if len(layer.shape) == 2:
            convolution = self.convolve(layer, kernel)
            return convolution[::2, ::2]

        ch_layer = self.reduce_layer(layer[:, :, 0])
        next_layer = np.zeros(
            list(ch_layer.shape) + [layer.shape[2]], dtype=ch_layer.dtype
        )
        next_layer[:, :, 0] = ch_layer

        for channel in range(1, layer.shape[2]):
            next_layer[:, :, channel] = self.reduce_layer(layer[:, :, channel])

        return next_layer

    def expand_layer(self, layer, kernel=generating_kernel(0.4)):
        if len(layer.shape) == 2:
            expand = np.zeros(
                (2 * layer.shape[0], 2 * layer.shape[1]), dtype=np.float64
            )
            expand[::2, ::2] = layer
            convolution = self.convolve(expand, kernel)
            return 4.0 * convolution

        ch_layer = self.expand_layer(layer[:, :, 0])
        next_layer = np.zeros(
            list(ch_layer.shape) + [layer.shape[2]], dtype=ch_layer.dtype
        )
        next_layer[:, :, 0] = ch_layer

        for channel in range(1, layer.shape[2]):
            next_layer[:, :, channel] = self.expand_layer(layer[:, :, channel])

        return next_layer

    def convolve(self, image, kernel=generating_kernel(0.4)):
        return ndimage.convolve(image.astype(np.float64), kernel, mode="mirror")

    def gaussian_pyramid(self, images, levels):
        # Convert images to float64
        for image in images:
            image = image.astype(np.float64, copy=False)
        pyramid = [images]

        while levels > 0:
            next_layer = self.reduce_layer(pyramid[-1][0])
            next_layer_size = [len(images)] + list(next_layer.shape)

            pyramid.append(np.memmap(tempfile.NamedTemporaryFile(), mode="w+", shape=tuple(next_layer_size), dtype=next_layer.dtype))
            pyramid[-1][0] = next_layer
            
            for layer in range(1, len(images)):
                pyramid[-1][layer] = self.reduce_layer(pyramid[-2][layer])
            levels -= 1

        return pyramid

    def laplacian_pyramid(self, images, gaussian):
        pyramid = [gaussian[-1]]
        for level in range(len(gaussian) - 1, 0, -1):
            gauss = gaussian[level - 1]
            d = gauss[0].shape
            pyramid.append(np.memmap(tempfile.NamedTemporaryFile(), mode="w+", shape=(len(images), d[0], d[1], d[2]), dtype=np.float64))
            
            for layer in range(len(images)):
                gauss_layer = gauss[layer]
                expanded = self.expand_layer(gaussian[level][layer])
                if expanded.shape != gauss_layer.shape:
                    expanded = expanded[: gauss_layer.shape[0], : gauss_layer.shape[1]]
                pyramid[-1][layer] = gauss_layer - expanded

        return pyramid[::-1]

    def collapse(self, pyramid):
        image = pyramid[-1]
        for layer in pyramid[-2::-1]:
            expanded = self.expand_layer(image)
            if expanded.shape != layer.shape:
                expanded = expanded[: layer.shape[0], : layer.shape[1]]
            image = expanded + layer

        return image

    def get_probabilities(self, gray_image):
        levels, counts = np.unique(gray_image.astype(np.uint8), return_counts=True)
        probabilities = np.zeros((256,), dtype=np.float64)
        probabilities[levels] = counts.astype(np.float64) / counts.sum()
        return probabilities

    def entropy(self, image, kernel_size):
        def _area_entropy(area, probabilities):
            levels = area.flatten()
            return -1.0 * (levels * np.log(probabilities[levels])).sum()

        probabilities = self.get_probabilities(image)
        pad_amount = int((kernel_size - 1) / 2)
        padded_image = cv2.copyMakeBorder(
            image, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_REFLECT101
        )
        entropies = np.zeros(image.shape[:2], dtype=np.float64)
        offset = np.arange(-pad_amount, pad_amount + 1)
        for row in range(entropies.shape[0]):
            for column in range(entropies.shape[1]):
                area = padded_image[
                    row + pad_amount + offset[:, np.newaxis],
                    column + pad_amount + offset,
                ]
                entropies[row, column] = _area_entropy(area, probabilities)

        return entropies

    def deviation(self, image, kernel_size):
        def _area_deviation(area):
            average = np.average(area).astype(np.float64)
            return np.square(area - average).sum() / area.size

        pad_amount = int((kernel_size - 1) / 2)
        padded_image = cv2.copyMakeBorder(
            image, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_REFLECT101
        )
        deviations = np.zeros(image.shape[:2], dtype=np.float64)
        offset = np.arange(-pad_amount, pad_amount + 1)
        for row in range(deviations.shape[0]):
            for column in range(deviations.shape[1]):
                area = padded_image[
                    row + pad_amount + offset[:, np.newaxis],
                    column + pad_amount + offset,
                ]
                deviations[row, column] = _area_deviation(area)

        return deviations

    def get_fused_base(self, images, kernel_size):
        layers = images.shape[0]
        entropies = np.zeros(images.shape[:3], dtype=np.float64)
        deviations = np.copy(entropies)
        for layer in range(layers):
            gray_image = cv2.cvtColor(
                images[layer].astype(np.float32), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)
            entropies[layer] = self.entropy(gray_image, kernel_size)
            deviations[layer] = self.deviation(gray_image, kernel_size)

        best_e = np.argmax(entropies, axis=0)
        best_d = np.argmax(deviations, axis=0)
        fused = np.zeros(images.shape[1:], dtype=np.float64)

        for layer in range(layers):
            fused += np.where(best_e[:, :, np.newaxis] == layer, images[layer], 0)
            fused += np.where(best_d[:, :, np.newaxis] == layer, images[layer], 0)

        return (fused / 2).astype(images.dtype)

    def get_fused_laplacian(self, laplacians):
        def region_energy(laplacian):
            return self.convolve(np.square(laplacian))

        layers = laplacians.shape[0]
        region_energies = np.zeros(laplacians.shape[:3], dtype=np.float64)

        for layer in range(layers):
            gray_lap = cv2.cvtColor(
                laplacians[layer].astype(np.float32), cv2.COLOR_BGR2GRAY
            )
            region_energies[layer] = region_energy(gray_lap)

        best_re = np.argmax(region_energies, axis=0)
        fused = np.zeros(laplacians.shape[1:], dtype=laplacians.dtype)

        for layer in range(layers):
            fused += np.where(best_re[:, :, np.newaxis] == layer, laplacians[layer], 0)

        return fused

    def fusePyramid(self, image_paths, parameters):
        images = []
        for path in image_paths:
            images.append(
                np.memmap(
                    self.Parent.rgb_images_temp_files[path],
                    mode="r",
                    shape=self.Parent.image_shape,
                )
            )

        # Calculate gaussian pyramid
        smallest_side = min(images[0].shape[:2])
        depth = int(np.log2(smallest_side / parameters["MinLaplacianSize"]))
        gaussian = self.gaussian_pyramid(images, depth)
        print("Just calculated gaussian pyramid.")

        # Calculate laplacian pyramid
        pyramids = self.laplacian_pyramid(images, gaussian)
        print("Just calculated laplacian pyramid")

        # Fuse pyramid
        kernel_size = 5
        fused = [self.get_fused_base(pyramids[-1], kernel_size)]
        for layer in range(len(pyramids) - 2, -1, -1):
            fused.append(self.get_fused_laplacian(pyramids[layer]))

        fused = fused[::-1]

        print("Just fused pyramid.")

        # Collapse pyramid
        image = fused[-1]
        for layer in fused[-2::-1]:
            expanded = self.expand_layer(image)
            if expanded.shape != layer.shape:
                expanded = expanded[: layer.shape[0], : layer.shape[1]]
            image = expanded + layer
        
        print("Just collapsed pyramid.")

        # Create memmap (same size as rgb input)
        stacked_temp_file = tempfile.NamedTemporaryFile()
        stacked_memmap = np.memmap(
            stacked_temp_file,
            mode="w+",
            shape=self.Parent.image_shape,
            dtype=images[0].dtype,
        )

        stacked_memmap[:] = image

        self.Parent.stacked_image_temp_file = stacked_temp_file  # Store temp file

        del stacked_memmap