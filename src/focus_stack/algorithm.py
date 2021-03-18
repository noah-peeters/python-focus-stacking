import numpy as np
import cv2
import tempfile
from scipy import ndimage
import logging
import ray
from src.focus_stack.utilities import Utilities
import src.focus_stack.RayFunctions as RayFunctions

# Setup logging
log = logging.getLogger(__name__)


class ImageHandler:
    image_storage = {}
    image_shape = []

    def __init__(self):
        # Initialize algorithms
        self.LaplacianPixelAlgorithm = LaplacianPixelAlgorithm(self)
        self.PyramidAlgorithm = PyramidAlgorithm(self)

    # Function to load a list of images in parallel
    def loadImages(self, image_paths, update_func):
        # Clear image_storage
        self.image_storage = {}

        # Start image loading in parallel
        data = [RayFunctions.loadImage.remote(path) for path in image_paths]

        # Run update loop (wait for one item to finish and send update back to UI)
        finished = []
        while True:
            ready_ref, remaining_refs = ray.wait(data, num_returns=1, timeout=None)
            data = remaining_refs

            ready_ref = ray.get(ready_ref)  # Get value
            finished.append(ready_ref[0])  # Add finished image to table
            update_func(ready_ref[0][0])  # Send loaded image path to UI

            if not data:
                break  # All images have been loaded

        # Extract data and write to image_storage
        image_paths = []
        for info_table in finished:
            image_path = info_table[0]
            image_shape = info_table[1]
            rgb_memmap = info_table[2]
            grayscale_memmap = info_table[3]

            image_paths.append(image_path)

            self.image_storage[image_path] = {
                "rgb_source": rgb_memmap,
                "grayscale_source": grayscale_memmap,
                "image_shape": image_shape,
            }

        del finished

        return image_paths  # Return loaded images to UI

    # Align a single image
    @ray.remote
    def alignImage(self, im1_path, im2_path, parameters):
        # Checks
        if not im1_path in self.image_storage:
            return
        elif not im2_path in self.image_storage:
            return
        elif not "grayscale_source" in self.image_storage[im1_path]:
            return
        elif not "grayscale_source" in self.image_storage[im2_path]:
            return
        elif not "rgb_source" in self.image_storage[im2_path]:
            return
        elif not "image_shape" in self.image_storage[im2_path]:
            return

        # Shorthands
        im2_storage = self.image_storage[im2_path]
        shape = im2_storage["image_shape"]

        # Get memmap's
        im1_gray = self.image_storage[im1_path]["grayscale_source"]
        im2_gray = im2_storage["grayscale_source"]
        im2_rgb = im2_storage["rgb_source"]

        # Get motion model from parameters
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

        # Store aligned grayscale image in new memmap
        im2_storage["grayscale_aligned"] = np.memmap(
            tempfile.NamedTemporaryFile(), mode="w+", shape=(shape[0], shape[1])
        )
        im2_storage["grayscale_aligned"][:] = im2_grayscale_aligned

        # Store aligned RGB image in new memmap
        im2_storage["rgb_aligned"] = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=shape,
        )
        im2_storage["rgb_aligned"][:] = im2_aligned

        log.info("Successfully aligned %s to %s".format(im1_path, im2_path))

        return im1_path, im2_path, True  # Operation success

    # Return image shape
    def getImageShape(self):
        return self.image_shape

    # Export image to path
    def exportImage(self, path):
        if "stacked image" in self.image_storage:
            output = self.image_storage["stacked image"]
            rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, rgb)
            del output
            return True
        else:
            return False  # No stacked image

    # Get image (specified type aka. RGB, grayscale, aligned, ...) from storage
    def getImageFromPath(self, path, im_type):
        if path in self.image_storage and im_type in self.image_storage[path]:
            return self.image_storage[path][im_type]

    # Downscale an image
    def downscaleImage(self, image, scale_percent):
        new_dim = (
            round(image.shape[1] * scale_percent / 100),
            round(image.shape[0] * scale_percent / 100),
        )  # New width and height
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


class LaplacianPixelAlgorithm:
    """
    Class that handles image stacking using a gaussian / laplacian pyramid.
    Uses images from the "ImageHandler" class.
    """

    def __init__(self, parent):
        self.Parent = parent
        log.info("Initialized Laplacian pixel algorithm.")

    # Compute the laplacian edges of an image
    @ray.remote
    def computeLaplacianEdges(self, image_path, parameters):
        if not self.Parent.image_storage[image_path]:
            return
        elif not self.Parent.image_storage[image_path]["grayscale_source"]:
            return

        if self.Parent.image_storage[image_path]["grayscale_aligned"]:
            grayscale_image = self.Parent.image_storage[image_path]["grayscale_aligned"]
        else:
            grayscale_image = self.Parent.image_storage[image_path]["grayscale_source"]

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

        # Save gaussian blurred grayscale
        memmapped_blur = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
            dtype=blurred.dtype,
        )
        memmapped_blur[:] = blurred
        self.Parent.image_storage[image_path]["grayscale_gaussian"] = memmapped_blur

        # Save laplacian grayscale
        memmapped_laplacian = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
            dtype=blurred.dtype,
        )
        memmapped_laplacian[:] = laplacian
        self.Parent.image_storage[image_path][
            "grayscale_laplacian"
        ] = memmapped_laplacian

        return True

    # Calculate output image (final stacking)
    @ray.remote
    def stackImages(self, image_paths):
        """
        Load rgb images and laplacian gradients
        Try using aligned RGB images (if there), or use source RGB images
        """
        rgb_images = []
        laplacian_images = []
        for im_path in image_paths:
            if "rgb_aligned" in self.Parent.image_storage[im_path]:
                rgb_image = self.Parent.image_storage[im_path]["rgb_aligned"]
            else:
                rgb_image = self.Parent.image_storage[im_path]["rgb_source"]

            rgb_images.append(
                np.memmap(
                    rgb_image,
                    mode="r",
                    shape=self.Parent.image_shape,
                )
            )
            if "grayscale_aligned" in self.Parent.image_storage[im_path]:
                grayscale_image = self.Parent.image_storage[im_path][
                    "grayscale_aligned"
                ]
            else:
                grayscale_image = self.Parent.image_storage[im_path]["grayscale_source"]

            laplacian_images.append(
                np.memmap(
                    grayscale_image,
                    mode="r",
                    shape=(self.Parent.image_shape[0], self.Parent.image_shape[1]),
                    dtype="float64",
                )
            )

        """
            Calculate output image
        """
        # Create memmap (same size as rgb input)
        stacked_memmap = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=self.Parent.image_shape,
            dtype=rgb_images[0].dtype,
        )

        for y in range(rgb_images[0].shape[0]):  # Loop through vertical pixels (rows)
            # Create holder for whole row
            holder = np.memmap(
                tempfile.NamedTemporaryFile(),
                mode="w+",
                shape=[1, stacked_memmap.shape[1], stacked_memmap.shape[2]],
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

        # Store stacked image
        self.Parent.image_storage["stacked image"] = stacked_memmap


class PyramidAlgorithm:
    """
    Class that handles image stacking using a gaussian / laplacian pyramid.
    Uses inherited images from the "ImageHandler" class.
    """

    def __init__(self, parent):
        self.Parent = parent
        log.info("Initialized gaussian/laplacian pyramid algorithm.")

    def generating_kernel(a1, a2=None):
        if a1 and not a2:
            # Not called with self
            kernel = np.array([0.25 - a1 / 2.0, 0.25, a1, 0.25, 0.25 - a1 / 2.0])
        else:
            # Called with self
            kernel = np.array([0.25 - a2 / 2.0, 0.25, a2, 0.25, 0.25 - a2 / 2.0])
        return np.outer(kernel, kernel)

    def expand_layer(self, layer, kernel=generating_kernel(0.4)):
        if len(layer.shape) == 2:
            expand = next_layer = np.memmap(
                tempfile.NamedTemporaryFile(),
                mode="w+",
                shape=(2 * layer.shape[0], 2 * layer.shape[1]),
                dtype=np.float64,
            )
            expand[::2, ::2] = layer
            convolution = self.convolve(expand, kernel)
            return 4.0 * convolution

        ch_layer = self.expand_layer(layer[:, :, 0])
        next_layer = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=tuple(list(ch_layer.shape) + [layer.shape[2]]),
            dtype=ch_layer.dtype,
        )
        next_layer[:, :, 0] = ch_layer

        for channel in range(1, layer.shape[2]):
            next_layer[:, :, channel] = self.expand_layer(layer[:, :, channel])

        return next_layer

    def convolve(self, image, kernel=generating_kernel(0.4)):
        return ndimage.convolve(image.astype(np.float64), kernel, mode="mirror")

    def gaussian_pyramid(self, levels, images):
        @ray.remote
        def reduce_layer(layer, kernel=self.generating_kernel(0.4)):
            if len(layer.shape) == 2:
                convolution = self.convolve(layer, kernel)
                return convolution[::2, ::2]

            ch_layer = ray.get(reduce_layer.remote(layer[:, :, 0]))
            next_layer = np.memmap(
                tempfile.NamedTemporaryFile(),
                mode="w+",
                shape=tuple(list(ch_layer.shape) + [layer.shape[2]]),
                dtype=ch_layer.dtype,
            )
            next_layer[:, :, 0] = ch_layer

            for channel in range(1, layer.shape[2]):
                next_layer[:, :, channel] = ray.get(
                    reduce_layer.remote(layer[:, :, channel])
                )

            return next_layer

        # Convert images to float64
        for image in images:
            image = image.astype(np.float64, copy=False)
        pyramid = [images]

        while levels > 0:
            next_layer = ray.get(reduce_layer.remote(pyramid[-1][0]))
            next_layer_size = [len(images)] + list(next_layer.shape)

            pyramid.append(
                np.memmap(
                    tempfile.NamedTemporaryFile(),
                    mode="w+",
                    shape=tuple(next_layer_size),
                    dtype=next_layer.dtype,
                )
            )
            pyramid[-1][0] = next_layer

            for layer in range(1, len(images)):
                pyramid[-1][layer] = ray.get(reduce_layer.remote(pyramid[-2][layer]))
            levels -= 1

        return pyramid

    def laplacian_pyramid(self, images, gaussian):
        pyramid = [gaussian[-1]]
        for level in range(len(gaussian) - 1, 0, -1):
            gauss = gaussian[level - 1]
            d = gauss[0].shape
            pyramid.append(
                np.memmap(
                    tempfile.NamedTemporaryFile(),
                    mode="w+",
                    shape=(len(images), d[0], d[1], d[2]),
                    dtype=np.float64,
                )
            )

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
        probabilities = np.memmap(
            tempfile.NamedTemporaryFile(), mode="w+", shape=(256,), dtype=np.float64
        )
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
        entropies = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=image.shape[:2],
            dtype=np.float64,
        )
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
        deviations = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=image.shape[:2],
            dtype=np.float64,
        )
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
        entropies = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=images.shape[:3],
            dtype=np.float64,
        )
        deviations = np.copy(entropies)
        for layer in range(layers):
            gray_image = cv2.cvtColor(
                images[layer].astype(np.float32), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)
            entropies[layer] = self.entropy(gray_image, kernel_size)
            deviations[layer] = self.deviation(gray_image, kernel_size)

        best_e = np.argmax(entropies, axis=0)
        best_d = np.argmax(deviations, axis=0)
        fused = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=images.shape[1:],
            dtype=np.float64,
        )

        for layer in range(layers):
            fused += np.where(best_e[:, :, np.newaxis] == layer, images[layer], 0)
            fused += np.where(best_d[:, :, np.newaxis] == layer, images[layer], 0)

        return (fused / 2).astype(images.dtype)

    def get_fused_laplacian(self, laplacians):
        def region_energy(laplacian):
            return self.convolve(np.square(laplacian))

        layers = laplacians.shape[0]
        region_energies = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=laplacians.shape[:3],
            dtype=np.float64,
        )

        for layer in range(layers):
            gray_lap = cv2.cvtColor(
                laplacians[layer].astype(np.float32), cv2.COLOR_BGR2GRAY
            )
            region_energies[layer] = region_energy(gray_lap)

        best_re = np.argmax(region_energies, axis=0)
        fused = np.memmap(
            tempfile.NamedTemporaryFile(),
            mode="w+",
            shape=laplacians.shape[1:],
            dtype=laplacians.dtype,
        )

        for layer in range(layers):
            fused += np.where(best_re[:, :, np.newaxis] == layer, laplacians[layer], 0)

        return fused

    def fusePyramid(self, image_paths, parameters):
        log.info("Start laplacian pyramid stacking.")

        images = []
        for path in image_paths:
            images.append(
                np.memmap(
                    self.Parent.rgb_images_temp_files[path],
                    mode="r",
                    shape=self.Parent.image_shape,
                )
            )

        log.info("Just loaded {} images.".format(len(image_paths)))
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        # Calculate gaussian pyramid
        smallest_side = min(images[0].shape[:2])
        depth = int(np.log2(smallest_side / parameters["MinLaplacianSize"]))
        gaussian = self.gaussian_pyramid(depth, images)
        log.info("Just calculated gaussian pyramid.")
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        # Calculate laplacian pyramid
        pyramids = self.laplacian_pyramid(images, gaussian)
        log.info("Just calculated laplacian pyramid.")
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        # Fuse pyramid
        kernel_size = 5
        fused = [self.get_fused_base(pyramids[-1], kernel_size)]
        for layer in range(len(pyramids) - 2, -1, -1):
            fused.append(self.get_fused_laplacian(pyramids[layer]))

        fused = fused[::-1]
        log.info("Just fused pyramid.")
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        # Collapse pyramid
        image = fused[-1]
        for layer in fused[-2::-1]:
            expanded = self.expand_layer(image)
            if expanded.shape != layer.shape:
                expanded = expanded[: layer.shape[0], : layer.shape[1]]
            image = expanded + layer

        log.info("Just collapsed pyramid.")
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        # Create memmap (same size as rgb input)
        stacked_temp_file = tempfile.NamedTemporaryFile()
        stacked_memmap = np.memmap(
            stacked_temp_file,
            mode="w+",
            shape=self.Parent.image_shape,
            dtype=images[0].dtype,
        )

        stacked_memmap[:] = image
        log.info("Successfully created stacked image.")
        log.info(
            "Memory is now: {} MB".format(round(Utilities().python_memory_usage()), 2)
        )

        self.Parent.stacked_image_temp_file = stacked_temp_file  # Store temp file

        del stacked_memmap
