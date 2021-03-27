import shutil, atexit, os, tempfile, logging
import numpy as np
import cv2
from scipy import ndimage
import ray

from src.focus_stack.utilities import Utilities
import src.focus_stack.RayFunctions as RayFunctions

# Setup logging
log = logging.getLogger(__name__)


class ImageHandler:
    image_storage = {}
    image_shape = ()
    temp_dir_path = None
    rgb_images_temp_files = {}

    def __init__(self):
        # Initialize algorithms
        self.LaplacianPixelAlgorithm = LaplacianPixelAlgorithm(self)
        self.PyramidAlgorithm = PyramidAlgorithm(self)

        # Create tempdirectory (for storing all image data)
        self.temp_dir_path = tempfile.mkdtemp(prefix="python_focus_stacking_")
        # Remove folder on program exit
        atexit.register(self.deleteTempFolder)

    # Load a list of images in parallel
    def loadImages(self, image_paths, update_func):
        # Clear image_storage
        self.image_storage = {}

        # Start image loading in parallel
        data = [
            RayFunctions.loadImage.remote(path, self.temp_dir_path)
            for path in image_paths
        ]

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

        # Extract data and place references to files inside image_storage
        image_paths = []
        for info_table in finished:
            image_path = info_table[0]
            image_shape = info_table[1]
            rgb_file_name = info_table[2]
            grayscale_file_name = info_table[3]

            image_paths.append(image_path)

            self.rgb_images_temp_files[image_path] = rgb_file_name

            self.image_storage[image_path] = {
                "image_shape": image_shape,
                "rgb_source": rgb_file_name,
                "grayscale_source": grayscale_file_name,
            }

        del finished

        return image_paths  # Return loaded images to UI

    # Align a list of images in parallel
    def alignImages(self, image_paths, parameters, update_func):
        data = [
            RayFunctions.alignImage.remote(
                path, parameters, self.image_storage, self.temp_dir_path
            )
            for path in image_paths
        ]

        # Run update loop (wait for one item to finish and send update back to UI)
        finished = []
        while True:
            ready_ref, remaining_refs = ray.wait(data, num_returns=1, timeout=None)
            data = remaining_refs

            ready_ref = ray.get(ready_ref)  # Get value
            finished.append(ready_ref[0])  # Add finished image to table
            update_func(ready_ref[0][0])  # Send loaded image path to UI

            if not data:
                break  # All images have been aligned

        # Extract data and place references to files inside image_storage
        image_paths = []
        for info_table in finished:
            image_path = info_table[0]

            image_paths.append(image_path)
            # Append to image storage ditionary
            self.image_storage[image_path]["rgb_aligned"] = info_table[1]
            self.image_storage[image_path]["grayscale_aligned"] = info_table[2]

        return image_paths

    # Return image shape
    def getImageShape(self):
        return self.image_shape

    # Export image to path
    def exportImage(self, path):
        if "stacked_image" in self.image_storage:
            output = self.getImageFromPath(None, "stacked")
            rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, rgb)
            del output
            return True
        else:
            return False  # No stacked image

    # Get image (specified type aka. RGB, grayscale, aligned, ...) from storage
    def getImageFromPath(self, path, im_type):
        if path in self.image_storage:
            im_root = self.image_storage[path]
            if im_type == "rgb_source" and im_type in im_root:
                return np.memmap(
                    im_root[im_type], mode="r", shape=im_root["image_shape"]
                )

            elif im_type == "rgb_aligned" and im_type in im_root:
                return np.memmap(
                    im_root[im_type], mode="r", shape=im_root["image_shape"]
                )

            elif im_type == "grayscale_gaussian" and im_type in im_root:
                return np.memmap(
                    im_root[im_type], mode="r", shape=im_root["image_shape"]
                )

            elif im_type == "grayscale_laplacian" and im_type in im_root:
                return np.memmap(
                    im_root[im_type],
                    mode="r",
                    shape=im_root["image_shape"],
                    dtype="float64",
                )

        elif im_type == "stacked" and "stacked_image" in self.image_storage:
            im = self.image_storage["stacked_image"]
            return np.memmap(im["file"], mode="r", shape=im["image_shape"])

    # Downscale an image
    def downscaleImage(self, image, scale_percent):
        new_dim = (
            round(image.shape[1] * scale_percent / 100),
            round(image.shape[0] * scale_percent / 100),
        )  # New width and height
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    def deleteTempFolder(self):
        log.info("Removing tempfile directory")
        shutil.rmtree(self.temp_dir_path)

    def clearImages(self):
        log.info("Clearing loaded images and their files")
        self.image_storage = {}
        # Remove all tempfiles inside directory
        folder = self.temp_dir_path
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                log.error("Failed to delete %s. Reason: %s" % (file_path, e))

    def rgbOrAligned(self, path, im_type):
        if path in self.image_storage:
            im = self.image_storage[path]
            # Get image shape based on im_type
            if im_type == "grayscale":
                shape = (im["image_shape"][0], im["image_shape"][1])
            else:
                shape = im["image_shape"]

            if im_type + "_aligned" in im:
                return im[im_type + "_aligned"], shape  # Return aligned image
            elif im_type + "_source" in im:
                return im[im_type + "_source"], shape  # Return source image


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
        from src.focus_stack.RayFunctions import reduceLayer

        self.reduceLayer = reduceLayer

    def generating_kernel(a):
        kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
        return np.outer(kernel, kernel)

    def convolve(self, image, kernel=generating_kernel(0.4)):
        return ndimage.convolve(image.astype(np.float64), kernel, mode="mirror")

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

    def gaussian_pyramid(self, images, levels):
        log.info("Started calculation of Gaussian pyramid.")
        # Convert images to float64
        for image in images:
            image = image.astype(np.float64, copy=False)
        pyramid = [images]

        while levels > 0:
            log.info("Start processing of level {}.".format(levels))
            next_layer = ray.get(self.reduceLayer.remote(pyramid[-1][0]))
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

            for image_index in range(1, len(images)):
                print("Start processing of image {}.".format(image_index))
                pyramid[-1][image_index] = ray.get(
                    self.reduceLayer.remote(pyramid[-2][image_index])
                )
            levels -= 1

        return pyramid

    def laplacian_pyramid(self, images, gaussian):
        log.info("Started calculation of Laplacian pyramid.")
        pyramid = [gaussian[-1]]
        for level in range(len(gaussian) - 1, 0, -1):
            log.info("Start processing of level {}.".format(level))
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

            for image_index in range(len(images)):
                print("Start processing of image {}.".format(image_index))
                gauss_layer = gauss[image_index]
                expanded = self.expand_layer(gaussian[level][image_index])
                if expanded.shape != gauss_layer.shape:
                    expanded = expanded[: gauss_layer.shape[0], : gauss_layer.shape[1]]
                pyramid[-1][image_index] = gauss_layer - expanded

        return pyramid[::-1]

    def fusePyramid(self, image_paths, parameters):
        images = []
        IMAGE_SHAPE = ()
        for path in image_paths:
            im_path, shape = self.Parent.rgbOrAligned(path, "rgb")
            images.append(
                np.memmap(
                    im_path,
                    mode="r",
                    shape=shape,
                )
            )
            IMAGE_SHAPE = shape

        # Calculate Gaussian pyramid
        log.info("Start Gaussian pyramid calculation.")
        smallest_side = min(images[0].shape[:2])
        depth = int(np.log2(smallest_side / parameters["MinLaplacianSize"]))
        gaussian = self.gaussian_pyramid(images, depth)
        log.info("Finished calculating Gaussian pyramid.")

        # Calculate Laplacian pyramid
        log.info("Start Laplacian pyramid calculation.")
        pyramids = self.laplacian_pyramid(images, gaussian)
        log.info("Finished calculating Laplacian pyramid.")

        # Fuse pyramid
        log.info("Start pyramid fusing.")
        kernel_size = 5
        fused = [RayFunctions.getFusedBase.remote(pyramids[-1], kernel_size)]
        for layer in range(len(pyramids) - 2, -1, -1):
            log.info("Start fusing layer {}.".format(layer))
            fused.append(RayFunctions.getFusedLaplacian.remote(pyramids[layer]))
        fused = ray.get(fused)

        # Invert list positions (fused = fused[::-1])
        fused.reverse()
        log.info("Finished fusing pyramid.")

        # Collapse pyramid
        log.info("Start collapsing pyramid.")
        image = fused[-1]
        for layer in fused[-2::-1]:
            expanded = self.expand_layer(image)
            if expanded.shape != layer.shape:
                expanded = expanded[: layer.shape[0], : layer.shape[1]]
            image = expanded + layer

        log.info("Finished collapsing pyramid.")

        # Create memmap (same size as rgb input)
        stacked_temp_file = tempfile.NamedTemporaryFile()
        stacked_memmap = np.memmap(
            stacked_temp_file,
            mode="w+",
            shape=IMAGE_SHAPE,
            dtype=images[0].dtype,
        )

        stacked_memmap[:] = image

        # Store memmap
        self.Parent.image_storage["stacked_image"] = {
            "file": stacked_temp_file,
            "image_shape": IMAGE_SHAPE,
        }

        return stacked_temp_file.name  # Return file name
