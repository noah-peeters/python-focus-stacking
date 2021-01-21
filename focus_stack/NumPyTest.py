import numpy
import dask
import glob
import time
import cv2
from PIL import Image

IMAGE_DIR = "images/*.jpg" # Directory containing images + extension of images

imageHeight = None
imageWidth = None

start_time = time.time()

# Function for loading all images (grayscale) inside of a folder
def loadImagesFromFolder(imgFolder):
    # Function for loading one image
    def loadImage(imPath):
        global imageHeight
        global imageWidth

        img = Image.open(imPath).convert("L") # TODO: Temporary conversion to Luminance
        grayscale = img.convert("L")
        width, height = img.size

        # Set image dimensions (for use in other functions)
        if not imageHeight:
            imageHeight = height
        if not imageWidth:
            imageWidth = width

        # Create image memmap
        memMappedImg = numpy.memmap(imPath + ".img", mode="w+", shape=(imageHeight, imageWidth, 3)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        RGBImage = numpy.asarray(cv2.imread(imPath))
        print(RGBImage.shape)
        memMappedImg[:] = RGBImage    # Copy img to memmap

        # Create grayscale image memmap
        memMappedGrayscale = numpy.memmap(imPath + ".grayscale", mode="w+", shape=(imageHeight, imageWidth)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedGrayscale[:] = numpy.asarray(grayscale)    # Copy img to memmap

        # Ummnount memmaps
        del memMappedImg
        del memMappedGrayscale

        # if imPath == "images/DSC_0372.jpg":
        #     memMappedImg = numpy.memmap(imPath + ".grayscale", mode="r", shape=(imageHeight, imageWidth))
        #     Image.fromarray(memMappedImg).show()
        #     del memMappedImg

    processes = []
    # Insert processes in list
    imgFileList = glob.glob(imgFolder)
    for imPath in sorted(imgFileList):
        processes.append(dask.delayed(loadImage)(imPath))
    
    # Load all images in parallel
    dask.compute(*processes)


# Function for applying a gaussian blur on all loaded images
def processGrayscaleLoadedImages(imgFolder):
    def processImage(imPath):
        global imageHeight
        global imageWidth

        memMappedImgArray = numpy.memmap(imPath + ".grayscale", mode="r+", shape=(imageHeight, imageWidth))

        blurredImg = cv2.GaussianBlur(memMappedImgArray, (5, 5), 0)
        laplacianGradient = cv2.Laplacian(memMappedImgArray, cv2.CV_64F, ksize=3)

        memMappedImgArray[:] = laplacianGradient   # Store processed (grayscale) image
        del memMappedImgArray

        # if imPath == "images/DSC_0372.jpg":
        #     memMappedImg = numpy.memmap(imPath + ".grayscale", mode="r", shape=(imageHeight, imageWidth))
        #     Image.fromarray(memMappedImg).show()
        #     del memMappedImg

    processes = []
    imgFileList = glob.glob(imgFolder)
    for imPath in sorted(imgFileList):
        processes.append(dask.delayed(processImage)(imPath))

    dask.compute(*processes)

# Function for stacking images using images and their laplacian gradients and
def stackLoadedImages(imgFolder):
    # Get all images (in sorted order) and their laplacian gradient
    images = []
    laplacians = []
    imgFileList = glob.glob(imgFolder)
    for imPath in sorted(imgFileList):
        images.append(numpy.memmap(imPath + ".img", mode="r", shape=(imageHeight, imageWidth, 3)))
        laplacians.append(numpy.memmap(imPath + ".grayscale", mode="r", shape=(imageHeight, imageWidth)))

    laplacians = numpy.asarray(laplacians)

    output = numpy.zeros(shape=images[0].shape, dtype=images[0].dtype)  # Create new array filled with zeros
    absLaplacian = numpy.absolute(laplacians)                           # Get absolute values of Laplacian gradients
    maxLaplacian = absLaplacian.max(axis=0)                             # Get max value of Laplacian gradients
    boolMask = numpy.array(absLaplacian == maxLaplacian)                # Create new array
    mask = boolMask.astype(numpy.uint8)

    for i, img in enumerate(images):
        output = cv2.bitwise_not(img, output, mask=mask[i])

    outputImage = 255 - output

    return outputImage

loadImagesFromFolder(IMAGE_DIR)
print("--- Loaded all images in: %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# Gaussian blur selected images and compute Laplacian gradient
processGrayscaleLoadedImages(IMAGE_DIR)
print("--- Processed grayscale images in: %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# Stack images
stackedImg = stackLoadedImages(IMAGE_DIR)
# Write stacked image to disk
memMappedImg = numpy.memmap("stacked.img", mode="w+", shape=(imageHeight, imageWidth, 3)) #! Output image is only one dimension?? (convert L to RGB)
memMappedImg[:] = stackedImg
del memMappedImg

print("--- Created stack in: %s seconds ---" % (time.time() - start_time))