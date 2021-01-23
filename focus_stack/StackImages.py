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

        image = cv2.imread(imPath)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width, channels = image.shape



        # Set image dimensions (for use in other functions)
        if not imageHeight:
            imageHeight = height
        if not imageWidth:
            imageWidth = width

        # Create image memmap
        memMappedImg = numpy.memmap(imPath + ".img", mode="w+", shape=(imageHeight, imageWidth, 3)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedImg[:] = numpy.asarray(image) # Copy RGB img to memmap

        # Unmount
        del memMappedImg

        # Create grayscale image memmap
        memMappedGrayscale = numpy.memmap(imPath + ".grayscale", mode="w+", shape=(imageHeight, imageWidth)) # Create a memory mapped array for storing raw image data (matching image dimensions)
        memMappedGrayscale[:] = numpy.asarray(grayscale) # Copy img to memmap
        # Unmount
        del memMappedGrayscale

    processes = []
    # Insert processes in list
    imgFileList = glob.glob(imgFolder) # TODO: Error when dir contains other items
    for imPath in sorted(imgFileList):
        processes.append(dask.delayed(loadImage)(imPath))
    
    # Load all images in parallel
    dask.compute(*processes)


# Function for applying a gaussian blur and a laplacian gradient on all loaded images (written back to .grayscale)
def processGrayscaleLoadedImages(imgFolder):
    def processImage(imPath):
        global imageHeight
        global imageWidth

        memMappedImgArray = numpy.memmap(imPath + ".grayscale", mode="r+", shape=(imageHeight, imageWidth))

        blurredImg = cv2.GaussianBlur(memMappedImgArray, (7, 7), 0)
        laplacianGradient = cv2.Laplacian(blurredImg, -1, ksize=3) # ddepth -1 for same as src image (cv2.CV_64F)

        memMappedImgArray[:] = laplacianGradient   # Store processed (grayscale) image
        del memMappedImgArray

    processes = []
    imgFileList = glob.glob(imgFolder)
    for imPath in sorted(imgFileList):
        processes.append(dask.delayed(processImage)(imPath))

    dask.compute(*processes)

# Function for stacking images using images and their laplacian gradients and
def stackLoadedImages(imgFolder):
    # Get all images (in sorted order) and their laplacian gradient
    images = []
    laplacianGradient = []
    imgFileList = glob.glob(imgFolder)
    for imPath in sorted(imgFileList):
        images.append(numpy.memmap(imPath + ".img", mode="r", shape=(imageHeight, imageWidth, 3)))
        laplacianGradient.append(numpy.memmap(imPath + ".grayscale", mode="r", shape=(imageHeight, imageWidth)))

    laplacianGradient = numpy.asarray(laplacianGradient)


    output = numpy.zeros_like(images[0])                    # Create new array filled with zeros
    absLaplacian = numpy.absolute(laplacianGradient)        # Get absolute values of Laplacian gradients
    maxLaplacian = absLaplacian.max(axis=0)                 # Get max value of Laplacian gradients
    boolMask = numpy.array(absLaplacian == maxLaplacian)    # Create bool mask (true or false values)
    mask = boolMask.astype(numpy.uint8)                     # Convert true/false into 1/0

    for i, img in enumerate(images):
        output = cv2.bitwise_not(img, output, mask=mask[i])

    outputImage = 255 - output

    return cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB) #* PILLOW (PIL) is in RGB and cv2 is in BGR! Image must be converted from BGR to RGB.

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
memMappedImg = numpy.memmap("stacked.img", mode="w+", shape=(imageHeight, imageWidth, 3))
memMappedImg[:] = stackedImg
del memMappedImg

Image.fromarray(stackedImg).show()

print("--- Created stack in: %s seconds ---" % (time.time() - start_time))

