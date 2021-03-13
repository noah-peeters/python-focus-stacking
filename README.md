# Focus Stacking
A focus stacking application written in Python that provides image alignment (OpenCV) and several stacking algorithms. Everything being nicely contained inside a GUI (Graphical User Interface) written using (Py)Qt5, that is easy to use.

Adapted algorithms from:

 - Image alignment and laplacian algorithm adapted from:
   [momonola/focus-stack](https://github.com/momonala/focus-stack) and
   https://osf.io/j8kby/. These implementations worked fine for small images, but every image had to be loaded into RAM-memory causing the program to crash. Solution was to use numpy memmaps.

 - Laplacian Pyramid algorithm adapted from: [enter link description here](https://github.com/sjawhar/focus-stacking)

[Documentation page](https://noah-peeters.github.io/python-focus-stacking/)

---
# Setup Conda in VSCode

## create conda virtual environment

 install miniconda: https://docs.conda.io/en/latest/miniconda.html
- create the virtual environment
```
conda env create -f environment.yml
```
- configure VSCode to use the proper miniconda virtual environment: https://code.visualstudio.com/docs/python/environments#_conda-environments

## update conda virtual environment
- update conda
```
conda update conda
```
- update virtua env
```
conda env update -f environment.yml
```

## activate the virtual environment
```
conda activate image-stack
```
---
# Build and Debug Snap Package Locally
## build package:
```
snapcraft
```
or get a shell within the snap:
```
snapcraft --debug
```

## install the local snap
```
sudo snap install ./pyqt-image-focus-stack*.snap --dangerous --devmode
```

## run snap
```
/snap/bin/pyqt-image-focus-stack
```
