# Focus Stacking
A focus stacking application written in Python that provides image alignment (OpenCV) and several stacking algorithms. Everything being nicely contained inside a GUI (Graphical User Interface) written using (Py)Qt5, that is easy to use. 

Main resources used:

 - Image alignment and laplacian algorithm adapted from:
   [momonola/focus-stack](https://github.com/momonala/focus-stack) and
   https://osf.io/j8kby/. Changes were needed for stacking larger images
   
 - Laplacian Pyramid algorithm adapted from:
   https://github.com/sjawhar/focus-stacking

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
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI1MjA2MDk0MSwtNDI0NTM0NTM3LC0xOD
k1NzQ5NTA2XX0=
-->