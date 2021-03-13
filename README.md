# Focus Stacking

This script is an optimized rework of  https://github.com/momonala/focus-stack using Dask and Memory mapped data for faster and more stable execution.


---

# Setup Conda / VSCode
## create conda virtual environment
- install miniconda: https://docs.conda.io/en/latest/miniconda.html
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
