# -*- coding: utf-8 -*-
import setuptools
from setuptools import setup
from focus_stack import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="focus-stack-concurrent",
    version=__version__,
    author="Noah Peeters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noah-peeters/python-focus-stacking",
    description="Tool to focus stack images using Dask and Memory mapped structures for better concurrency.",
    packages=setuptools.find_packages(),
    package_dir={"focus_stack": "focus_stack", "QtUi": "QtUi"},
    entry_points={
        "console_scripts": ["focusstack = focus_stack.run:main"],
        "gui_scripts": ["QtUi = QtUi.main:main"],
    },
    install_requires=["PyQt5"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
