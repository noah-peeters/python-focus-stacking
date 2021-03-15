# -*- coding: utf-8 -*-
import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="focus-stack-concurrent",
    version="0.0.1",
    author="Noah Peeters",
    description="An Image Focus Stacking GUI written in python3 / pyqt5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noah-peeters/python-focus-stacking",
    project_urls = {
        "Documentation page": "https://noah-peeters.github.io/python-focus-stacking/",
        "Github README": "https://github.com/noah-peeters/python-focus-stacking/blob/master/README.md",
    },
    packages=setuptools.find_packages(where="QtUi"),
    package_dir={"": "QtUi"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
