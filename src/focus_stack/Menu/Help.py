"""
Module containing classes for the "Help" menubar.
These classes are Qt widgets that will get shown on menu click.
"""

import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import platform, psutil


class AboutApplication(qtw.QMessageBox):
    def __init__(self):
        super().__init__()
        self.setIcon(qtw.QMessageBox.Information)
        self.setWindowTitle("About PyStacker")
        self.setText(
            "<b>PyStacker</b> is an application written in <b>Python</b> and <b>Qt</b> that is completely open-source: "
            "<a href='https://github.com/noah-peeters/python-focus-stacking'>PyStacker on github</a>"
            " and aims to provide an advanced focus stacking application for free.",
        )
        self.setStandardButtons(qtw.QMessageBox.Ok)
        self.exec_()


class AboutThisPc(qtw.QMessageBox):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("About this PC")
        self.setStandardButtons(qtw.QMessageBox.Ok)

        try:
            info = {}
            info["platform"] = platform.system()
            info["platform-release"] = platform.release()
            info["platform-version"] = platform.version()
            info["architecture"] = platform.machine()
            info["processor"] = platform.processor()
            info["ram"] = (
                str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
            )
            # Success loading PC data
            self.setIcon(qtw.QMessageBox.Information)

        except Exception as e:
            # Failed to load PC data
            self.setIcon(qtw.QMessageBox.Critical)
            self.setText(
                "Something went wrong while getting your PC's information... ("
                + e
                + ")"
            )
        
        self.exec_()