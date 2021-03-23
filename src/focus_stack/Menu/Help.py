"""
Module containing classes for the "Help" menubar.
These classes are Qt widgets that will get shown on menu click.
"""

import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import platform, psutil, socket, re, uuid, psutil, logging, multiprocessing, traceback
import math
import GPUtil

log = logging.getLogger(__name__)


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


class AboutThisPc(qtw.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("About this PC")

        try:
            # Convert bytes to other formats for easier reading
            def convert_size(size_bytes):
                if size_bytes == 0:
                    return "0B"
                size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
                i = int(math.floor(math.log(size_bytes, 1024)))
                p = math.pow(1024, i)
                s = round(size_bytes / p, 2)
                return "%s %s" % (s, size_name[i])

            # Hardware info
            platfrm = platform.system()
            node_name = platform.node()
            platform_release = platform.release()
            platform_version = platform.version()
            architecture = platform.machine()
            processor = platform.processor()
            ram = convert_size(psutil.virtual_memory().total)
            cpu_count = str(multiprocessing.cpu_count())
            # Network info
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(socket.gethostname())
            mac_address = ":".join(re.findall("..", "%012x" % uuid.getnode()))

            # Success loading PC data
            layout = qtw.QFormLayout()
            # Hardware
            layout.addRow("<big><b>Hardware information</b></big>", qtw.QLabel(""))
            layout.addRow("Platform:", qtw.QLabel(platfrm))
            layout.addRow("Node name:", qtw.QLabel(node_name))
            layout.addRow("Platform-release:", qtw.QLabel(platform_release))
            layout.addRow("Platform-version:", qtw.QLabel(platform_version))
            layout.addRow("Architecure:", qtw.QLabel(architecture))
            layout.addRow("Processor:", qtw.QLabel(processor))
            layout.addRow("CPU core count:", qtw.QLabel(cpu_count))
            layout.addRow(
                "Amount of RAM:",
                qtw.QLabel(ram),
            )
            # Network
            layout.addRow("<big><b>Network information</b></big>", qtw.QLabel(""))
            layout.addRow("Hostname:", qtw.QLabel(hostname))
            layout.addRow("IP Address:", qtw.QLabel(ip_address))
            layout.addRow("Mac Address:", qtw.QLabel(mac_address))
            # GPU
            layout.addRow("<big><b>GPU information</b></big>", qtw.QLabel(""))

            self.setLayout(layout)

            # Try getting GPU information
            try:
                gpus = GPUtil.getGPUs()
                gpus_info = []

                for gpu in gpus:
                    total_memory = f"{gpu.memoryTotal}MB"

                    gpus_info.append(
                        {
                            "name": gpu.name,
                            "id": gpu.id,
                            "uuid": gpu.uuid,
                            "total_memory": total_memory,
                        }
                    )

                tabwidget = qtw.QTabWidget()
                # Create QFormLayouts for each GPU
                for info in gpus_info:
                    l = qtw.QFormLayout()
                    l.addRow("GPU name:", qtw.QLabel(info["name"]))
                    l.addRow("GPU id:", qtw.QLabel(str(info["id"])))
                    l.addRow("GPU uuid:", qtw.QLabel(info["uuid"]))
                    l.addRow("Total memory:", qtw.QLabel(str(info["total_memory"])))

                    widget = qtw.QWidget()
                    widget.setLayout(l)
                    # Add layout to QTabWidget
                    tabwidget.addTab(widget, info["name"])

                # Add tabwidget to layout
                layout.addWidget(tabwidget)

            except Exception as e:
                # Failed to load GPU information (display error)
                layout.addRow("Failed to load GPU information", qtw.QLabel(""))

                log.error("Failed to detect/show GPU information")
                log.error(str(e))
                traceback.print_exc()

        except Exception as e:
            # Failed to load PC data
            layout = qtw.QVBoxLayout()
            layout.addWidget(
                qtw.QLabel(
                    "Something went wrong while getting your PC's information..."
                )
            )
            log.error(str(e))
            self.setLayout(layout)

        self.exec_()
