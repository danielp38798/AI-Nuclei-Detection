# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# IMPORT PACKAGES AND MODULES
# ///////////////////////////////////////////////////////////////
import os
from PySide6 import QtWidgets
from gui.core.inference import *
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QDialog, QCheckBox, QProgressBar
from PySide6.QtCore import QThread, Signal
import json
from PySide6.QtCore import Signal as pyqtSignal


# APP FUNCTIONS
# ///////////////////////////////////////////////////////////////
class Functions:

    # SET SVG ICON
    # ///////////////////////////////////////////////////////////////
    @staticmethod
    def set_svg_icon(icon_name: str) -> str:
        """
        Get the full path of an SVG icon.

        Args:
            icon_name (str): The name of the SVG icon file.

        Returns:
            str: The full path to the SVG icon file.
        """
        app_path = os.path.abspath(os.getcwd())
        folder = "gui/images/svg_icons/"
        path = os.path.join(app_path, folder)
        icon = os.path.normpath(os.path.join(path, icon_name))
        return icon

    # SET SVG IMAGE
    # ///////////////////////////////////////////////////////////////
    @staticmethod
    def set_svg_image(icon_name: str) -> str:
        """
        Get the full path of an SVG image.

        Args:
            icon_name (str): The name of the SVG image file.

        Returns:
            str: The full path to the SVG image file.
        """
        app_path = os.path.abspath(os.getcwd())
        folder = "gui/images/svg_images/"
        path = os.path.join(app_path, folder)
        icon = os.path.normpath(os.path.join(path, icon_name))
        return icon

    # SET IMAGE
    # ///////////////////////////////////////////////////////////////
    @staticmethod
    def set_image(image_name: str) -> str:
        """
        Get the full path of an image.

        Args:
            image_name (str): The name of the image file.

        Returns:
            str: The full path to the image file.
        """
        app_path = os.path.abspath(os.getcwd())
        folder = "gui/images/images/"
        path = os.path.join(app_path, folder)
        image = os.path.normpath(os.path.join(path, image_name))
        return image


# IMPORTER CLASS TO IMPORT IMAGES
# ///////////////////////////////////////////////////////////////
import os

class ImportThread(QThread):
    progress_signal = Signal(int)  # Signal to send progress updates
    finished_signal = Signal()  # Signal to indicate that the import is finished
    status_signal = Signal(str)  # Signal to send status updates

    def __init__(self, import_function: callable):
        """
        Initialize the ImportThread.

        Args:
            import_function (callable): The function that performs the import.
        """
        super().__init__()
        self.import_function = import_function  # The function that performs the import

    def run(self) -> None:
        """
        Run the import process and emit signals for progress and status.

        Emits:
            progress_signal (int): The progress of the import.
            status_signal (str): The status of the import.
            finished_signal (): Indicates whether the import is finished.
        """
        # Call the import function and emit progress and status updates
        for progress, status in self.import_function():
            self.progress_signal.emit(progress)
            self.status_signal.emit(status)  # Emit the status signal
        self.finished_signal.emit()  # Emit the finished signal when the import is done

    

# ANALYSIS THREAD TO RUN INFERENCE
# ///////////////////////////////////////////////////////////////
class AnalysisThread(QThread):
    progress_signal = Signal(int)  # Signal to send progress updates
    finished_signal = Signal(bool)  # Signal to indicate that the import is finished
    status_signal = Signal(str)  # Signal to send status updates
    image_num_signal = Signal(int)  # Signal to send the image number
    stop_process_signal = Signal()  # Signal to stop the process
    time_per_image_signal = Signal(float)  # Signal to send the time per image

    def __init__(self, parent: QWidget, inference_function: callable):
        """
        Initialize the AnalysisThread.

        Args:
            parent (QWidget): The parent widget.
            inference_function (callable): The function that performs the inference.
        """
        super(AnalysisThread, self).__init__(parent)
        self.parent = parent
        self.inference_function = inference_function
        self._stop_process = False

    def stop_process(self) -> None:
        """
        Stop the analysis process.
        """
        self._stop_process = True
        self.stop_process_signal.emit()
        self.wait()  # Wait for the thread to finish
        print("Stopped Analysis Thread")

    def run(self) -> None:
        """
        Run the analysis process and emit signals for progress, status, and image number.
        """
        process = 0
        for image_num, process, status, results_dict, time_per_image in self.inference_function():
            if self._stop_process:  # Check if the stop process signal has been emitted
                print("Stopping process")
                self.finished_signal.emit(False)
                break

            self.progress_signal.emit(process)
            self.status_signal.emit(status)
            self.image_num_signal.emit(image_num)
            self.time_per_image_signal.emit(time_per_image)

        if process == 100:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)




# StoreResultsThread
# ///////////////////////////////////////////////////////////////
class StoreResultsThread(QThread):
    progress_signal = Signal(int)  # Signal to send progress updates
    finished_signal = Signal(bool)  # Signal to indicate that the import is finished
    status_signal = Signal(str)  # Signal to send status updates

    def __init__(self, parent: QWidget, store_results_function: callable):
        """
        Initialize the StoreResultsThread.

        Args:
            parent (QWidget): The parent widget.
            store_results_function (callable): The function that performs the storing of results.
        """
        super(StoreResultsThread, self).__init__(parent)
        self.parent = parent
        self.store_results_function = store_results_function

    def run(self) -> None:
        """
        Run the store results process and emit signals for progress and status.
        """
        for process, status in self.store_results_function:
            self.progress_signal.emit(process)
            self.status_signal.emit(status)

        if process == 100:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)


# GENERATE REPORT THREAD
# ///////////////////////////////////////////////////////////////
class GenerateReportThread(QThread):
    success_value_signal = Signal(bool)  # Signal to send progress updates
    finished_signal = Signal(bool)  # Signal to indicate that the import is finished
    status_signal = Signal(str)  # Signal to send status updates
    progress_signal = Signal(int)  # Signal to send progress updates
    failed_trials_signal = Signal(list)  # Signal to send the failed trials


    def __init__(self, parent: QWidget, generate_report_function: callable):
        """
        Initialize the GenerateReportThread.

        Args:
            parent (QWidget): The parent widget.
            generate_report_function (callable): The function that generates the report.
        """
        super(GenerateReportThread, self).__init__(parent)
        self.parent = parent
        self.generate_report_function = generate_report_function
        self.failed_trials = []

    def run(self) -> None:
        """
        Run the report generation process and emit signals for progress, status, and failed trials.

        Emits:
            progress_signal (int): The progress of the report generation.
            status_signal (str): The status of the report generation.
            failed_trials_signal (str): The name of the failed trial.
            finished_signal (bool): Indicates whether the report generation is finished successfully.
        """
        # 100, f"Report for trial {trial_name} could not be generated", trial_name, False
        for progress, status, trial_name, success in self.generate_report_function:
            self.progress_signal.emit(progress)
            self.status_signal.emit(status)
            if success == False:
                self.failed_trials.append(trial_name)
                self.failed_trials_signal.emit(self.failed_trials)
                self.finished_signal.emit(False)
                #break
        self.finished_signal.emit(True)
            
            


# EXPORT THREAD
# ///////////////////////////////////////////////////////////////
class ExportThread(QThread):
    progress_signal = Signal(int)  # Signal to send progress updates
    finished_signal = Signal(bool)  # Signal to indicate that the import is finished
    status_signal = Signal(str)  # Signal to send status updates

    def __init__(self, parent: QWidget, export_function: callable):
        """
        Initialize the ExportThread.

        Args:
            parent (QWidget): The parent widget.
            export_function (callable): The function that performs the export.
        """
        super(ExportThread, self).__init__(parent)
        self.parent = parent
        self.export_function = export_function

    def run(self) -> None:
        """
        Run the export process and emit signals for progress and status.
        """
        for progress, status in self.export_function:
            self.progress_signal.emit(progress)
            self.status_signal.emit(status)

        if progress == 100:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)

    def stop(self) -> None:
        """
        Stop the export process.
        """
        self.terminate()
        self.finished_signal.emit(False)
        self.status_signal.emit("Export process stopped.")
        self.progress_signal.emit(0)
        
class DeleteThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to send progress updates
    finished_signal = pyqtSignal(bool)  # Signal to indicate that the import is finished
    status_signal = pyqtSignal(str)  # Signal to send status updates

    def __init__(self, parent: QWidget, delete_function: callable):
        """
        Initialize the DeleteThread.

        Args:
            parent (QWidget): The parent widget.
            delete_function (callable): The function that performs the delete operation.
        """
        super(DeleteThread, self).__init__(parent)
        self.parent = parent
        self.delete_function = delete_function

    def run(self) -> None:
        """
        Run the delete process and emit signals for progress and status.
        """
        for progress, status in self.delete_function:
            self.progress_signal.emit(progress)
            self.status_signal.emit(status)

        if progress == 100:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)

    def stop(self) -> None:
        """
        Stop the delete process.
        """
        self.terminate()
        self.finished_signal.emit(False)
        self.status_signal.emit("Delete process stopped.")
        self.progress_signal.emit(0)

# TRAINING THREAD
# ///////////////////////////////////////////////////////////////
class TrainThread(QThread):
    time_left_signal = pyqtSignal(int)  # Signal to send progress updates
    finished_signal = pyqtSignal(bool)  # Signal to indicate that the import is finished
    status_signal = pyqtSignal(str)  # Signal to send status updates

    def __init__(self, parent: QWidget, train_function: callable):
        """
        Initialize the TrainThread.

        Args:
            parent (QWidget): The parent widget.
            train_function (callable): The function that performs the training.
        """
        super(TrainThread, self).__init__(parent)
        self.parent = parent
        self.train_function = train_function

    def run(self) -> None:
        """
        Run the training process and emit signals for time left and status.

        Emits:
            time_left_signal (int): The time left for the training to complete.
            status_signal (str): The status of the training process.
            finished_signal (bool): Indicates whether the training is finished successfully.
        """
        for time_left, status in self.train_function:
            self.time_left_signal.emit(time_left)
            self.status_signal.emit(status)

        if time_left == 0:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)

    def stop(self) -> None:
        """
        Stop the training process.
        """
        self.terminate()
        self.finished_signal.emit(False)
        self.status_signal.emit("Training process stopped.")
        self.time_left_signal.emit(0)

# OPEN FILE THREAD
# ///////////////////////////////////////////////////////////////
class OpenFileThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to send progress updates
    finished_signal = pyqtSignal(bool)  # Signal to indicate that the import is finished
    status_signal = pyqtSignal(str)  # Signal to send status updates

    def __init__(self, parent: QWidget, open_file_function: callable):
        """
        Initialize the OpenFileThread.

        Args:
            parent (QWidget): The parent widget.
            open_file_function (callable): The function that performs the file opening.
        """
        super(OpenFileThread, self).__init__(parent)
        self.parent = parent
        self.open_file_function = open_file_function

    def run(self) -> None:
        """
        Run the file opening process and emit signals for progress and status.

        Emits:
            progress_signal (int): The progress of the file opening.
            status_signal (str): The status of the file opening.
            finished_signal (bool): Indicates whether the file opening is finished successfully.
        """
        for progress, status in self.open_file_function:
            self.progress_signal.emit(progress)
            self.status_signal.emit(status)

        if progress == 100:
            self.finished_signal.emit(True)
        else:
            self.finished_signal.emit(False)

    def stop(self) -> None:
        """
        Stop the file opening process.
        """
        self.terminate()
        self.finished_signal.emit(False)
        self.status_signal.emit("Open file process stopped.")
        self.progress_signal.emit(0)