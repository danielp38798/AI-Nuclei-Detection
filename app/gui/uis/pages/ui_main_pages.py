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

# IMPORT QT CORE
# ///////////////////////////////////////////////////////////////
from qt_core import *
from gui.core.functions import Functions
import os
import PySide6.QtWidgets as QtWidgets
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QMovie
from PySide6.QtCore import Signal, QObject, Qt
from PySide6.QtWidgets import QMainWindow, QTextEdit, QWidget
from gui.widgets import PyPushButton
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Patch
import numpy as np
import time

# IMPORT THEME COLORS
# ///////////////////////////////////////////////////////////////
from gui.core.json_themes import Themes

# IMPORT WINDOW CLASS
# ///////////////////////////////////////////////////////////////
from gui.widgets.py_window.py_window import PyWindow
from gui.widgets.py_window.styles import Styles

# ANALYSIS Thread
# ///////////////////////////////////////////////////////////////
from gui.core.functions import ImportThread, AnalysisThread, StoreResultsThread, GenerateReportThread, ExportThread, DeleteThread, TrainThread
from gui.core.inference.ModelInterface import ModelInteractor

# IMPORT Thread
# ///////////////////////////////////////////////////////////////
import shutil
import cv2
from gui.core.inference.ImageTools import ImageCropper
from gui.uis.windows.main_window.functions_main_window import *


# RESULTS VIEWER
from gui.core.reporting.reporting import Reporting, Report
from PIL import Image

# IMPORT SETTINGS
# ///////////////////////////////////////////////////////////////
from gui.core.json_settings import Settings

# custom progress bar
from gui.widgets.py_progress_bar import PYProgressBar
from gui.widgets.py_combo_box.py_combo_box import PyComboBox

# TRAINING
#from gui.core.trainer.trainer import Trainer

# nuclei tool box
#from gui.core.nuclei_annotation_toolbox.nuclei_data_toolbox import NucleiAnnotationToolbox


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Generator, Tuple, List

def convert_str_to_bool(value: str) -> bool:
    """
    Convert a string representation of a boolean value to a boolean.

    Args:
        value (str): The string representation of the boolean value.

    Returns:
        bool: The boolean value corresponding to the input string.
    """
    if value == "True" or value == "true" or value == True:
        return True
    else:
        return False


def get_base_path():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundled executable, the PyInstaller
        # bootloader sets a sys._MEIPASS attribute to the path of the temp folder it
        # extracts its bundled files to.
        return sys._MEIPASS
    else:
        # Otherwise, just use the directory of the script being run
        return os.path.dirname(os.path.abspath(__file__))



class ConsoleStream(QObject):
    """ A class to redirect stdout and stderr to a QTextEdit widget. """
    new_text = Signal(str)

    def __init__(self, text_edit_widget: QTextEdit):
        """
        Initialize the ConsoleStream object.

        Args:
            text_edit_widget (QTextEdit): The QTextEdit widget to display the console output.
        """
        super().__init__()
        self.text_edit_widget = text_edit_widget
        self.new_text.connect(self.on_new_text)

    def write(self, text: str) -> None:
        """
        Emit the new text signal.

        Args:
            text (str): The text to emit.
        """
        self.new_text.emit(str(text))

    def flush(self) -> None:
        """ Flush method to comply with file-like object standards (no-op). """
        pass

    def on_new_text(self, text: str) -> None:
        """
        Append the new text to the QTextEdit widget.

        Args:
            text (str): The text to append.
        """
        self.text_edit_widget.append(text)
        self.text_edit_widget.verticalScrollBar().setValue(self.text_edit_widget.verticalScrollBar().maximum())  # Auto-scroll to bottom

class ConsoleWindow(QMainWindow):
    """ A separate window for displaying the console output. """
    def __init__(self):
        """
        Initialize the ConsoleWindow object.
        """
        super().__init__()
        self.setWindowTitle("Console Output")
        self.setGeometry(300, 300, 600, 400)

        # QTextEdit to show console output
        self.console_output = QTextEdit(self)
        self.console_output.setReadOnly(True)
        self.setCentralWidget(self.console_output) # Set the QTextEdit as the central widget

        # Redirect stdout and stderr to the console window's QTextEdit
        self.console_stream = ConsoleStream(self.console_output)
        sys.stdout = self.console_stream
        sys.stderr = self.console_stream

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Restore sys.stdout and sys.stderr when the window is closed.

        Args:
            event (QCloseEvent): The close event.
        """
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Close the window when the Escape key is pressed.

        Args:
            event (QKeyEvent): The key press event.
        """
        if event.key() == Qt.Key_Escape:
            self.close()

class ResultsViewer(QWidget):
    """
    A class to view the generated plots containing the predictions and class labels.
    """
    def __init__(self, parent: QWidget, trial_path: str, update_ui: bool = True):
        """
        Initialize the ResultsViewer object.

        Args:
            parent (QWidget): The parent widget.
            trial_path (str): The path to the trial directory.
            update_ui (bool): Whether to update the UI. Defaults to True.
        """
        super().__init__(parent)

        self.trial_path = trial_path
        self.aux_files_dir = os.path.join(trial_path, "_aux_files")
        self.image_analysis_dir = os.path.join(trial_path, "01_image_analysis")
        self.prediction_dir = os.path.join(self.image_analysis_dir, "02_model_predictions")
        self.plots_dir = self.prediction_dir
        self.clusters_dir = os.path.join(self.image_analysis_dir, "05_clusters_results")
        self.json_files_path = os.path.join(self.trial_path, "_aux_files")

        self.themes = Themes().items
        self.settings = Settings().items
        self.prediction_images = []
        self.json_files = []
        self.color_dict = {}
        self.category_name_dict = {}
        self.figsize = (20, 12)

        self.model_name = self.settings["processing_settings"]["model_selection"]
        if "maskrcnn_resnet50_c4" in self.model_name or "Small Model (R_50_C4)" in self.model_name:
            self.model_name = "maskrcnn_resnet50_c4"
        elif "maskrcnn_resnet101_dc5" in self.model_name or "Accurate Model (R_101_DC5)" in self.model_name:
            self.model_name = "maskrcnn_resnet101_dc5"
        else:
            print("Model not found")

        if update_ui:
            self.central_widget = QWidget()
            self.results_page_layout = QVBoxLayout()
            
            self.current_index = 0 # store the current index of the image being processed
            self.stop_analysis_index = 0 # store the index of the image where the analysis was stopped

            self.figure = None
            self.fig = plt.figure(figsize=self.figsize)
            ax = self.fig.add_subplot(111)  
            self.canvas = FigureCanvas(self.fig)
            self.canvas.draw()
            self.toolbar = NavigationToolbar(self.canvas, self, False)
            
            self.results_page_layout.addWidget(self.canvas, 0, Qt.AlignCenter)  # Add the canvas to the results_page_layout
            self.results_page_layout.addWidget(self.toolbar, 0, Qt.AlignCenter)  # Add the toolbar to the results_page_layout
            
            self.index_label = QLabel()
            self.index_label.setAlignment(Qt.AlignCenter)
            self.results_page_layout.addWidget(self.index_label)

            if self.settings["language"] == "eng":
                text = "Next"
            elif self.settings["language"] == "de":
                text = "Nächste"
            self.next_button = PyPushButton(
                parent=self,
                text=text,
                radius=8,
                color=self.themes["app_color"]["text_foreground"],
                bg_color=self.themes["app_color"]["dark_one"],
                bg_color_hover=self.themes["app_color"]["dark_three"],
                bg_color_pressed=self.themes["app_color"]["dark_four"]
            )
            self.next_button.clicked.connect(self.increment_figure_num)

            if self.settings["language"] == "eng":
                text = "Previous"
            elif self.settings["language"] == "de":
                text = "Vorherige"
            self.previous_button = PyPushButton(
                parent=self,
                text=text,
                radius=8,
                color=self.themes["app_color"]["text_foreground"],
                bg_color=self.themes["app_color"]["dark_one"],
                bg_color_hover=self.themes["app_color"]["dark_three"],
                bg_color_pressed=self.themes["app_color"]["dark_four"]
            )
            self.previous_button.clicked.connect(self.decrement_figure_num)

            # Checkbox to toggle between prediction and clusters views
            self.show_clusters_cb = QtWidgets.QCheckBox("Show clusters")
            self.show_clusters_cb.setChecked(False)
            self.show_clusters_cb.toggled.connect(self.toggle_clusters_view)
            

            button_layout = QHBoxLayout()
            button_layout.addWidget(self.show_clusters_cb)
            button_layout.addWidget(self.previous_button)
            button_layout.addWidget(self.next_button)

            self.results_page_layout.addLayout(button_layout)
            self.results_page_layout.setAlignment(Qt.AlignCenter)
            self.setLayout(self.results_page_layout)
            # Lists for prediction and clusters images
            self.clusters_images = []
            self.view_mode = "prediction"  # or 'clusters'

            self.check_prediction_data()

    def load_json(self, file_path: str) -> dict:
        """
        Load a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The loaded JSON data.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    def check_prediction_data(self) -> None:
        """
        Check if there are any prediction images to display.
        """
        results_dir = os.path.join(self.trial_path, "_aux_files")
        #prediction_dir = os.path.join(results_dir, self.image_analysis_dir, "02_model_predictions")
        #clusters_dir = os.path.join(results_dir, self.image_analysis_dir, "05_clusters_results")
        #print(f"Checking for prediction data in: {self.prediction_dir}")
        if not os.path.exists(self.prediction_dir):
            print("No data to display")
            self.close()
            return
        else:
            self.prediction_images = []
            self.clusters_images = []
            instances_results_path = os.path.join(self.aux_files_dir, "image_analysis_results.json")
            if os.path.exists(instances_results_path):
                instances_results = self.load_json(instances_results_path)
                #print(f"Loaded instances results from: {instances_results_path}")
                #print(f"Instances results: {instances_results}")
                image_data = instances_results["image_data"]
                for image in image_data:
                    if image_data[image] != {}:
                        image = image.split(".")[0]
                        # prediction image
                        pred_file = os.path.join(self.prediction_dir, f"{image}_prediction.png")
                        if os.path.exists(pred_file) and pred_file not in self.prediction_images:
                            self.prediction_images.append(pred_file)
                        # clusters image (optional)
                        clusters_file = os.path.join(self.clusters_dir, f"{image}_clusters.png")
                        if os.path.exists(clusters_file) and clusters_file not in self.clusters_images:
                            self.clusters_images.append(clusters_file)
            #print(f"Prediction images found: {len(self.prediction_images)}")
            if len(self.prediction_images) == 0:
                print("No data to display")
                self.close()
            if len(self.prediction_images) == 1:
                self.next_button.setEnabled(False)
                self.previous_button.setEnabled(False)
            else:
                self.next_button.setEnabled(True)
                self.previous_button.setEnabled(True)
            
            # If clusters images are present, ensure view toggle is enabled
            if len(self.clusters_images) == 0:
                self.show_clusters_cb.setEnabled(False)
            else:
                self.show_clusters_cb.setEnabled(True)

            self.show_figure()
            
    def get_plots_to_generate(self) -> int:
        """
        Get the number of plots to generate.

        Returns:
            int: The number of plots to generate.
        """
        results_dir = self.trial_path
        #prediction_dir = os.path.join(results_dir, "predictions")
        prediction_dir = self.prediction_dir
        if not os.path.exists(prediction_dir):
            return 0
        else:
            prediction_images = [os.path.join(prediction_dir, file) for file in os.listdir(prediction_dir) if file.endswith(".png")]
            return len(prediction_images)

    def get_instances_color_category_json_files(self) -> tuple[dict, dict]:
        """
        Get the color and category name dictionaries from JSON files.

        Returns:
            tuple[dict, dict]: The color and category name dictionaries.
        """
        json_files_path = self.json_files_path
        self.json_files = [file for file in os.listdir(json_files_path) if file.endswith(".json")]
        self.color_dict = {}
        colors =  [[255, 112,31], [44,153, 168]]

        self.color_dict = {i: color for i, color in enumerate(colors)}

        list_of_files = os.listdir(json_files_path)

        if "color_dict.json" in list_of_files:
            color_dict_path = os.path.join(json_files_path, "color_dict.json")
            self.color_dict = self.load_json(color_dict_path)
        if "category_name_dict.json" in list_of_files:
            category_name_dict_path = os.path.join(json_files_path, "category_name_dict.json")
            self.category_name_dict = self.load_json(category_name_dict_path)
        
        self.color_dict = {k: self.color_dict[k] for k in sorted(self.color_dict)}
        self.category_name_dict = {k: self.category_name_dict[k] for k in sorted(self.category_name_dict)}

        return self.color_dict, self.category_name_dict
    
    def generate_plot(self, image_file: str) -> plt.Figure:
        """
        Generates a plot for the given image file.

        Args:
            image_file (str): The path to the image file for which the plot is to be generated.

        Returns:
            plt.Figure: The generated plot as a matplotlib Figure object.
        """
        self.settings = Settings().items
        # Load the color dict and category name dict
        self.color_dict, self.category_name_dict = self.get_instances_color_category_json_files()
        
        if not self.color_dict or not self.category_name_dict:
            if self.settings["language"] == "eng":
                print("Error generating plot. Data is being written. Please wait a few seconds and try again.")
            else:
                print("Fehler beim Generieren des Diagramms. Möglicherweise keine Instanzen erkannt.")
            return
        
        handles = [Patch(color=np.array(self.color_dict[category_id])/255, label=category_name) for category_id, category_name in self.category_name_dict.items()]
        image = plt.imread(image_file)
        # Determine the size of the image
        image_size = image.shape
        # Adapt the size of the plot to the size of the image
        adapted_figsize = image_size
        
        # Clear previous plot
        if hasattr(self, "fig"):
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=adapted_figsize)
        
        ax = self.fig.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")

        if self.settings["language"] == "eng":
            title = "Classes"
        elif self.settings["language"] == "de":
            title = "Klassen"
        
        legend_1 = ax.legend(title=title, handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
        ax.add_artist(legend_1)

        # Get unit from settings.items
        unit = self.settings["processing_settings"]["unit"]
        hyperparameters_path = os.path.join(self.trial_path, "_aux_files", "hyperparameters.json")
        reporting = Reporting(unit=unit, 
                                trial_dir=self.trial_path, 
                                hp_params_path=hyperparameters_path,
                                language=self.settings["language"],
                                verbose=False)
        image_file_name = os.path.basename(image_file).replace("_prediction.png", ".tif")

        if reporting.check_results_file_integrity():
            # For every class, get the count of the class and add it to the legend
            count_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
            used_model = reporting.get_used_model_for_analysis()

            for class_id, class_name in self.category_name_dict.items():
                #if "yolo" in used_model:
                #    class_id = int(class_id) + 1 
                #    class_id = str(class_id)
                count_of_classname = reporting.get_count_for_class(image_file_name, class_id)
                #if "yolo" in used_model:
                #    class_id = int(class_id) - 1
                #    class_id = str(class_id)
                if count_of_classname > 0:
                    count_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{count_of_classname}")
        
            if self.settings["language"] == "eng":
                text = "Counts"
            elif self.settings["language"] == "de":
                text = "Zahlen"
            count_legend = ax.legend(title=text, handles=count_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.8), fontsize="small")
            ax.add_artist(count_legend)

            # For every class, get the area taken up by the class and add it to the legend
            area_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
            for class_id, class_name in self.category_name_dict.items():
                #if "yolo" in used_model:
                #   class_id = int(class_id) + 1 
                #    class_id = str(class_id)
                area_of_classname = reporting.get_area_for_class(image_file_name, class_id)
                #if "yolo" in used_model:
                #    class_id = int(class_id) - 1
                #    class_id = str(class_id)
                if area_of_classname > 0:
                    if unit == "mm":
                        area_of_classname = round(area_of_classname, 4)
                    else:
                        area_of_classname = round(area_of_classname, 2)
                    area_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{area_of_classname}")

            if self.settings["language"] == "eng":
                text = f"Area ({unit}^2)"
            elif self.settings["language"] == "de":
                text = f"Fläche ({unit}^2)"
            area_legend = ax.legend(title=text, handles=area_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.6), fontsize="small")
            ax.add_artist(area_legend)

            # For every class, get the relative area taken up by the class and add it to the legend
            relative_area_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
            for class_id, class_name in self.category_name_dict.items():
                #if "yolo" in used_model:
                #    class_id = int(class_id) + 1 
                #    class_id = str(class_id)
                relative_area_of_classname = reporting.get_relative_area_for_class(image_file_name, class_id)
                #if "yolo" in used_model:
                #    class_id = int(class_id) - 1
                #    class_id = str(class_id)
                if relative_area_of_classname > 0:
                    if unit == "mm":
                        relative_area_of_classname = round(relative_area_of_classname, 4)
                    else:
                        relative_area_of_classname = round(relative_area_of_classname, 2)
                    relative_area_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{relative_area_of_classname}")

            if self.settings["language"] == "eng":
                text = "Relative area (%)"
            elif self.settings["language"] == "de":
                text = "Relative Fläche (%)"
            relative_area_legend = ax.legend(title=text, handles=relative_area_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.4), fontsize="small")
            ax.add_artist(relative_area_legend)

            # For every class, get the density of the class and add it to the legend
            density_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
            for class_id, class_name in self.category_name_dict.items():
                #if "yolo" in used_model:
                #    class_id = int(class_id) + 1 
                #    class_id = str(class_id)
                density_of_classname = reporting.get_density_for_class(image_file_name, class_id)
                #if "yolo" in used_model:
                #    class_id = int(class_id) - 1
                #    class_id = str(class_id)
                if density_of_classname > 0:
                    if unit == "mm":
                        density_of_classname = round(density_of_classname, 4)
                    else:
                        density_of_classname = density_of_classname
                    density_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{density_of_classname}")

            if self.settings["language"] == "eng":
                text = f"Density (1/{unit}^2)"
            elif self.settings["language"] == "de":
                text = f"Dichte ({1/unit}^2)"
            density_legend = ax.legend(title=text, handles=density_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.2), fontsize="small")
            ax.add_artist(density_legend)
            
        else:
            print("Error: Unable to get results data")
        
        return self.fig
    
    def store_results(self) -> Generator[Tuple[int, str], None, None]:
        """
        Generate and store all the plots for the images in the trial.

        This function generates plots for each prediction image in the trial directory.
        It includes various legends such as class counts, area, relative area, and density.
        The generated plots are saved in the 'plots' directory within the trial directory.
        
        Yields:
            Tuple[int, str]: The progress value and status message.
        """
    
        prediction_dir = self.prediction_dir
        if not os.path.exists(prediction_dir):
            print("No figures to display")
        else:
            prediction_images = [os.path.join(prediction_dir, file) for file in os.listdir(prediction_dir) if file.endswith("_prediction.png")]

            if len(prediction_images) > 0:
                plots_dir = self.plots_dir
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)

                for idx, image_file in enumerate(prediction_images):
                    plot_file = image_file.replace(".png", "_plot.png")
                    image = plt.imread(image_file)
                    fig = plt.figure(figsize=self.figsize)
                    settings = Settings().items
                    color_dict, category_name_dict = self.get_instances_color_category_json_files()
                    if not color_dict or not category_name_dict:
                        if settings["language"] == "eng":
                            print("Error generating plot. Data is being written. Please wait a few seconds and try again.")
                        else:
                            print("Fehler beim Generieren des Diagramms. Möglicherweise keine Instanzen erkannt.")
                        return
                    handles = [Patch(color=np.array(color_dict[category_id])/255, label=category_name) for category_id, category_name in self.category_name_dict.items()]

                    ax = fig.add_subplot(111)
                    ax.imshow(image)
                    ax.axis("off")
                    if self.settings["language"] == "eng":
                        title = "Classes"
                    elif self.settings["language"] == "de":
                        title = "Klassen"
                    legend_1 = ax.legend(title=title, handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
                    ax.add_artist(legend_1)

                    unit = self.settings["processing_settings"]["unit"]
                    hyperparameters_path = os.path.join(self.trial_path, "_aux_files", "hyperparameters.json")
                    reporting = Reporting(unit=unit, 
                                        trial_dir=self.trial_path, 
                                        hp_params_path=hyperparameters_path,
                                        language=self.settings["language"],
                                        verbose=False)
                    image_file_name = os.path.basename(image_file).replace("_prediction.png", ".tif")

                    if reporting.check_results_file_integrity():
                        count_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
                        used_model = reporting.get_used_model_for_analysis()

                        for class_id, class_name in self.category_name_dict.items():
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) + 1
                            #    class_id = str(class_id)
                            count_of_classname = reporting.get_count_for_class(image_file_name, class_id)
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) - 1
                            #    class_id = str(class_id)
                            count_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{count_of_classname}")

                        if self.settings["language"] == "eng":
                            text = "Counts"
                        elif self.settings["language"] == "de":
                            text = "Zahlen"
                        count_legend = ax.legend(title=text, handles=count_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.8), fontsize="small")
                        ax.add_artist(count_legend)

                        area_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
                        for class_id, class_name in self.category_name_dict.items():
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) + 1
                            #   class_id = str(class_id)
                            area_of_classname = reporting.get_area_for_class(image_file_name, class_id)
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) - 1
                            #    class_id = str(class_id)
                            if area_of_classname > 0:
                                if unit == "mm":
                                    area_of_classname = round(area_of_classname, 4)
                                else:
                                    area_of_classname = round(area_of_classname, 2)
                            area_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{area_of_classname}")

                        if self.settings["language"] == "eng":
                            text = f"Area ({unit}^2)"
                        elif self.settings["language"] == "de":
                            text = f"Fläche ({unit}^2)"
                        area_legend = ax.legend(title=text, handles=area_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.6), fontsize="small")
                        ax.add_artist(area_legend)

                        relative_area_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
                        for class_id, class_name in self.category_name_dict.items():
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) + 1
                            #    class_id = str(class_id)
                            relative_area_of_classname = reporting.get_relative_area_for_class(image_file_name, class_id)
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) - 1
                            #    class_id = str(class_id)
                            if relative_area_of_classname > 0:
                                if unit == "mm":
                                    relative_area_of_classname = round(relative_area_of_classname, 4)
                                else:
                                    relative_area_of_classname = round(relative_area_of_classname, 2)
                            relative_area_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{relative_area_of_classname}")

                        if self.settings["language"] == "eng":
                            text = "Relative area (%)"
                        elif self.settings["language"] == "de":
                            text = "Relative Fläche (%)"
                        relative_area_legend = ax.legend(title=text, handles=relative_area_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.4), fontsize="small")
                        ax.add_artist(relative_area_legend)

                        density_legend_handles = {category_id: None for category_id in self.category_name_dict.keys()}
                        for class_id, class_name in self.category_name_dict.items():
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) + 1
                            #    class_id = str(class_id)
                            density_of_classname = reporting.get_density_for_class(image_file_name, class_id)
                            #if "yolo" in used_model:
                            #    class_id = int(class_id) - 1
                            #    class_id = str(class_id)
                            if density_of_classname > 0:
                                if unit == "mm":
                                    density_of_classname = round(density_of_classname, 4)
                                else:
                                    density_of_classname = density_of_classname
                            density_legend_handles[class_id] = Patch(color=np.array(self.color_dict[class_id])/255, label=f"{density_of_classname}")

                        if self.settings["language"] == "eng":
                            text = f"Density (1/{unit}^2)"
                        elif self.settings["language"] == "de":
                            text = f"Dichte ({1/unit}^2)"
                        density_legend = ax.legend(title=text, handles=density_legend_handles.values(), loc="upper left", bbox_to_anchor=(1, 0.2), fontsize="small")
                        ax.add_artist(density_legend)

                    plot_file = image_file.replace(".png", "_plot.png")
                    plot_path = os.path.join(plots_dir, os.path.basename(plot_file))
                    fig.savefig(plot_path)
                    plt.close(fig)
                    
                    progress = np.floor((idx + 1) / len(prediction_images) * 100)
                    yield progress, f"Generated plot for {os.path.basename(image_file).replace('_prediction.png', '.tif')}"
            else:
                print("No figures to display")



    def show_figure(self) -> None:
        """
        Display the current prediction image and update the index label.

        This method ensures that the current index is within the valid range,
        generates the plot for the current prediction image, and updates the
        index label with the current prediction number and file name.
        """
        # Avoid index out of range
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.prediction_images):
            self.current_index = len(self.prediction_images) - 1

        # Select the active image list based on view mode
        if self.view_mode == "prediction":
            images = self.prediction_images
            mode_label = "Prediction"
        else:
            images = self.clusters_images
            mode_label = "Clusters"

        # Avoid empty list
        if len(images) == 0:
            if self.settings["language"] == "eng":
                print(f"No {mode_label.lower()} images to display")
            else:
                print(f"Keine {mode_label.lower()} Bilder zum Anzeigen")
            return

        # Ensure index in range for the chosen list
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(images):
            self.current_index = len(images) - 1

        image_file = images[self.current_index]
        if not os.path.exists(image_file):
            print(f"File not found: {image_file}")
            return

        # If clusters image, simply load the PNG into the axes; otherwise, generate plot
        if self.view_mode == "prediction":
            self.generate_plot(image_file)
        else:
            # clusters image may already be an overlay PNG; show it directly
            img = plt.imread(image_file)
            if not hasattr(self, "fig"):
                self.fig = plt.figure(figsize=self.figsize)
            else:
                self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")

        # Update index label
        if self.settings["language"] == "eng":
            self.index_label.setText(f"{mode_label} {self.current_index + 1} of {len(images)} \n File: {os.path.basename(image_file)} \n")
        elif self.settings["language"] == "de":
            self.index_label.setText(f"{mode_label} {self.current_index + 1} von {len(images)} \n Datei: {os.path.basename(image_file)} \n")
        self.canvas.draw()

    def increment_figure_num(self) -> None:
        """
        Increment the current index and display the next prediction image.

        If there are no prediction images, a warning message is displayed.
        Otherwise, the current index is incremented and the next prediction
        image is displayed.
        """
        images = self.prediction_images if self.view_mode == "prediction" else self.clusters_images

        if len(images) == 0:
            if self.settings["language"] == "eng":
                QMessageBox.critical(self, "Warning", "No figures generated!", defaultButton=QMessageBox.Ok)
            elif self.settings["language"] == "de":
                QMessageBox.critical(self, "Warnung", "Keine Diagramme generiert!", defaultButton=QMessageBox.Ok)
            print("No figures generated")
        else:
            if self.current_index < len(images) - 1:
                self.current_index += 1
                self.show_figure()

    def decrement_figure_num(self) -> None:
        """
        Decrement the current index and display the previous prediction image.

        If there are no prediction images, a warning message is displayed.
        Otherwise, the current index is decremented and the previous prediction
        image is displayed.
        """
        images = self.prediction_images if self.view_mode == "prediction" else self.clusters_images

        if len(images) == 0:
            if self.settings["language"] == "eng":
                QMessageBox.critical(self, "Warning", "No figures generated!", defaultButton=QMessageBox.Ok)
            elif self.settings["language"] == "de":
                QMessageBox.critical(self, "Warnung", "Keine Diagramme generiert!", defaultButton=QMessageBox.Ok)
            print("No figures generated")
        else:
            if self.current_index > 0:
                self.current_index -= 1
                self.show_figure()

    def toggle_clusters_view(self, checked: bool) -> None:
        """
        Toggle between prediction and clusters image views.

        Args:
            checked (bool): True if clusters view enabled, False otherwise.
        """
        self.view_mode = "clusters" if checked else "prediction"
        # Reset index to 0 for the new view to avoid out-of-range
        self.current_index = 0
        self.show_figure()

               

# MAIN PAGES
class Ui_MainPages(object):
    """
    A class to set up the main pages of the application.
    """

    def __init__(self, ui_object):
        """
        Initialize the Ui_MainPages class.

        Args:
            ui_object (object): The UI object to be initialized.
        """
        self.ui = ui_object

    def setupUi(self, MainPages: QWidget, base_path: str) -> None:
        """
        Set up the UI for the main pages.

        Args:
            MainPages (QWidget): The main pages widget.
            base_path (str): The base path for the application resources.
        """
        if not MainPages.objectName():
            MainPages.setObjectName(u"MainPages")
        
        # Track the visibility state of the console
        self.console_visible = True

        self.analysis_thread_should_stop = False
        self.base_path = base_path

        # Load settings
        settings = Settings()
        self.settings = settings.items
        model_name = self.settings["processing_settings"]["model_selection"]

        # Read the model info from ./application_resources/
        model_info_path = os.path.join(self.base_path, "application_resources", "models", "model_info.json")
        with open(model_info_path, "r") as file:
            self.model_info = json.load(file)

        # Check if the model name is in the model info
        if "maskrcnn_resnet50_c4" in model_name or "Small Model (R_50_C4)" in model_name:
            model_name = "maskrcnn_resnet50_c4"
        elif "maskrcnn_resnet101_dc5" in model_name or "Accurate Model (R_101_DC5)" in model_name:
            model_name = "maskrcnn_resnet101_dc5"

        else:
            print("Model not found") 
        
        if model_name in self.model_info:
            self.model_name = model_name
            self.instances_dict = self.model_info[model_name]["instances"]
            # Convert "1" to int(1) e.g. for the key "1" in the instances_dict
            self.instances_dict = {int(key): value for key, value in self.instances_dict.items()}
        else:
            print("Model not found")

        self.images_file_paths = []  # Store the complete paths to the images
        self.imported_files_dir = os.path.join(self.base_path, "imports")  # Store the directory where the images are imported
        self.imported_trial_paths = []  # Store the paths to the imported trials
        self.trial_name = ""  # Store the name of the trial
        self.current_trial_path = ""  # Store the path to the current trial
        self.selected_trial = ""  # Store the selected trial

        self.results_dir = ""  # Store the directory where the results are stored
        self.results_trial_paths = []  # Store the paths to the results trials
        self.results_trial_names = []
        self.results_trial_names_dict = {}
        self.results_dict = {}  # Store the results of the analysis

        self.total_images = 0
        self.prediction_images = []  # Store the paths to the prediction images
        self.current_image_index = 0
        self.convert_img_to_8bit = True
        self.import_process_should_stop = False

        self.physical_image_width = self.settings["processing_settings"]["image_size"]
        self.failed_trials = []  # Store the trials that failed to generate predictions

        self.themes = Themes().items
        self.main_pages_layout = QVBoxLayout(MainPages)
        self.main_pages_layout.setSpacing(0)
        self.main_pages_layout.setObjectName(u"main_pages_layout")
        self.main_pages_layout.setContentsMargins(5, 5, 5, 5)
        self.pages = QStackedWidget(MainPages)
        self.pages.setObjectName(u"pages")
        
        # Setup pages
        self.setup_welcome_page()
        self.setup_import_page()
        self.setup_analysis_page()
        self.setup_results_page()
        self.setup_datamanagement_page()

        self.retranslateUi(MainPages)
        self.pages.setCurrentIndex(0)
        self.main_pages_layout.addWidget(self.pages)
        QMetaObject.connectSlotsByName(MainPages)


    # ------------------------------ GUI HELPER FUNCTIONS ------------------------------ #

    # SETUP PAGES FUNCTIONS
    def setup_widgets_page(self) -> None:
        """
        Set up the widgets page.

        This function initializes and configures the widgets page, including the layout, scroll area,
        and various UI elements such as labels and layouts.
        """
        self.pages.addWidget(self.welcome_page)
        self.widgets_page = QWidget()
        self.widgets_page.setObjectName(u"widgets_page")
        self.widgets_page_layout = QVBoxLayout(self.widgets_page)
        self.widgets_page_layout.setSpacing(5)
        self.widgets_page_layout.setObjectName(u"widgets_page_layout")
        self.widgets_page_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_area = QScrollArea(self.widgets_page)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setStyleSheet(u"background: transparent;")
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.contents = QWidget()
        self.contents.setObjectName(u"contents")
        self.contents.setGeometry(QRect(0, 0, 840, 580))
        self.contents.setStyleSheet(u"background: transparent;")
        self.verticalLayout = QVBoxLayout(self.contents)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.title_label = QLabel(self.contents)
        self.title_label.setObjectName(u"title_label")
        self.title_label.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setPointSize(16)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet(u"font-size: 16pt")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.verticalLayout.addWidget(self.title_label)

        self.description_label = QLabel(self.contents)
        self.description_label.setObjectName(u"description_label")
        self.description_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.description_label.setWordWrap(True)
        self.verticalLayout.addWidget(self.description_label)

        self.row_1_layout = QHBoxLayout()
        self.row_1_layout.setObjectName(u"row_1_layout")
        self.verticalLayout.addLayout(self.row_1_layout)

        self.row_2_layout = QHBoxLayout()
        self.row_2_layout.setObjectName(u"row_2_layout")
        self.verticalLayout.addLayout(self.row_2_layout)

        self.row_3_layout = QHBoxLayout()
        self.row_3_layout.setObjectName(u"row_3_layout")
        self.verticalLayout.addLayout(self.row_3_layout)

        self.row_4_layout = QVBoxLayout()
        self.row_4_layout.setObjectName(u"row_4_layout")
        self.verticalLayout.addLayout(self.row_4_layout)

        self.row_5_layout = QVBoxLayout()
        self.row_5_layout.setObjectName(u"row_5_layout")
        self.verticalLayout.addLayout(self.row_5_layout)
        self.scroll_area.setWidget(self.contents)
        self.widgets_page_layout.addWidget(self.scroll_area)

        self.pages.addWidget(self.widgets_page)

    def setup_welcome_page(self) -> None:
        """
        Set up the welcome page.

        This function initializes and configures the welcome page, including the layout, logo, and labels.
        """
        self.welcome_page = QWidget()
        self.welcome_page.setObjectName(u"welcome_page")
        self.welcome_page.setStyleSheet(u"font-size: 14pt")
        self.welcome_page_layout = QVBoxLayout(self.welcome_page)
        self.welcome_page_layout.setSpacing(5)
        self.welcome_page_layout.setObjectName(u"welcome_page_layout")
        self.welcome_page_layout.setContentsMargins(5, 5, 5, 5)
        
        self.welcome_base = QFrame(self.welcome_page)
        self.welcome_base.setObjectName(u"welcome_base")
        self.welcome_base.setMinimumSize(QSize(400, 200))
        self.welcome_base.setMaximumSize(QSize(400, 200))
        self.welcome_base.setFrameShape(QFrame.NoFrame)
        self.welcome_base.setFrameShadow(QFrame.Raised)
        
        self.center_page_layout = QVBoxLayout(self.welcome_base)
        self.center_page_layout.setSpacing(20)
        self.center_page_layout.setObjectName(u"center_page_layout")
        self.center_page_layout.setContentsMargins(0, 0, 0, 0)
        
        # PAGE 1 - ADD LOGO TO MAIN PAGE
        self.logo_png = QLabel(self.welcome_base)
        self.logo_png.setObjectName(u"logo_png")
        self.logo_png.setPixmap(QPixmap(Functions.set_image("ai_logo.png")))
        self.logo_png.setAlignment(Qt.AlignCenter)
        self.center_page_layout.addWidget(self.logo_png)

        self.label = QLabel(self.welcome_base)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)
        self.center_page_layout.addWidget(self.label)
        
        self.welcome_page_layout.addWidget(self.welcome_base, Qt.AlignCenter, Qt.AlignCenter)

        # Add welcome page to the stack
        self.pages.addWidget(self.welcome_page)

    # ----- SETUP IMPORT PAGE ----- #

    def setup_import_page(self) -> None:
        """
        Setup the import page.

        This function initializes and configures the import page, including the layout, status labels, 
        progress bar, and buttons for importing images and stopping the import process.
        """
        # --- import page --- #
        self.import_page = QWidget()
        self.import_page.setObjectName(u"import_page")
        self.import_page.setStyleSheet(u"font-size: 14pt")
        self.import_page_layout = QVBoxLayout(self.import_page)
        self.import_page_layout.setObjectName(u"import_page_layout")
        
        self.import_page_base = QFrame(self.import_page)
        self.import_page_base.setObjectName(u"import_page_base")

        self.center_page_layout = QVBoxLayout(self.import_page_base)
        self.center_page_layout.setObjectName(u"center_page_layout")
        self.center_page_layout.setContentsMargins(0, 0, 0, 0)
    
        self.import_page_layout.addWidget(self.import_page_base, 0, Qt.AlignCenter)
        self.import_page_layout.setAlignment(Qt.AlignCenter)

        # IMPORT LOGO
        self.import_status_logo = QLabel(self.import_page) 
        self.import_status_logo.setGeometry(10, 10, 50, 50)
        pixmap = QPixmap(Functions.set_image("red_cross.png"))
        self.import_status_logo.setPixmap(pixmap)
        self.import_status_logo.hide()
        self.import_status_logo.setObjectName(u"import_status_logo")
        self.import_status_label = QLabel(self.import_page)
        self.import_status_label.setObjectName(u"import_hint_label")
        if self.settings["language"] == "eng":
            self.import_status_label.setText("No images imported yet. \n Click the button below to import images.")
        elif self.settings["language"] == "de":
            self.import_status_label.setText("Noch keine Bilder importiert. \n Klicken Sie auf die Schaltfläche unten, um Bilder zu importieren.")
        self.import_status_label.setAlignment(Qt.AlignCenter)
        self.import_status_label.setContentsMargins(0, 0, 0, 0)
        self.import_status_label.setStyleSheet(u"font-size: 14pt")

        # Add a QlineEdit to let the user set a Trial name /a folder to import the images to
        self.trial_name_label = QLabel(self.import_page)
        self.trial_name_label.setGeometry(QRect(0, 0, 200, 40))
        self.trial_name_label.setObjectName(u"trial_name_label")
        if self.settings["language"] == "eng":
            self.trial_name_label.setText("Trial Name")
        elif self.settings["language"] == "de":
            self.trial_name_label.setText("Versuchsname")
        self.trial_name_label.setAlignment(Qt.AlignCenter)
        self.trial_name_label.setContentsMargins(0, 0, 0, 0)

        trial_name_layout = QVBoxLayout()
        trial_name_layout.setObjectName(u"trial_name_layout")
        trial_name_layout.setContentsMargins(0, 0, 0, 0)
        if self.settings["language"] == "eng":
            trial_name_selection_text = "Specify Trial Name for Import"
        elif self.settings["language"] == "de":
            trial_name_selection_text = "Geben Sie den Versuchsnamen für den Import an"
        self.trial_name_le = PyLineEdit(text=trial_name_selection_text, 
                                     radius=8, 
                                     border_size=2,
                                     color=self.themes["app_color"]["white"], 
                                     bg_color=self.themes["app_color"]["dark_three"])
        self.trial_name_le.setFixedHeight(80)
        self.trial_name_le.setFixedWidth(500)
        self.trial_name_le.setObjectName(u"trial_name")
        self.trial_name_le.setPlaceholderText("Enter trial name")
        self.trial_name_le.setClearButtonEnabled(True)
        self.trial_name_le.clear()
        self.trial_name_le.setAlignment(Qt.AlignCenter)
        trial_name_layout.addWidget(self.trial_name_label)
        trial_name_layout.addWidget(self.trial_name_le)
        trial_name_layout.setAlignment(Qt.AlignCenter)  

        # place the status logo left and the status label right in one row at the center of the page
        self.logo_label_layout = QHBoxLayout()
        self.logo_label_layout.setSpacing(10)
        self.logo_label_layout.addWidget(self.import_status_logo)
        self.logo_label_layout.addWidget(self.import_status_label)
        self.logo_label_layout.setAlignment(Qt.AlignCenter)

        # movie label
        self.import_movie_label = QLabel(self.import_page_base)
        if self.settings["theme_mode"] == "dark":
            movie_path = os.path.join(os.getcwd(), r"gui\images\movies\import_screen_animation_black.gif")
        else:
            movie_path = os.path.join(os.getcwd(), r"gui\images\movies\import_screen_animation_white.gif")
        
        pixmap = QPixmap(movie_path)
        path = QPainterPath()
        path.addRoundedRect(pixmap.rect(), 20, 20)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.import_movie_label.setMask(region)
        self.import_movie_label.setObjectName("movie label")
        self.import_movie_label.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.import_movie = QMovie(movie_path)
        self.import_movie_label.setMovie(self.import_movie)
        self.import_movie_label.setAlignment(Qt.AlignCenter)
        self.import_movie.start()
        self.import_movie.stop()
        self.import_movie_label.hide()

        # add a progress bar to show the progress of the import; hide it initially but show it when the import starts
        self.import_page.progress_bar = PYProgressBar(parent=self.import_page,
                                                    color = "black",
                                                    bg_color = self.themes["app_color"]["context_color"],
                                                    border_color = self.themes["app_color"]["white"],
                                                    border_radius = "5px")
        self.import_page.progress_bar.setAlignment(Qt.AlignCenter)
        self.import_page.progress_bar.setMaximumSize(QSize(400, 20))
        self.import_page.progress_bar.setObjectName(u"import_progress_bar")
        self.import_page.progress_bar.setOrientation(Qt.Horizontal)
        self.import_page.progress_bar.setRange(0, 100)
        self.import_page.progress_bar.setValue(0)
        self.import_page.progress_bar.hide()

        # import button to load images
        if self.settings["language"] == "eng":
            text = "Import Images"
        elif self.settings["language"] == "de":
            text = "Bilder importieren"
        self.import_btn = PyPushButton(
            parent=self.import_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"]
        )
        self.import_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.import_btn.clicked.connect(self.open_import_dialog)
        self.import_btn_icon = QIcon(Functions.set_svg_icon("import_white.svg"))
        self.import_btn.setIcon(self.import_btn_icon)
        self.import_btn.setIconSize(QSize(30, 30))

        # add cancel button to stop the import
        if self.settings["language"] == "eng":
            text = "Stop Import"
        elif self.settings["language"] == "de":
            text = "Import stoppen"
        self.stop_import_btn = PyPushButton(
            parent=self.import_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"]
        )
        self.stop_import_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.stop_import_btn.clicked.connect(self.stop_import)
        self.stop_import_btn_icon = QIcon(Functions.set_svg_icon("stop_icon.svg"))
        self.stop_import_btn.setIcon(self.stop_import_btn_icon)
 
        # add the import button to the layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.import_btn)
        button_layout.addSpacing(200)
        button_layout.addWidget(self.stop_import_btn)
        button_layout.setAlignment(Qt.AlignCenter)

        self.import_page_layout.addLayout(self.logo_label_layout)
        self.import_page_layout.addWidget(self.import_movie_label, 0, Qt.AlignHCenter)
        self.import_page_layout.addSpacing(20)
        self.import_page_layout.addWidget(self.import_page.progress_bar, 0, Qt.AlignHCenter)
        self.import_page_layout.addSpacing(20)
        self.import_page_layout.addLayout(trial_name_layout)
        self.import_page_layout.addSpacing(20)
        self.import_page_layout.addLayout(button_layout)
        self.import_page_layout.setAlignment(Qt.AlignCenter)

        self.pages.addWidget(self.import_page)
        
    def open_import_dialog(self) -> None:
        """
        Open a dialog to select a directory for importing images.

        This function prompts the user to enter a trial name and select a directory containing images to import.
        It then starts the import process in a separate thread, updating the UI with progress and status messages.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial name is entered or no folder is selected.
        """
        # update settings
        self.settings = Settings().items

        # shost is a QString object
        self.trial_name = self.trial_name_le.text()
        self.trial_name_insert_translation = {"eng": "Specify Trial Name for Import", "de": "Geben Sie den Versuchsnamen für den Import an"}
        if self.trial_name == "" or self.trial_name in self.trial_name_insert_translation.values():
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.import_page, "Warning", "Please enter a trial name")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.import_page, "Warnung", "Bitte geben Sie einen Versuchsnamen ein")
            return
        
        # ask the user to select a folder
        if self.settings["language"] == "eng":
            self.image_import_base_dir = QtWidgets.QFileDialog.getExistingDirectory(self.import_page, 
                                                                                    "Select Directory",
                                                                                    options=QtWidgets.QFileDialog.ShowDirsOnly,
                                                                                    dir=os.path.expanduser("~"),                   
                                                                                    )
        elif self.settings["language"] == "de":
            self.image_import_base_dir = QtWidgets.QFileDialog.getExistingDirectory(self.import_page, 
                                                                                    "Ordner auswählen",
                                                                                    options=QtWidgets.QFileDialog.ShowDirsOnly,
                                                                                    dir=os.path.expanduser("~")
                                                                                    )

        if not self.image_import_base_dir:
            # show message box that no folder was selected
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.import_page, "Warning", "No folder selected")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.import_page, "Warnung", "Kein Ordner ausgewählt")
        else:
            # hide the status logo
            self.import_movie_label.show()
            self.import_status_logo.hide()
            self.import_status_label.hide()
            
            # hide the status logo
            self.import_movie.start()
            self.import_page.progress_bar.show()

            # Create the import thread
            self.import_thread = ImportThread(self.import_images)  # Create the import thread

            # Connect the progress signal to the progress bar
            self.import_thread.progress_signal.connect(self.import_page.progress_bar.setValue)
            self.import_thread.status_signal.connect(self.import_status_label.setText)  # Connect the status signal to the label's setText method

            # Connect the finished signal to a function
            self.import_thread.finished_signal.connect(self.on_import_finished)  # Connect the finished signal to a function
            # run the import
            self.import_thread.start()

    def on_import_finished(self) -> None:
        """
        Handle the completion of the image import process.

        This function is called when the image import process is finished. It updates the UI to reflect the import status,
        including stopping the import animation, hiding the progress bar, and displaying the appropriate status message
        based on the number of images imported.

        Returns:
            None
        """
        # This function will be called when the import is finished
        self.import_movie.stop()
        self.import_movie_label.hide()
        self.import_page.progress_bar.setValue(100)
        self.import_page.progress_bar.hide()
        self.trial_name_le.hide()
        self.trial_name_label.hide()

        # check how many images were imported
        if len(self.images_file_paths) == 0:
            # show a warning message that no images were imported
            if self.settings["language"] == "eng":
                #QtWidgets.QMessageBox.warning(self.import_page, "Warning", "No images were imported")
                self.import_status_label.setText("No images were imported. Please modify the import name filter (if set) and try again.")
            elif self.settings["language"] == "de":
                #QtWidgets.QMessageBox.warning(self.import_page, "Warnung", "Keine Bilder wurden importiert")
                self.import_status_label.setText("Keine Bilder wurden importiert. Bitte ändern Sie den Importnamenfilter (falls festgelegt) und versuchen Sie es erneut.")
            # load red cross
            self.import_status_logo.setPixmap(QPixmap(Functions.set_image("red_cross.png")))
            self.import_status_logo.show()
            self.import_status_label.show()
            # show the trial name layout again
            self.trial_name_le.show()
            self.trial_name_label.show()
            return
        else:
            # load green checkmark
            self.import_status_logo.setPixmap(QPixmap(Functions.set_image("green_check.png")))
            self.import_status_logo.show()
            self.import_status_label.show()

            if self.settings["language"] == "eng":
                self.import_status_label.setText(f"{len(self.images_file_paths)} images imported. Click the 'Image Analysis' button to continue.")
            elif self.settings["language"] == "de":
                self.import_status_label.setText(f"{len(self.images_file_paths)} Bilder importiert. Klicken Sie auf die Schaltfläche 'Bildanalyse', um fortzufahren.")
            # show the trial name layout again
            self.trial_name_le.show()
            self.trial_name_label.show()

    def update_settings(self) -> None:
        """
        Update the settings of the main window.

        This function retrieves the latest settings and updates the settings
        attribute of the main window if it exists.

        Args:
            self (object): The main window object.
        """
        settings = Settings()
        print("Settings updated")   
        if hasattr(self, "settings"):
            self.settings = settings.items    
        print("Settings: ", self.settings)
    
    def import_images(self) -> Generator[Tuple[int, str], None, None]:
        """
        Import the images from the selected directory. Yielded progress and status messages are sent to the import thread and used to update the progress bar and status label.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        self.update_settings()
        

        trial_name = self.trial_name_le.text()

        self.import_movie_label.show()
        self.import_movie.start()

        # Create a folder to store the output images
        current_trial_destination = os.path.join(self.imported_files_dir, trial_name)
        imported_images_dir = os.path.join(current_trial_destination, "00_imported_images")
        aux_files_dir = os.path.join(current_trial_destination, "_aux_files")

        if not os.path.exists(current_trial_destination):
            os.makedirs(current_trial_destination, exist_ok=True)

        # Create subdirectories for imported images and auxiliary files
        os.makedirs(imported_images_dir, exist_ok=True)
        os.makedirs(aux_files_dir, exist_ok=True)

        # Store complete paths to the images in a list
        self.images_file_paths = []
        for root, dirs, files in os.walk(self.image_import_base_dir):
            for file in files:
                if file.endswith(".tif"):
                    if convert_str_to_bool(self.settings["processing_settings"]["filter_images"]):
                        if self.settings["processing_settings"]["name_filter"] in file or self.settings["processing_settings"]["name_filter"].upper() in file or self.settings["processing_settings"]["name_filter"].lower() in file:
                            self.images_file_paths.append(os.path.join(root, file))
                    else:
                        self.images_file_paths.append(os.path.join(root, file))
        print(f"Images found: {len(self.images_file_paths)}")
        print(f"Images: {self.images_file_paths}")

        self.images_dims = {os.path.basename(image_file): {"initial": None, "final": None} for image_file in self.images_file_paths}

        self.images_imported = 0
        for idx, image_file in enumerate(self.images_file_paths):
            file_name = os.path.basename(image_file)
            progress_value = int((idx + 1) / len(self.images_file_paths) * 100)
            if self.settings["language"] == "eng":
                status = f"Importing image {idx + 1} of {len(self.images_file_paths)}"
            elif self.settings["language"] == "de":
                status = f"Bild {idx + 1} von {len(self.images_file_paths)} wird importiert"
            yield progress_value, status

            if self.import_process_should_stop:
                print("Import process stopped")
                print(f"Images imported: {self.images_imported}")
                break

            source_path = image_file
            destination_path = os.path.join(imported_images_dir, os.path.basename(image_file))

            # Read the 16-bit image
            img_16bit = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
            if img_16bit is None:
                print(f"Error: Unable to read the image from {source_path}")
                return

            if self.settings["processing_settings"]["convert_img_to_8bit"]:
                img_8bit = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                resolution_unit = 2  # 2 indicates inches
                x_dpi = 72
                y_dpi = 72

                self.images_dims[file_name]["initial"] = img_16bit.shape

                if convert_str_to_bool(self.settings["processing_settings"]["crop_images"]):
                    image_cropper = ImageCropper(
                        image=img_8bit,
                        crop_left=float(self.settings["processing_settings"]["crop_left"]) / 100,
                        crop_right=float(self.settings["processing_settings"]["crop_right"]) / 100,
                        crop_top=float(self.settings["processing_settings"]["crop_top"]) / 100,
                        crop_bottom=float(self.settings["processing_settings"]["crop_bottom"]) / 100,
                        show_cropped_image=False,
                        filename=file_name,
                        saving_path=destination_path
                    )
                    cropped_image = image_cropper.cropped_image
                    cropped_image_dims = cropped_image.shape
                    self.images_dims[file_name]["final"] = cropped_image_dims
                else:
                    cv2.imwrite(destination_path, img_8bit, [
                        cv2.IMWRITE_TIFF_COMPRESSION, 1,
                        cv2.IMWRITE_TIFF_RESUNIT, resolution_unit,
                        cv2.IMWRITE_TIFF_XDPI, x_dpi,
                        cv2.IMWRITE_TIFF_YDPI, y_dpi
                    ])
                    self.images_dims[file_name]["final"] = img_8bit.shape
                
                self.images_imported += 1
                
            else:
                if convert_str_to_bool(self.settings["processing_settings"]["crop_images"]):
                    image_cropper = ImageCropper(
                        image=None,
                        image_path=source_path,
                        crop_left=self.settings["processing_settings"]["crop_left"],
                        crop_right=self.settings["processing_settings"]["crop_right"],
                        crop_top=self.settings["processing_settings"]["crop_top"],
                        crop_bottom=self.settings["processing_settings"]["crop_bottom"],
                        show_cropped_image=False,
                        filename=file_name,
                        saving_path=destination_path
                    )
                    cropped_image = image_cropper.cropped_image
                    cropped_image_dims = cropped_image.shape
                    self.images_dims[file_name]["final"] = cropped_image_dims

                    self.images_imported += 1
                else:
                    shutil.copyfile(source_path, destination_path)
                    self.images_dims[file_name]["final"] = img_16bit.shape
                    self.images_imported += 1

            with open(os.path.join(aux_files_dir, "images_dims.json"), "w") as f:
                json.dump(self.images_dims, f, indent=4)
        
        print(f"Images imported: {self.images_imported}")

        self.total_images += len(self.images_file_paths)
        self.imported_trial_paths.append(current_trial_destination)
        if self.settings["language"] == "eng":
            status = f"Importing images finished. Total images imported: {self.total_images}"
        elif self.settings["language"] == "de":
            status = f"Bilderimport abgeschlossen. Insgesamt importierte Bilder: {self.total_images}"
        yield 100, status

    def get_trial_folders(self) -> list:
        """
        Get the list of folders (Trials) in the Imports directory.

        This function checks if the Imports directory exists and creates it if it doesn't.
        It then retrieves the list of folders within the Imports directory, which represent the trials.

        Returns:
            list: A list of folder names (trials) in the Imports directory.
        """
        # Get the list of folders in the Imports directory
        if not os.path.exists(self.imported_files_dir):
            os.makedirs(self.imported_files_dir, exist_ok=True)
        trial_folders = [folder for folder in os.listdir(self.imported_files_dir) if os.path.isdir(os.path.join(self.imported_files_dir, folder))]
        self.trial_folders = trial_folders
        return trial_folders
    
    def refresh_trial_folder_names(self) -> list:
        """
        Reconstruct the paths to the imported trials and update the trial selection combo boxes.

        This function scans the Imports directory for trial folders, updates the list of trial folders,
        and refreshes the trial selection combo boxes on the analysis, results, and data management pages.

        Returns:
            list: A list of folder names (trials) in the Imports directory.
        """
        # Get the list of folders in the Imports directory
        if not os.path.exists(self.imported_files_dir):
            os.makedirs(self.imported_files_dir, exist_ok=True)
        trial_folders = [folder for folder in os.listdir(self.imported_files_dir) if os.path.isdir(os.path.join(self.imported_files_dir, folder))]
        
        self.trial_folders = trial_folders

        # also of the analysis page, results page and data management page
        self.trial_selection_combo_analysis_page.clear()
        self.trial_selection_combo_results_page.clear()
        self.trial_selection_combo_datamanagement_page.clear()
        if len(self.trial_folders) == 0:
            if self.settings["language"] == "eng":
                self.trial_selection_combo_analysis_page.addItem("No Trials")
                self.trial_selection_combo_results_page.addItem("No Trials")
                self.trial_selection_combo_datamanagement_page.addItem("No Trials")
            elif self.settings["language"] == "de":
                self.trial_selection_combo_analysis_page.addItem("Keine Versuche")
                self.trial_selection_combo_results_page.addItem("Keine Versuche")
                self.trial_selection_combo_datamanagement_page.addItem("Keine Versuche")

        elif len(self.trial_folders) == 1:
            if self.settings["language"] == "eng":
                self.trial_selection_combo_analysis_page.addItem("Select Trial")
                self.trial_selection_combo_analysis_page.addItems(self.trial_folders)
                self.trial_selection_combo_results_page.addItem("Select Trial")
                self.trial_selection_combo_results_page.addItems(self.trial_folders)
                self.trial_selection_combo_datamanagement_page.addItem("Select Trial")
                self.trial_selection_combo_datamanagement_page.addItems(self.trial_folders)
            elif self.settings["language"] == "de":
                self.trial_selection_combo_analysis_page.addItem("Versuch auswählen")
                self.trial_selection_combo_analysis_page.addItems(self.trial_folders)
                self.trial_selection_combo_results_page.addItem("Versuch auswählen")
                self.trial_selection_combo_results_page.addItems(self.trial_folders)
                self.trial_selection_combo_datamanagement_page.addItem("Versuch auswählen")
                self.trial_selection_combo_datamanagement_page.addItems(self.trial_folders)
        else:
            if self.settings["language"] == "eng":  
                self.trial_selection_combo_analysis_page.addItem("Select Trial")
                self.trial_selection_combo_analysis_page.addItem("All Trials")
                self.trial_selection_combo_analysis_page.addItems(self.trial_folders)
                self.trial_selection_combo_results_page.addItem("Select Trial")
                self.trial_selection_combo_results_page.addItem("All Trials")
                self.trial_selection_combo_results_page.addItems(self.trial_folders)
                self.trial_selection_combo_datamanagement_page.addItem("Select Trial")
                self.trial_selection_combo_datamanagement_page.addItem("All Trials")
                self.trial_selection_combo_datamanagement_page.addItems(self.trial_folders)
            elif self.settings["language"] == "de":
                self.trial_selection_combo_analysis_page.addItem("Versuch auswählen")
                self.trial_selection_combo_analysis_page.addItem("Alle Versuche")
                self.trial_selection_combo_analysis_page.addItems(self.trial_folders)
                self.trial_selection_combo_results_page.addItem("Versuch auswählen")
                self.trial_selection_combo_results_page.addItem("Alle Versuche")
                self.trial_selection_combo_results_page.addItems(self.trial_folders)
                self.trial_selection_combo_datamanagement_page.addItem("Versuch auswählen")
                self.trial_selection_combo_datamanagement_page.addItem("Alle Versuche")
                self.trial_selection_combo_datamanagement_page.addItems(self.trial_folders)

        self.trial_selection_combo_analysis_page.setCurrentIndex(0)
        self.trial_selection_combo_results_page.setCurrentIndex(0)
        self.trial_selection_combo_datamanagement_page.setCurrentIndex(0)

        return trial_folders

    def stop_import(self) -> None:
        """
        Stop the image import process.

        This function sets the flag to stop the import process, terminates the import thread,
        hides the progress bar and import animation, and updates the status label and logo
        to indicate that the import process has been stopped.

        Returns:
            None
        """
        self.import_process_should_stop = True
        self.import_thread.terminate()
        self.import_page.progress_bar.hide()
        self.import_movie.stop()
        self.import_movie_label.hide()
        if self.settings["language"] == "eng":
            self.import_status_label.setText("Import process stopped")
        elif self.settings["language"] == "de":
            self.import_status_label.setText("Importprozess gestoppt")
        self.import_status_logo.setPixmap(QPixmap(Functions.set_image("red_cross.png")))
        self.import_status_logo.show()
        self.import_status_label.show()
        self.import_process_should_stop = False


    # ----- SETUP ANALYSIS PAGE ----- #
    
    def toggle_console_window(self) -> None:
        """
        Show or hide the separate console window.

        This function toggles the visibility of the console window. If the console window is currently visible,
        it will be hidden, and the button text will be updated accordingly. If the console window is hidden,
        it will be shown, and the button text will be updated accordingly.
        """
        if self.console_visible:
            self.console_window.hide()
            if self.settings["language"] == "eng":
                self.toggle_console_button.setText("Show Console")
            elif self.settings["language"] == "de":
                self.toggle_console_button.setText("Konsole anzeigen")
        else:
            self.console_window.show()
            if self.settings["language"] == "eng":
                self.toggle_console_button.setText("Hide Console")
            elif self.settings["language"] == "de":
                self.toggle_console_button.setText("Konsole ausblenden")

        self.console_visible = not self.console_visible

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Restore stdout and stderr when the main window is closed.

        Args:
            event (QCloseEvent): The close event.
        """
        sys.stdout = sys.__stdout__  # Restore original stdout
        sys.stderr = sys.__stderr__  # Restore original stderr
        super().closeEvent(event)

    def setup_analysis_page(self) -> None:
        """
        Set up the image analysis page.

        This function initializes and configures the image analysis page, including the layout, status labels,
        progress bar, trial selection combo box, and buttons for running and stopping the analysis.
        """
        self.image_analysis_page = QWidget()
        self.image_analysis_page.setObjectName(u"image_analysis_page")
        self.image_analysis_page.setStyleSheet(u"font-size: 14pt")
        self.image_analysis_page_layout = QVBoxLayout(self.image_analysis_page)
        self.image_analysis_page_layout.setObjectName(u"image_analysis_page_layout")
        
        self.image_analysis_page_base = QFrame(self.image_analysis_page)
        self.image_analysis_page_base.setObjectName(u"image_analysis_page_base")

        self.center_page_layout = QVBoxLayout(self.image_analysis_page_base)
        self.center_page_layout.setObjectName(u"center_page_layout")
        self.center_page_layout.setContentsMargins(0, 0, 0, 0)
    
        self.image_analysis_page_layout.addWidget(self.image_analysis_page_base, 0, Qt.AlignHCenter)

        # movie label
        self.movie_label = QLabel(self.image_analysis_page_base)
        if self.settings["theme_mode"] == "dark":
            movie_path = os.path.join(os.getcwd(), r"gui\images\movies\ai_processing_screen_black.gif")
        else:
            movie_path = os.path.join(os.getcwd(), r"gui\images\movies\ai_processing_screen_black.gif")
        
        self.movie_label = QLabel(self.image_analysis_page_base)
        pixmap = QPixmap(movie_path)
        path = QPainterPath()
        path.addRoundedRect(pixmap.rect(), 20, 20)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.movie_label.setMask(region)
        self.movie_label.setObjectName("movie label")
        self.movie_label.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.movie_label.setObjectName("movie_label")
        self.movie = QMovie(movie_path)
        self.movie_label.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()
        self.image_analysis_page_layout.addWidget(self.movie_label)
        self.image_analysis_page_layout.setAlignment(Qt.AlignCenter)
        self.image_analysis_page_layout.addSpacing(10)

        # add a text label to show the status of the analysis
        self.analysis_status_label = QLabel(self.image_analysis_page_base)
        self.analysis_status_label.setObjectName(u"analysis_status_label")
        self.analysis_status_label.setAlignment(Qt.AlignCenter)
        if self.settings["language"] == "eng":
            self.analysis_status_label.setText("Click the 'Run Analysis' button to start the analysis")
        elif self.settings["language"] == "de":
            self.analysis_status_label.setText("Klicken Sie auf die Schaltfläche 'Analyse starten', um die Analyse zu starten")
        self.image_analysis_page_layout.addWidget(self.analysis_status_label)
        self.image_analysis_page_layout.addSpacing(10)

        # add a second label to show the status of the analysis
        self.analysis_status_label_2 = QLabel(self.image_analysis_page_base)
        self.analysis_status_label_2.setObjectName(u"analysis_status_label_2")
        self.analysis_status_label_2.setAlignment(Qt.AlignCenter)
        self.analysis_status_label_2.setText(" ")
        self.image_analysis_page_layout.addWidget(self.analysis_status_label_2)
        self.image_analysis_page_layout.addSpacing(5)
        
        
        # add a eta label to show the estimated time of arrival of the analysis
        self.analysis_eta_label = QLabel(self.image_analysis_page_base)
        self.analysis_eta_label.setObjectName(u"analysis_eta_label")
        self.analysis_eta_label.setAlignment(Qt.AlignCenter)
        self.analysis_eta_label.setText(" ")
        self.image_analysis_page_layout.addWidget(self.analysis_eta_label)
        self.image_analysis_page_layout.addSpacing(5)

        # add a progress bar to show the progress of the analysis; hide it initially but show it when the analysis starts
        self.analysis_progress_bar = PYProgressBar(parent=self.image_analysis_page_base,
                                        color = "black",
                                        bg_color = self.themes["app_color"]["context_color"],
                                        border_color=self.themes["app_color"]["white"],
                                        border_radius="5px")
        self.analysis_progress_bar.setMaximumSize(QSize(600, 100))
        self.analysis_progress_bar.setObjectName(u"analysis_progress_bar")
        self.analysis_progress_bar.setOrientation(Qt.Horizontal)
        self.analysis_progress_bar.setRange(0, 100)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.hide()
        self.analysis_progress_bar.setAlignment(Qt.AlignHCenter)
        self.image_analysis_page_layout.addWidget(self.analysis_progress_bar, 0, Qt.AlignHCenter)
        self.image_analysis_page_layout.addSpacing(5)
        self.image_analysis_page_layout.setAlignment(Qt.AlignCenter)

        # show the user a dropdown list of the trials to select from for analysis also add an "ALL TRIALS" option to analyze all trials
        self.trial_selection_label_analysis = QLabel(self.image_analysis_page_base)
        self.trial_selection_label_analysis.setObjectName(u"trial_selection_label_analysis")
        if self.settings["language"] == "eng":
            self.trial_selection_label_analysis.setText("Select Trial: ")
        elif self.settings["language"] == "de":
            self.trial_selection_label_analysis.setText("Versuch auswählen: ")
        self.trial_selection_label_analysis.setAlignment(Qt.AlignCenter)
        self.trial_selection_label_analysis.setContentsMargins(0, 0, 0, 0)
        self.trial_selection_label_analysis.setStyleSheet(u"font-size: 14pt")

        # create a combo box to display the list of trials
        self.trial_selection_combo_analysis_page = PyComboBox(self.image_analysis_page_base)
        self.trial_selection_combo_analysis_page.setMaximumWidth(600)
        self.trial_selection_combo_analysis_page.setObjectName(u"trial_selection_combo_analysis_page")
        self.trial_selection_combo_analysis_page.addItems(self.get_trial_folders())

        # add the trial selection label and combo to the layout
        self.image_analysis_page_layout.addWidget(self.trial_selection_label_analysis, 0, Qt.AlignHCenter)

        trial_selection_layout = QHBoxLayout()
        trial_selection_layout.addWidget(self.trial_selection_label_analysis)
        trial_selection_layout.addWidget(self.trial_selection_combo_analysis_page)
        trial_selection_layout.setAlignment(Qt.AlignCenter)

        self.image_analysis_page_layout.addLayout(trial_selection_layout)
        self.image_analysis_page_layout.setAlignment(Qt.AlignCenter)
        self.image_analysis_page_layout.addSpacing(15)

        # analysis button
        if self.settings["language"] == "eng":
            text = "Run Analysis"
        elif self.settings["language"] == "de":
            text = "Analyse starten"
        self.analysis_btn = PyPushButton(
                parent=self.image_analysis_page,
                text=text,
                radius=8,
                color=self.themes["app_color"]["text_foreground"],
                bg_color=self.themes["app_color"]["dark_one"],
                bg_color_hover=self.themes["app_color"]["dark_three"],
                bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.analysis_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.analysis_btn.clicked.connect(self.open_analysis_dialog)
        self.analysis_btn_icon = QIcon(Functions.set_svg_icon("run_analysis_icon.svg")) 
        self.analysis_btn.setIcon(self.analysis_btn_icon)
        self.analysis_btn.setIconSize(QSize(30, 20))

        # add a cancel button to stop the analysis
        if self.settings["language"] == "eng":
            text = "Stop Analysis"
        elif self.settings["language"] == "de":
            text = "Analyse stoppen"
        self.stop_analysis_btn = PyPushButton(
                parent=self.image_analysis_page,
                text=text,
                radius=8,
                color=self.themes["app_color"]["text_foreground"],
                bg_color=self.themes["app_color"]["dark_one"],
                bg_color_hover=self.themes["app_color"]["dark_three"],
                bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.stop_analysis_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn_icon = QIcon(Functions.set_svg_icon("stop_icon.svg"))
        self.stop_analysis_btn.setIcon(self.stop_analysis_btn_icon)
        self.stop_analysis_btn.setIconSize(QSize(30, 20))
        
        # Button to open/close the console window
        if self.settings["language"] == "eng":
            text = "Show Console"
        elif self.settings["language"] == "de":
            text = "Konsole anzeigen"
        self.toggle_console_button = PyPushButton(
            parent=self.image_analysis_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.toggle_console_button.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.show_console_btn_icon = QIcon(Functions.set_svg_icon("show_console_icon.svg"))
        self.toggle_console_button.setIcon(self.show_console_btn_icon)
        self.toggle_console_button.setIconSize(QSize(30, 20))
        self.toggle_console_button.clicked.connect(self.toggle_console_window)

        # Create the console window but do not show it yet
        self.console_window = ConsoleWindow()
        self.console_window.setWindowTitle("Console")
        self.console_visible = False
    
        # Define the button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.analysis_btn)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.stop_analysis_btn)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.toggle_console_button)
        self.image_analysis_page_layout.addLayout(button_layout)
        
        # add the page to the stack
        self.pages.addWidget(self.image_analysis_page)
        
    def check_imported_files(self) -> bool:
        """
        Check if the imported files directory exists and store the absolute paths of images in the directory.

        This function verifies the existence of the imported files directory and iterates through its subdirectories
        to find image files with the ".tif" extension. It stores the absolute paths of these images in a dictionary
        where the keys are the trial names and the values are lists of image file paths.

        Returns:
            bool: True if the directory exists and images are found, False otherwise.
        """
        if not os.path.exists(self.imported_files_dir):
            return False
        else:
            self.images_file_paths = {}
            for trial_name in os.listdir(self.imported_files_dir):
                trial_dir = os.path.join(self.imported_files_dir, trial_name)
                imported_images_path = os.path.join(trial_dir, "00_imported_images")
                if os.path.isdir(imported_images_path):
                    trial_images = [os.path.join(imported_images_path, file) for file in os.listdir(imported_images_path) if file.endswith(".tif")]
                    if len(trial_images) > 0:
                        self.images_file_paths[trial_name] = trial_images
            if len(self.images_file_paths) == 0:
                return False
            else:
                return True
            
    def stop_analysis(self) -> None:
        """
        Stop the image analysis process.

        This function updates the UI to reflect the stopping of the analysis process, including changing the button icon and text,
        setting the stop flag, hiding the progress bar, and updating the status labels.

        Returns:
            None
        """
        # Update the icon
        self.analysis_btn_icon = QIcon(Functions.set_svg_icon("resume_white.svg"))
        self.analysis_btn.setIcon(self.analysis_btn_icon)
        self.analysis_btn.setIconSize(QSize(30, 20))

        # Set the stop analysis flag to True
        self.analysis_thread_should_stop = True

        # Update the status label
        #if self.settings["language"] == "eng":
        #    self.analysis_status_label.setText("Analysis stopped. You may resume the analysis below.")
        #elif self.settings["language"] == "de":
        #    self.analysis_status_label.setText("Analyse gestoppt. Sie können die Analyse unten fortsetzen.")
        #self.analysis_status_label_2.setText(" ")
        self.analysis_status_label.setText(" ")
        if self.settings["language"] == "eng":
            self.analysis_status_label_2.setText("Analysis stopped. You may resume the analysis below.")
        elif self.settings["language"] == "de":
            self.analysis_status_label_2.setText("Analyse gestoppt. Sie können die Analyse unten fortsetzen.")
        self.analysis_eta_label.setText(" ")

        self.movie.stop()
        self.analysis_progress_bar.hide()
        self.analysis_status_label.hide()
        
        # Change the text of the analysis button to "Resume Analysis"
        if self.settings["language"] == "eng":
            self.analysis_btn.setText("Resume Analysis")
        elif self.settings["language"] == "de":
            self.analysis_btn.setText("Analyse fortsetzen")

        # Show the trial selection combo box
        self.trial_selection_label_analysis.show()
        self.trial_selection_combo_analysis_page.show()
        
    def open_analysis_dialog(self) -> None:
        """
        Open the analysis dialog and start the image analysis process.

        This function checks if images have been imported and if a trial has been selected.
        If the conditions are met, it hides the trial selection combo box, shows the progress bar,
        and starts the image analysis process in a separate thread. It also updates the UI with
        progress and status messages.

        Returns:
            None
        """
        self.analysis_status_label.show()
        self.analysis_status_label_2.show()
        self.analysis_status_label_2.setText(" ")
        self.analysis_status_label.setText(" ")
        self.analysis_progress_bar.setValue(0)

        # update the icon
        self.analysis_btn_icon = QIcon(Functions.set_svg_icon("run_analysis_icon.svg"))
        self.analysis_btn.setIcon(self.analysis_btn_icon)
        self.analysis_btn.setIconSize(QSize(30, 20))

        # check if images have been imported
        if not self.check_imported_files():
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.image_analysis_page, "Warning", "No images imported. Please import images first.")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.image_analysis_page, "Warnung", "Keine Bilder importiert. Bitte importieren Sie zuerst Bilder.")
        else:
            selected_trial = self.trial_selection_combo_analysis_page.currentText()
            if selected_trial == "Select Trial" or selected_trial == "Versuch auswählen":
                if self.settings["language"] == "eng":
                    QtWidgets.QMessageBox.warning(self.image_analysis_page, "Warning", "Please select a trial to analyze")
                elif self.settings["language"] == "de":
                    QtWidgets.QMessageBox.warning(self.image_analysis_page, "Warnung", "Bitte wählen Sie einen Versuch zur Analyse aus")
            else:
                # hide combo box label and combo box
                self.trial_selection_label_analysis.hide()
                self.trial_selection_combo_analysis_page.hide()

                # run the analysis
                self.movie.start()
                # show the progress bar
                self.analysis_progress_bar.show()
                # show the status label
                if self.settings["language"] == "eng":
                    self.analysis_status_label.setText("Running Image Analysis")
                elif self.settings["language"] == "de":
                    self.analysis_status_label.setText("Bildanalyse läuft")
                self.analysis_thread = AnalysisThread(parent=self.image_analysis_page, inference_function=self.perform_inference)
                self.analysis_thread.start()
                self.analysis_thread.finished_signal.connect(self.analysis_finished_slot)
                self.analysis_thread.progress_signal.connect(self.analysis_progress_bar.setValue)
                self.analysis_thread.stop_process_signal.connect(self.stop_analysis)
                
    def cache_current_results(self, results_dict: dict) -> None:
        """
        Cache the current results in a dictionary.

        Args:
            results_dict (dict): The dictionary containing the results to be cached.
        """
        if results_dict != {}:   
            self.results_dict = results_dict

    def update_image_num(self, value: int) -> None:
        """
        Update the current image index.

        Args:
            value (int): The current image index.
        """
        self.current_image_index = value

    def analysis_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the image analysis process.

        Args:
            value (bool): The status of the analysis process.
        """
        self.movie.stop()
        self.analysis_progress_bar.hide()
        self.analysis_eta_label.setText(" ")

        self.trial_selection_label_analysis.show()
        self.trial_selection_combo_analysis_page.show()

        if self.settings["language"] == "eng":
            self.analysis_status_label.setText("Image Analysis completed. You may view the results (via the 'View Results' tab) \n or start a new analysis below.")
        elif self.settings["language"] == "de":
            self.analysis_status_label.setText("Bildanalyse abgeschlossen. Sie können die Ergebnisse anzeigen (über den Tab 'Ergebnisse anzeigen') \n oder unten eine neue Analyse starten.")
            
    def perform_inference(self) -> Generator[Tuple[int, str, str, dict], None, None]:
        """
        Perform inference on the images in the selected trial.

        This function iterates over the images in the selected trial(s) and performs inference using the specified model.
        It updates the UI with progress and status messages and stores the results in a dictionary.

        Yields:
            Tuple[int, str, str, dict]: The current image index, process name, status message, and results dictionary.
        """
        self.settings = Settings().items

        # Get the selected trial
        selected_trial = self.trial_selection_combo_analysis_page.currentText()

        # Update the status label
        if self.settings["language"] == "eng":
            upper_label_text = f"Analyzing images ... "
            self.analysis_status_label.setText(upper_label_text)
        elif self.settings["language"] == "de":
            upper_label_text = f"Bilder werden analysiert ... "
            self.analysis_status_label.setText(upper_label_text)

        # Get the images to analyze
        if selected_trial == "All Trials" or selected_trial == "Alle Versuche":
            trials = self.images_file_paths.keys()
            images = []
            for trial in trials:
                images.extend(self.images_file_paths[trial])
            total_images = len(images)
            print(f"Total images to analyze: {total_images}")
        else:
            trials = [selected_trial]
            total_images = len(self.images_file_paths[selected_trial])

        num_trials = len(trials)
        eta = 0
        mean_time_per_image = 0
        mean_time_per_trial = 0
        
        total_images_to_be_analyzed = 0
        for trial in trials:
            prediction_dir = os.path.join(self.imported_files_dir, trial, "01_image_analysis", "02_model_predictions")
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
                images = self.images_file_paths[trial]
            else:
                prediction_files = [file for file in os.listdir(prediction_dir) if file.endswith(".png")]
                # determine the images that have not been analyzed
                images = [image for image in self.images_file_paths[trial] if f"{os.path.splitext(os.path.basename(image))[0]}_prediction.png" not in prediction_files]
            total_images_to_be_analyzed += len(images)
            print(f"Total images to be analyzed in trial {trial}: {len(images)}")
        already_analyzed_images = total_images - total_images_to_be_analyzed
        print(f"Total images to be analyzed: {total_images_to_be_analyzed}")
            
            
        #for trial in trials:
        for idx, trial in enumerate(trials):
            
            # track time needed for analysis
            start_time_trial = time.time()
            
            # Get the total number of images in the current trial
            all_images_of_trial = self.images_file_paths[trial]
            total_images_trial = len(all_images_of_trial)

            # Get the predictions directory
            predictions_dir = os.path.join(self.imported_files_dir, trial, "01_image_analysis", "02_model_predictions")
            if not os.path.exists(predictions_dir):
                os.makedirs(predictions_dir, exist_ok=True)
                images = all_images_of_trial
            else:
                # Get the list of prediction files
                prediction_files = [file for file in os.listdir(predictions_dir) if file.endswith(".png")]
                print(f"Prediction files: {prediction_files}")
                # Get the list of images that have not been analyzed
                images = [image for image in all_images_of_trial if f"{os.path.splitext(os.path.basename(image))[0]}_prediction.png" not in prediction_files]
            print(f"Images to analyze for trial {trial}: {len(images)}")
            print(f"images in trial {trial}: {images}")
            
            already_analyzed_images_in_trial = total_images_trial - len(images)

            if self.settings["language"] == "eng":
                label_text = f"Processing Import: '{trial}'"
            elif self.settings["language"] == "de":
                label_text = f"Import '{trial}' wird verarbeitet"

            if selected_trial == "All Trials" or selected_trial == "Alle Versuche":
                if self.settings["language"] == "eng":
                    label_text = upper_label_text + f"\n images in all imports: {total_images} ({already_analyzed_images} already analyzed)"
                    label_text = label_text + f"\n images in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} already analyzed)"
                elif self.settings["language"] == "de":
                    label_text = upper_label_text + f"\n Bilder in allen Imports: {total_images} ({already_analyzed_images} bereits analysiert)"
                    label_text = label_text + f"\n Bilder in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} bereits analysiert)"
            else:
                if self.settings["language"] == "eng":
                    label_text = upper_label_text + f"\n images in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} already analyzed)"
                elif self.settings["language"] == "de":
                    label_text = upper_label_text + f"\n Bilder in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} bereits analysiert)"

            self.analysis_status_label.setText(label_text)

            # Construct a model interface with the current trial as output directory
            trial_dir = os.path.join(self.imported_files_dir, trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]

            # Check if there already exists an image_analysis_results.json file in the trial directory
            # If it exists, load the results dictionary from the file
            # If it does not exist, create a new results dictionary
            results_file = os.path.join(trial_dir, "_aux_files", "image_analysis_results.json")
            if os.path.exists(results_file):
                if hasattr(self, "results_dict"):
                    del self.results_dict
                with open(results_file, "r") as f:
                    self.results_dict = json.load(f)
                    print(f"Results file loaded from {results_file}")
            else:
                self.results_dict = {}
                print(f"Results file created")

            # If the process has not been stopped, run the inference passing None as the results_dict
            if hasattr(self, "model_interface"):
                del self.model_interface
                torch.cuda.empty_cache()
                
            if self.settings["language"] == "eng":
                status = f"Calculating ETA ..."
            elif self.settings["language"] == "de":
                status = f"ETA wird berechnet ..."
            self.analysis_eta_label.setText(status)

            if hasattr(self, "instances_dict"):
                self.model_interface = ModelInteractor(
                    instances_dict=self.instances_dict,
                    store_coco_files=convert_str_to_bool(self.settings["processing_settings"]["save_coco"]),
                    store_csv_file=convert_str_to_bool(self.settings["processing_settings"]["save_csv"]),
                    store_xlsx_file=convert_str_to_bool(self.settings["processing_settings"]["save_excel"]),
                    output_dir=trial_dir,
                    physical_image_width=self.physical_image_width,
                )

                # Run the inference
                gen_return = self.model_interface.run_inference(
                    images, trial_dir,
                    language=self.settings["language"],
                    results_dict=self.results_dict,
                    analysis_thread_should_stop=self.analysis_thread_should_stop
                )
            

                for image_num, process, status, results_dict, time_take_per_image in gen_return:
                    if self.analysis_thread_should_stop:
                        print("Analysis stopped")
                        break
                    
                    
                    # Update the status label
                    self.analysis_status_label_2.setText(status)
                 

                    # Store the current index of the image being analyzed
                    self.current_image_index = image_num
                    current_progress = int((image_num / len(images)) * 100)
                    
                    if time_take_per_image != None:
                        print(f"Current progress: {current_progress} % of total of {total_images_trial} images of trial {trial}; {image_num} images analyzed, {len(images) - image_num} images remaining; total images: {total_images}")
                        print(f"Time taken for image {image_num}: {time_take_per_image} seconds")
   
                    #status = f"{status} ({image_num}/{len(images)})"
                    
                    
                    if selected_trial == "All Trials" or selected_trial == "Alle Versuche":
                        if self.settings["language"] == "eng":
                            label_text = upper_label_text + f"\n Images in all imports: {total_images} ({already_analyzed_images} already analyzed)"
                            label_text = label_text + f"\n images to analyse in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} already analyzed)"
                        elif self.settings["language"] == "de":
                            label_text = upper_label_text + f"\n Bilder in allen Imports: {total_images} ({already_analyzed_images} bereits analysiert)"
                            label_text = label_text + f"\n Bilder in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} bereits analysiert)"
                    else:
                        if self.settings["language"] == "eng":
                            label_text = upper_label_text + f"\n Images in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} already analyzed)"
                        elif self.settings["language"] == "de":
                            label_text = upper_label_text + f"\n Bilder in '{trial}': {total_images_trial} ({already_analyzed_images_in_trial} bereits analysiert)"
                    #label_text = f"{label_text} \n {status} ({image_num}/{len(images)})"
                    self.analysis_status_label.setText(label_text)
                    
                    if time_take_per_image != None:
                        yield image_num, process, status, results_dict, time_take_per_image
                        
                        #print(f"time needed for image {image_num}: {np.round(time_take_per_image, 2)} seconds")
                        print(f"Images to analyze in '{trial}': {len(images)}")
                        print(f"Images analyzed in '{trial}': {already_analyzed_images_in_trial}")
                        print(f"Images to analyze in all imports: {total_images} ({already_analyzed_images} already analyzed)")
                        images_remaining = total_images - already_analyzed_images
                        print(f"Images remaining: {images_remaining}")
                        
                        if idx == 0:
                            mean_time_per_trial = time_take_per_image
                        else:
                            mean_time_per_trial = (mean_time_per_trial * (idx - 1) + time_take_per_image) / idx
                        eta = mean_time_per_trial * images_remaining
                        eta = eta / 60
                        eta = round(eta, 2)
                        eta_min = int(eta)
                        eta_sec = int((eta - eta_min) * 60)
                        eta_text = f"{eta_min} min {eta_sec} sec"
                        if self.settings["language"] == "eng":
                            eta_text = f"Estimated amount of time remaining (total): {eta_text}"
                        elif self.settings["language"] == "de":
                            eta_text = f"Voraussichtliche verbleibende Zeit (insgesamt): {eta_text}"
                        self.analysis_eta_label.setText(eta_text)
                    else:
                        yield image_num, process, status, results_dict, 0
                        
                already_analyzed_images_in_trial += 1
                already_analyzed_images += 1

                    
                    # mean time per image
                    #if mean_time_per_image == 0:
                    #    mean_time_per_image = time_take_per_image
                    #else:
                    #    mean_time_per_image = (mean_time_per_image * (image_num - 1) + time_take_per_image) / image_num
                        
                    # calculate the estimated time of arrival
                    #eta += mean_time_per_image
          
                    
                self.analysis_thread_should_stop = False
                
            else:
                print("instances_dict not found")
            

            # Delete the model interface upon completion
            del self.model_interface
            del self.results_dict
            torch.cuda.empty_cache()
            print("Model interface removed from memory after analysis")
            
            # track time needed for analysis
            end_time_trial = time.time()
            time_needed = end_time_trial - start_time_trial
            #print(f"Time needed for analysis of trial {trial}: {np.round(time_needed, 2)} seconds")
            """            
            # calculate the estimated time of arrival
            if idx == 0:
                mean_time_per_trial = time_needed
            else:
                mean_time_per_trial = (mean_time_per_trial * (idx - 1) + time_needed) / idx
            
            eta += mean_time_per_trial
            trials_remaining = num_trials - idx
            eta = eta / trials_remaining
            eta = eta / 60
            eta = round(eta, 2)
            eta_min = int(eta)
            eta_sec = int((eta - eta_min) * 60)
            eta_text = f"{eta_min} min {eta_sec} sec"
            if self.settings["language"] == "eng":
                eta_text = f"Estimated amount of time remaining: {eta_text}"
            elif self.settings["language"] == "de":
                eta_text = f"Voraussichtliche verbleibende Zeit: {eta_text}"
            self.analysis_eta_label.setText(eta_text)
            """
            

        
        # Update the status label
        if self.settings["language"] == "eng":
            self.analysis_status_label.setText("Image Analysis completed. You may view the results or start another analysis by clicking the 'Run Analysis' button.")
        elif self.settings["language"] == "de":
            self.analysis_status_label.setText("Bildanalyse abgeschlossen. Sie können die Ergebnisse anzeigen oder eine weitere Analyse starten, indem Sie auf die Schaltfläche 'Analyse starten' klicken.")
        self.analysis_status_label_2.setText(" ")


    # ----- SETUP RESULTS PAGE ----- #
    def setup_results_page(self) -> None:
        """
        Set up the results page.

        This function initializes and configures the results page, including the layout, trial selection combo box,
        progress bar for report generation, and buttons for viewing results, storing results, and generating reports.
        """
        self.results_page = QWidget()
        self.results_page.setObjectName(u"results_page")
        self.results_page.setStyleSheet(u"font-size: 14pt")
        self.results_page_layout = QVBoxLayout(self.results_page)
        self.results_page_layout.setObjectName(u"results_page_layout")
        self.results_page_layout.setContentsMargins(5, 5, 5, 5)
        self.results_page_layout.setSpacing(5)
        
        # let the user select a trial
        self.trial_selection_label_results = QLabel(self.results_page)
        self.trial_selection_label_results.setObjectName(u"trial_selection_label_results")
        self.trial_selection_label_results.setAlignment(Qt.AlignLeft)
        self.trial_selection_label_results.setText("Select a trial to view the results")
        self.results_page_layout.addWidget(self.trial_selection_label_results)
        self.results_page_layout.addSpacing(10)

        # add progress bar to show the progress of the report generation
        self.results_page_progress_bar = PYProgressBar(parent=self.results_page,
                                        color = "black",
                                        bg_color = self.themes["app_color"]["context_color"],
                                        border_color=self.themes["app_color"]["white"],
                                        border_radius="5px")
        self.results_page_progress_bar.setMaximumSize(QSize(600, 100))
        self.results_page_progress_bar.setObjectName(u"results_page_progress_bar")
        self.results_page_progress_bar.setOrientation(Qt.Horizontal)
        self.results_page_progress_bar.setRange(0, 100)
        self.results_page_progress_bar.setValue(0)
        self.results_page_progress_bar.hide()
        self.results_page_progress_bar.setAlignment(Qt.AlignHCenter)
       

        # create a combo box to select the trial
        self.trial_selection_combo_results_page = PyComboBox(self.results_page)
        self.trial_selection_combo_results_page.setMinimumWidth(200)
        self.trial_selection_combo_results_page.setObjectName(u"trial_selection_combo_results_page")
        #self.trial_selection_combo_results_page.addItem("Select Trial")
        self.trial_selection_combo_results_page.addItems(self.get_trial_folders())

        # add a refresh button to refresh the list of trials
        """
            self.refresh_trials_btn_results_page = PyPushButton(
            parent=self.results_page,
            text="Refresh Trials",
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.refresh_trials_btn_results_page.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.refresh_trials_btn_results_page.clicked.connect(self.refresh_trial_folder_names)
        """
        

        # add a button to view the results
        if self.settings["language"] == "eng":
            text = "View Results"
        elif self.settings["language"] == "de":
            text =  "Ergebnisse anzeigen"
        self.view_results_btn = PyPushButton(
            parent=self.results_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.view_results_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))   
        self.results_page_layout.addWidget(self.view_results_btn, 0, Qt.AlignCenter)
        self.view_results_btn.clicked.connect(self.show_results_is_clicked)
        self.view_results_btn_icon = QIcon(Functions.set_svg_icon("view_icon.svg"))
        self.view_results_btn.setIcon(self.view_results_btn_icon)
        self.view_results_btn.setIconSize(QSize(30, 30))

        # add a button to store results for selected trial
        if self.settings["language"] == "eng":
            text = "Store Results"
        elif self.settings["language"] == "de":
            text = "Ergebnisse speichern"
        self.store_results_btn = PyPushButton(
            parent=self.results_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.store_results_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.results_page_layout.addWidget(self.store_results_btn, 0, Qt.AlignCenter)
        self.store_results_btn.clicked.connect(self.store_results_dialog)
        self.store_results_btn_icon = QIcon(Functions.set_svg_icon("icon_save.svg"))
        self.store_results_btn.setIcon(self.store_results_btn_icon)
        self.store_results_btn.setIconSize(QSize(30, 30))



        # add a button to generate a report for selected trial
        if self.settings["language"] == "eng":
            text = "Generate Report"
        elif self.settings["language"] == "de":
            text = "Bericht generieren"
        self.generate_report_btn = PyPushButton(
            parent=self.results_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.generate_report_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.results_page_layout.addWidget(self.generate_report_btn, 0, Qt.AlignCenter)
        self.generate_report_btn.clicked.connect(self.open_generate_report_dialog)
        self.generate_report_btn_icon = QIcon(Functions.set_svg_icon("report_white.svg"))
        self.generate_report_btn.setIcon(self.generate_report_btn_icon)
        self.generate_report_btn.setIconSize(QSize(30, 30))

        trial_selection_layout = QVBoxLayout()
        trial_selection_layout.addWidget(self.trial_selection_label_results)

        trial_selection_button_layout = QHBoxLayout()
        trial_selection_button_layout.addWidget(self.trial_selection_combo_results_page)
        #trial_selection_button_layout.addSpacing(200)
        trial_selection_button_layout.addSpacing(50)
        
        trial_selection_button_layout.addWidget(self.results_page_progress_bar)
        trial_selection_button_layout.addSpacing(50)
        
        #trial_selection_button_layout.addWidget(self.refresh_trials_btn_results_page)
        trial_selection_button_layout.addWidget(self.view_results_btn)
        trial_selection_button_layout.addSpacing(50)
        trial_selection_button_layout.addWidget(self.store_results_btn)
        trial_selection_button_layout.addSpacing(50)
        trial_selection_button_layout.addWidget(self.generate_report_btn)
        
        trial_selection_layout.addLayout(trial_selection_button_layout)
        self.results_page_layout.addLayout(trial_selection_layout)
        # align the trial selection layout to the center
        self.results_page_layout.setAlignment(Qt.AlignCenter)
        self.results_page_layout.addSpacing(50)
        
        #self.results_page_layout.addWidget(self.results_page_progress_bar, 0, Qt.AlignHCenter)
        #self.results_page_layout.addSpacing(10)
        
        self.pages.addWidget(self.results_page)
        
    def store_results_dialog(self) -> None:
        """
        Store the results of the selected trial.

        This function retrieves the selected trial from the combo box and stores the results for that trial.
        If "All Trials" is selected, it stores the results for all trials. The function updates the status label
        and uses a separate thread to perform the storage operation.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial is selected or if there are no trials to store results for.
        """
        # get the selected trial
        selected_trial = self.trial_selection_combo_results_page.currentText()
        if selected_trial == "Select Trial" or selected_trial == "No Trials" or \
            selected_trial == "Keine Versuche" or selected_trial == "Versuch auswählen":
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Please select a trial to store the results")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Bitte wählen Sie einen Versuch aus, um die Ergebnisse zu speichern")
        elif selected_trial == "All Trials" or selected_trial == "Alle Versuche":
            if self.settings["language"] == "eng":
                self.trial_selection_label_results.setText("Storing results for all trials ...")
            elif self.settings["language"] == "de":
                self.trial_selection_label_results.setText("Ergebnisse für alle Versuche werden gespeichert ...")
            # get the directories of the trials
            trial_dirs = [os.path.join(self.imported_files_dir, trial) for trial in self.trial_folders if trial not in self.folder_translation.values()]
            if len(trial_dirs) == 0:
                if self.settings["language"] == "eng":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warning", "No trials to store results for.")
                elif self.settings["language"] == "de":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Keine Versuche, für die Ergebnisse gespeichert werden können.")
            else:
                if self.settings["language"] == "eng":
                    self.trial_selection_label_results.setText("Storing results for all trials ...")
                elif self.settings["language"] == "de":
                    self.trial_selection_label_results.setText("Ergebnisse für alle Versuche werden gespeichert ...")
                    
                # store the results for all the trials
                self.results_page_progress_bar.show()
                self.results_page_progress_bar.setValue(0)
                self.store_results_thread = StoreResultsThread(parent=self.results_page, store_results_function=self.store_results_for_multiple_trials(trial_paths=trial_dirs))
                self.store_results_thread.start()
                self.store_results_thread.finished_signal.connect(self.store_results_finished_slot)
                self.store_results_thread.status_signal.connect(self.trial_selection_label_results.setText)
                self.store_results_thread.progress_signal.connect(self.results_page_progress_bar.setValue)
        else:
            if self.settings["language"] == "eng":
                self.trial_selection_label_results.setText(f"Storing results for trial {selected_trial}")
            elif self.settings["language"] == "de":
                self.trial_selection_label_results.setText(f"Ergebnisse für Versuch {selected_trial} werden gespeichert")
            
            self.results_page_progress_bar.show()
            self.results_page_progress_bar.setValue(0)
            # get the directory of the selected trial
            trial_dir = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]
            # store the results
            self.store_results_thread = StoreResultsThread(parent=self.results_page, store_results_function=self.store_results_function(trial_dir=trial_dir))
            self.store_results_thread.start()
            self.store_results_thread.finished_signal.connect(self.store_results_finished_slot)
            self.store_results_thread.status_signal.connect(self.trial_selection_label_results.setText)
            self.store_results_thread.progress_signal.connect(self.results_page_progress_bar.setValue)
            
            
    def store_results_function(self, trial_dir: str) -> Generator[Tuple[int, str], None, None]:
        """
        Store the results of the selected trial.

        This function initializes a ResultsViewer object for the specified trial directory,
        stores the results, and yields the progress and status message.

        Args:
            trial_dir (str): The directory of the trial for which results are to be stored.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        self.results_page_progress_bar.show()
        self.results_page_progress_bar.setValue(0)
        results_viewer = ResultsViewer(parent=None, trial_path=trial_dir, update_ui=False)
        plot_to_generate = results_viewer.get_plots_to_generate()
        print(f"Plots to generate: {plot_to_generate}")
        gen_return = results_viewer.store_results()
        
        for progress, status in gen_return:
            progress = int(progress / plot_to_generate * 100)
            yield progress, status
        #results_viewer.store_results()
        #yield 100, "Results stored successfully"

    def store_results_for_multiple_trials(self, trial_paths: List[str]) -> Generator[Tuple[int, str], None, None]:
        """
        Store the results for multiple trials.

        This function iterates over the provided list of trial paths, initializes a ResultsViewer object
        for each trial, stores the results, and yields the progress and status message for each trial.

        Args:
            trial_paths (List[str]): A list of directories for the trials for which results are to be stored.

        Yields:
            Tuple[int, str]: The progress value and status message for each trial.
        """
        total_plots_to_generate = 0
        for trial_path in trial_paths:
            results_viewer = ResultsViewer(parent=None, trial_path=trial_path, update_ui=False)
            total_plots_to_generate += results_viewer.get_plots_to_generate()
        print(f"Total plots to generate: {total_plots_to_generate}")
        amount_of_trials = len(trial_paths)
        
        for idx, trial_path in enumerate(trial_paths):
            results_viewer = ResultsViewer(parent=None, trial_path=trial_path, update_ui=False)
            #results_viewer.store_results()
            gen_return = results_viewer.store_results()
           
            status_trial = f"Storing results for trial {idx+1}/{amount_of_trials}: {os.path.basename(trial_path)} ..."
           
            for progress, status in gen_return:
                #progress = int(progress / total_plots_to_generate * 100)
                progress = np.floor(progress / total_plots_to_generate * 100)
                yield progress, status_trial
                
            
    def store_results_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the store results process.

        This function is called when the store results process is finished. It updates the UI to reflect the status of the process,
        including displaying a message box and updating the status label based on whether the process was successful or not.

        Args:
            value (bool): The status of the store results process. True if successful, False otherwise.

        Returns:
            None
        """
        print(f"Store results finished: {value}")
        self.results_page_progress_bar.setValue(100)
   
        if self.settings["language"] == "eng":
            QtWidgets.QMessageBox.information(self.results_page, "Info", "Results stored successfully")
            self.trial_selection_label_results.setText("Results stored successfully")
        elif self.settings["language"] == "de":
            QtWidgets.QMessageBox.information(self.results_page, "Info", "Ergebnisse erfolgreich gespeichert")
            self.trial_selection_label_results.setText("Ergebnisse erfolgreich gespeichert")
        # wait for 2 seconds
        time.sleep(2)
        self.trial_selection_label_results.setText("Select a trial to view the results")
        self.results_page_progress_bar.hide()
    

        

    def open_generate_report_dialog(self):
        """
        Open the generate report dialog
        """

        # get the selected trial
        selected_trial = self.trial_selection_combo_results_page.currentText()
        if selected_trial == "Select Trial" or selected_trial == "No Trials" or \
        selected_trial == "Keine Versuche" or selected_trial == "Versuch auswählen":
            #QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Please select a trial to generate a report")
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Please select a trial to generate a report")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Bitte wählen Sie einen Versuch aus, um einen Bericht zu generieren")
        elif selected_trial == "All Trials" or selected_trial == "Alle Versuche":
        
            # get the directories of the trials
            
            trial_dirs = [os.path.join(self.imported_files_dir, trial) for trial in self.trial_folders if trial not in self.folder_translation.values()]
        
            if len(trial_dirs) == 0:
                #QtWidgets.QMessageBox.warning(self.results_page, "Warning", "No trials to generate a report for.")
                if self.settings["language"] == "eng":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warning", "No trials to generate a report for.")
                elif self.settings["language"] == "de":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Keine Versuche, für die ein Bericht generiert werden kann.")
            elif len(trial_dirs) == 1:
                #QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Only one trial found. Please select a trial to generate a report.")
                if self.settings["language"] == "eng":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Only one trial found. Please select a trial to generate a report.")
                elif self.settings["language"] == "de":
                    QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Nur ein Versuch gefunden. Bitte wählen Sie einen Versuch aus, um einen Bericht zu generieren.")

            else:
                print(f"Generating report for all trials ...")
                # create a report for all the trials
                self.settings = Settings().items
                unit = self.settings["processing_settings"]["unit"]
                
                self.results_page_progress_bar.show()
                self.results_page_progress_bar.setValue(0)
                
                if self.settings["language"] == "eng":
                    self.trial_selection_label_results.setText("Generating report for all trials ...")
                elif self.settings["language"] == "de":
                    self.trial_selection_label_results.setText("Bericht für alle Versuche wird generiert ...")
                    
                if hasattr(self, "generate_report_thread"):
                    del self.generate_report_thread
                self.generate_report_thread = GenerateReportThread(parent=self.results_page, generate_report_function=self.generate_report_function(trial_dirs=trial_dirs, unit=unit, language=self.settings["language"]))
                self.generate_report_thread.start()
                self.generate_report_thread.finished_signal.connect(self.generate_report_finished_slot)
                self.generate_report_thread.status_signal.connect(self.trial_selection_label_results.setText)
                self.generate_report_thread.progress_signal.connect(self.results_page_progress_bar.setValue)
                self.generate_report_thread.failed_trials_signal.connect(self.failed_trials_slot)
              
                
        else:
            print(f"Selected trial: {selected_trial}")
            # get the selected trial
            selected_trial = self.trial_selection_combo_results_page.currentText()
            # set the text of the status label
            #self.trial_selection_label_results.setText("Generating report for trial {}".format(selected_trial))
            if self.settings["language"] == "eng":
                self.trial_selection_label_results.setText(f"Generating report for trial {selected_trial}")
            elif self.settings["language"] == "de":
                self.trial_selection_label_results.setText(f"Bericht für Versuch {selected_trial} wird generiert")
            trial_dir = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]

            self.results_page_progress_bar.show()   
            #self.results_page_progress_bar.setValue(0)

            if hasattr(self, "generate_report_thread"):
                del self.generate_report_thread
            self.generate_report_thread = GenerateReportThread(parent=self.results_page, generate_report_function=self.generate_report_function(trial_dirs=[trial_dir], unit=self.settings["processing_settings"]["unit"], language=self.settings["language"]))
            self.generate_report_thread.start()
            #self.generate_report_thread.finished_signal.connect(self.generate_report_finished_slot)
            self.generate_report_thread.status_signal.connect(self.trial_selection_label_results.setText)
            self.generate_report_thread.progress_signal.connect(self.results_page_progress_bar.setValue)

            

    def failed_trials_slot(self, failed_trials: List[str]) -> None:
        """
        Handle the failed trials.

        This function is called when there are failed trials during the report generation process.
        It displays a message box indicating the failed trials and updates the status label to reflect the failed trials.

        Args:
            failed_trials (List[str]): A list of trial names for which the report generation failed.

        Returns:
            None
        """
        #if self.settings["language"] == "eng":
        #    QtWidgets.QMessageBox.warning(self.results_page, "Warning", f"Failed trials: {failed_trials}")
        #elif self.settings["language"] == "de":
        #    QtWidgets.QMessageBox.warning(self.results_page, "Warnung", f"Fehlgeschlagene Versuche: {failed_trials}")
        #self.trial_selection_label_results.setText(f"Failed trials: {failed_trials}")
        #self.results_page_progress_bar.hide()
        self.failed_trials = failed_trials
        
    def generate_report_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the report generation process.

        This function is called when the report generation process is finished. It updates the UI to reflect the status of the process,
        including displaying a message box and updating the status label based on whether the process was successful or not.

        Args:
            value (bool): The status of the report generation process. True if successful, False otherwise.

        Returns:
            None
        """
        print(f"Generate report finished: {value}")
        self.results_page_progress_bar.setValue(100)
        
        if value == True:

            if hasattr(self, "failed_trials"):
                if len(self.failed_trials) > 0:
                    if self.settings["language"] == "eng":
                        self.trial_selection_label_results.setText(f"Report(s) generated, some failed: {self.failed_trials}")
                    elif self.settings["language"] == "de":
                        self.trial_selection_label_results.setText(f"Bericht(e) generiert, einige schlugen fehl: {self.failed_trials}")
                else:
                    if self.settings["language"] == "eng":
                        self.trial_selection_label_results.setText("Report(s) generated successfully")
                    elif self.settings["language"] == "de":
                        self.trial_selection_label_results.setText("Bericht(e) erfolgreich generiert")
                        
                
            # wait for 2 seconds
            time.sleep(2)
            #self.trial_selection_label_results.setText("Select a trial to view the results")
            self.results_page_progress_bar.hide()
        else:
            if self.settings["language"] == "eng":
                if hasattr(self, "failed_trials"):
                    if len(self.failed_trials) > 0:
                        self.trial_selection_label_results.setText(f"Report generation failed for trials: {self.failed_trials}")
                    else:
                        self.trial_selection_label_results.setText("Report generation failed")
            elif self.settings["language"] == "de":
                if hasattr(self, "failed_trials"):
                    if len(self.failed_trials) > 0:
                        self.trial_selection_label_results.setText(f"Berichterstellung fehlgeschlagen für Versuche: {self.failed_trials}")
                    else:
                        self.trial_selection_label_results.setText("Berichterstellung fehlgeschlagen")

        
            # wait for 2 seconds
            time.sleep(2)
            #self.trial_selection_label_results.setText("Select a trial to view the results")
            self.results_page_progress_bar.hide()
            

    
    def generate_report_function(self, trial_dirs: List[str], unit: str, language: str) -> Generator[Tuple[int, str, str, bool], None, None]:
        """
        Generate a report for a single trial.

        This function initializes the Reporting class, checks the integrity of the results file,
        extracts necessary data, creates plots and a PDF report, and saves the report in the trial directory.

        Args:
            trial_dirs (List[str]): A list of directories for the trials for which reports are to be generated.
            unit (str): Unit of measurement, either "mm" or "µm".
            language (str): Language for the report, either "eng" for English or "de" for German.

        Yields:
            Tuple[int, str, str, bool]: A tuple containing the progress percentage, status message, trial name, and success value.
        """
        failed_trials = []
        for idx, trial_dir in enumerate(trial_dirs):
            hp_params_path = os.path.join(trial_dir, "_aux_files", "hyperparameters.json")
            if not os.path.exists(hp_params_path):
                print(f"Hyperparameters file not found for trial {trial_dir}")
                continue
            reporting = Reporting(unit=unit, 
                            language=language,
                            trial_dir=trial_dir, 
                            hp_params_path=hp_params_path, 
                            verbose=False)
            trial_name = os.path.basename(trial_dir)
            if reporting.check_results_file_integrity():
                progress = int((idx) / len(trial_dirs) * 100)
                if self.settings["language"] == "eng":
                    status = f"Generating report for trial '{trial_name}' ({idx+1}/{len(trial_dirs)}) ..."
                elif self.settings["language"] == "de":
                    status = f"Bericht für Versuch '{trial_name}' wird generiert ({idx+1}/{len(trial_dirs)}) ..."
                yield progress, status, trial_name, True
            
                success_value, status, trial_name = reporting.generate_report_for_trial(unit=unit, language=language, trial_dir=trial_dir, hp_params_path=hp_params_path)
          
                progress = int((idx + 1) / len(trial_dirs) * 100)
                if success_value == True:
                    if len(failed_trials) == 0:
                        yield progress, f"Report for trial '{trial_name}' generated successfully ({idx+1}/{len(trial_dirs)}", trial_name, True
                    else:
                        yield progress, f"Report for trial '{trial_name}' generated successfully ({idx+1}/{len(trial_dirs)}) Failed trials: {failed_trials}", trial_name, True
                else:
                    failed_trials.append(trial_name)
                    print(f"Failed trials: {failed_trials}")    
                    if len(failed_trials) == 0:
                        yield progress, f"Report for trial '{trial_name}' could not be generated ({idx+1}/{len(trial_dirs)})", trial_name, False
                    else:
                        yield progress, f"Report for trial '{trial_name}' could not be generated ({idx+1}/{len(trial_dirs)}) Failed trials: {failed_trials}", trial_name, False
            
            else:
                progress = int((idx + 1) / len(trial_dirs) * 100)
                yield progress, f"Report for trial '{trial_name}' could not be generated ({idx+1}/{len(trial_dirs)})", trial_name, False
    
        
    

    

    def show_results_is_clicked(self) -> None:
        """
        Toggle the display of the results viewer.

        This function toggles between showing and hiding the results viewer based on the current state of the view results button.
        If the button text is "View Results", it opens the results viewer. If the button text is "Hide Results", it closes the results viewer.

        Returns:
            None
        """
        if self.view_results_btn.text() == "View Results":
            self.open_results_viewer()
        else:
            self.view_results_btn.setText("View Results")
            self.view_results_btn.setIcon(QIcon(Functions.set_svg_icon("view_icon.svg")))
            if hasattr(self, "results_viewer"):
                self.results_viewer.close()
                
    def open_results_viewer(self) -> None:
        """
        Open the results viewer.

        This function retrieves the selected trial from the combo box and opens the results viewer for that trial.
        If no trial or an invalid trial is selected, it displays a warning message. If the "All Trials" option is selected,
        it prompts the user to select a single trial. If the selected trial has no results, it displays a message indicating
        that there are no results to display. Otherwise, it opens the results viewer and displays the results for the selected trial.

        Returns:
            None
        """
        # get the selected trial
        selected_trial = self.trial_selection_combo_results_page.currentText()
        if selected_trial == "Select Trial" or selected_trial == "No Trials" or selected_trial == "Keine Versuche" or selected_trial == "Versuch auswählen":
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Please select a trial to view the results")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Bitte wählen Sie einen Versuch aus, um die Ergebnisse anzuzeigen")
        elif selected_trial == "All Trials" or selected_trial == "Alle Versuche":
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.results_page, "Warning", "Please select a single trial to view the results")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.results_page, "Warnung", "Bitte wählen Sie einen einzelnen Versuch aus, um die Ergebnisse anzuzeigen")
        else:
            
            """
            results_dir = os.path.join(self.imported_files_dir, selected_trial)
            prediction_dir = os.path.join(results_dir, "predictions")
       
            if not os.path.exists(prediction_dir):
                    # show a hint that there are no figures
                    if self.settings["language"] == "eng":
                        warning_name = "No Results"
                        warning_text = "No results to display for the selected trial. Please run the analysis first."
                    elif self.settings["language"] == "de":
                        warning_name = "Keine Ergebnisse"
                        warning_text = "Keine Ergebnisse zum Anzeigen für den ausgewählten Versuch. Bitte führen Sie zuerst die Analyse durch."
                    QMessageBox.critical(self.results_page, warning_name, warning_text)
            else:
                if self.settings["language"] == "eng":
                    self.trial_selection_label_results.setText(f"Viewing results for trial {selected_trial}")
                elif self.settings["language"] == "de":
                    self.trial_selection_label_results.setText(f"Ergebnisse für Versuch {selected_trial} werden angezeigt")

                self.view_results_btn.setText("Hide Results")
                self.view_results_btn.setIcon(QIcon(Functions.set_svg_icon("hide_icon.svg")))
        
                # if the results viewer is already open, close it
                if hasattr(self, "results_viewer"):
                    self.results_viewer.close()
                    # remove it from the layout
                    self.results_page_layout.removeWidget(self.results_viewer)

                # open the results viewer
                trial_path = os.path.join(self.imported_files_dir, selected_trial)
                if isinstance(trial_path, tuple):
                    trial_path = trial_path[0]
                self.results_viewer = ResultsViewer(parent=self.results_page, trial_path=trial_path, update_ui=True)
                # add the results viewer to the layout and align it to the center
                self.results_page_layout.addWidget(self.results_viewer, 0, Qt.AlignCenter)
                self.results_viewer.show()
                """
            
            self.view_results_btn.setText("Hide Results")
            self.view_results_btn.setIcon(QIcon(Functions.set_svg_icon("hide_icon.svg")))
                
            if hasattr(self, "results_viewer"):
                self.results_viewer.close()
                self.results_page_layout.removeWidget(self.results_viewer)
            
            trial_path = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_path, tuple):
                trial_path = trial_path[0]

            self.results_viewer = ResultsViewer(parent=self.results_page, trial_path=trial_path, update_ui=True)
            
            if hasattr(self.results_viewer, "prediction_dir"):
                if not os.path.exists(self.results_viewer.prediction_dir):
                    # show a hint that there are no figures
                    if self.settings["language"] == "eng":
                        warning_name = "No Results"
                        warning_text = "No results to display for the selected trial. Please run the analysis first."
                    elif self.settings["language"] == "de":
                        warning_name = "Keine Ergebnisse"
                        warning_text = "Keine Ergebnisse zum Anzeigen für den ausgewählten Versuch. Bitte führen Sie zuerst die Analyse durch."
                    QMessageBox.critical(self.results_page, warning_name, warning_text)
                    del self.results_viewer
                else:
                    if self.settings["language"] == "eng":
                        self.trial_selection_label_results.setText(f"Viewing results for trial {selected_trial}")
                    elif self.settings["language"] == "de":
                        self.trial_selection_label_results.setText(f"Ergebnisse für Versuch {selected_trial} werden angezeigt")
                     # add the results viewer to the layout and align it to the center
                    self.results_page_layout.addWidget(self.results_viewer, 0, Qt.AlignCenter)
                    self.results_viewer.show()
            else:
                if self.settings["language"] == "eng":
                    warning_name = "No Results"
                    warning_text = "No results to display for the selected trial. Please run the analysis first."
                elif self.settings["language"] == "de":
                    warning_name = "Keine Ergebnisse"
                    warning_text = "Keine Ergebnisse zum Anzeigen für den ausgewählten Versuch. Bitte führen Sie zuerst die Analyse durch."
                QMessageBox.critical(self.results_page, warning_name, warning_text)
                del self.results_viewer
            
            
            
         
            
            
                
   
    
                
                
                
                
                
                
                
                
                
                

    # ----- SETUP EMPTY PAGE ----- #
    def setup_empty_page(self):
        self.empty_page = QWidget()
        self.empty_page.setObjectName(u"empty_page")
        self.empty_page.setStyleSheet(u"QFrame {\n""	font-size: 16pt;\n""}")
        self.empty_page_layout = QVBoxLayout(self.empty_page)
        self.empty_page_layout.setObjectName(u"empty_page_layout")
        self.empty_page_label = QLabel(self.empty_page)
        self.empty_page_label.setObjectName(u"empty_page_label")
        font = QFont()
        font.setPointSize(16)
        self.empty_page_label.setFont(font)
        self.empty_page_label.setAlignment(Qt.AlignCenter)
        self.empty_page_layout.addWidget(self.empty_page_label)
        self.pages.addWidget(self.empty_page)

    # ----- SETUP DATAMANAGEMENT PAGE ----- #
    def setup_datamanagement_page(self) -> None:
        """
        Set up the data management page.

        This function initializes and configures the data management page, including the layout, 
        trial selection combo box, progress bar for export operations, and buttons for exporting results, 
        renaming trials, deleting predictions, and deleting trials.
        """
        self.datamanagement_page = QWidget()
        self.datamanagement_page.setObjectName(u"datamanagement_page")
        self.datamanagement_page.setStyleSheet(u"font-size: 14pt")
        self.datamanagement_page_layout = QVBoxLayout(self.datamanagement_page)
        self.datamanagement_page_layout.setObjectName(u"datamanagement_page_layout")
        self.datamanagement_page_label = QLabel(self.datamanagement_page)
        self.datamanagement_page_label.setObjectName(u"datamanagement_page_label")
        self.datamanagement_page_label.setAlignment(Qt.AlignCenter)
        self.datamanagement_page_layout.addWidget(self.datamanagement_page_label)
        self.pages.addWidget(self.datamanagement_page)

        # add a combo box to select the trial
        self.datamanagement_page_label.setText("Data Management Page")
        self.datamanagement_page_layout.addWidget(self.datamanagement_page_label)
        self.datamanagement_page_layout.setAlignment(Qt.AlignCenter)
        self.datamanagement_page_layout.addSpacing(20)

        # add pixmap
        self.datamanagement_page_pixmap = QLabel(self.datamanagement_page)
        self.datamanagement_page_pixmap.setObjectName(u"datamanagement_page_pixmap")
        self.datamanagement_page_pixmap.setAlignment(Qt.AlignCenter)
        self.datamanagement_page_pixmap.setPixmap(QPixmap(Functions.set_image(Functions.set_svg_icon("data_management.svg"))))
        self.datamanagement_page_layout.addWidget(self.datamanagement_page_pixmap, 0, Qt.AlignCenter)

        self.trial_selection_label_datamanagement = QLabel(self.datamanagement_page)
        self.trial_selection_label_datamanagement.setObjectName(u"trial_selection_label_datamanagement")
        self.trial_selection_label_datamanagement.setAlignment(Qt.AlignCenter)
        self.trial_selection_label_datamanagement.setText("Select a trial to manage")
        self.datamanagement_page_layout.addWidget(self.trial_selection_label_datamanagement, 0, Qt.AlignCenter)

        # create a combo box to select the trial
        self.trial_selection_combo_datamanagement_page = PyComboBox(self.datamanagement_page)
        self.trial_selection_combo_datamanagement_page.setMaximumWidth(600)
        self.trial_selection_combo_datamanagement_page.setObjectName(u"trial_selection_combo_datamanagement_page")
        #self.trial_selection_combo_datamanagement_page.addItem("Select Trial")
        self.trial_selection_combo_datamanagement_page.addItems(self.get_trial_folders())
        self.datamanagement_page_layout.addWidget(self.trial_selection_combo_datamanagement_page, 0, Qt.AlignHCenter)

        # add a progress bar to show the progress of the export; hide it initially but show it when the export starts
        self.data_management_progress_bar = PYProgressBar(parent=self.datamanagement_page,
                                        color = "black",
                                        bg_color = self.themes["app_color"]["context_color"],
                                        border_color=self.themes["app_color"]["white"],
                                        border_radius="5px")
        self.data_management_progress_bar.setMaximumSize(QSize(600, 100))
        self.data_management_progress_bar.setObjectName(u"data_management_progress_bar")
        self.data_management_progress_bar.setOrientation(Qt.Horizontal)
        self.data_management_progress_bar.setRange(0, 100)
        self.data_management_progress_bar.setValue(0)
        self.data_management_progress_bar.hide()
        self.data_management_progress_bar.setAlignment(Qt.AlignHCenter)
        self.datamanagement_page_layout.addWidget(self.data_management_progress_bar, 0, Qt.AlignHCenter)
        self.datamanagement_page_layout.addSpacing(5)

        # add a label to show the status of the export
        self.data_management_status_label = QLabel(self.datamanagement_page)
        self.data_management_status_label.setObjectName(u"data_management_status_label")
        self.data_management_status_label.setAlignment(Qt.AlignCenter)
        self.data_management_status_label.setText(" ")
        self.datamanagement_page_layout.addWidget(self.data_management_status_label)
        self.datamanagement_page_layout.addSpacing(5)

        # add a button to export the results
        if self.settings["language"] == "eng":
            text = "Export Results"
        elif self.settings["language"] == "de":
            text = "Ergebnisse exportieren"
        self.export_results_btn = PyPushButton(
            parent=self.datamanagement_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.export_results_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.datamanagement_page_layout.addWidget(self.export_results_btn, 0, Qt.AlignCenter)
        self.export_results_btn.clicked.connect(self.export_results)

        self.export_results_btn_icon = QIcon(Functions.set_svg_icon("export_white.svg"))
        self.export_results_btn.setIcon(self.export_results_btn_icon)
        self.export_results_btn.setIconSize(QSize(30, 30))

        # add a button to rename the selected trial
        if self.settings["language"] == "eng":
            text = "Rename Trial"
        elif self.settings["language"] == "de":
            text = "Versuch umbenennen"
        self.rename_trial_btn = PyPushButton(
            parent=self.datamanagement_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.rename_trial_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.datamanagement_page_layout.addWidget(self.rename_trial_btn, 0, Qt.AlignCenter)
        self.rename_trial_btn.clicked.connect(self.rename_trial)
        self.rename_trial_btn_icon = QIcon(Functions.set_svg_icon("rename_white.svg"))
        self.rename_trial_btn.setIcon(self.rename_trial_btn_icon)
        self.rename_trial_btn.setIconSize(QSize(30, 30))

        # add a button to delete only the predictions of the selected trial
        if self.settings["language"] == "eng":
            text = "Delete Predictions"
        elif self.settings["language"] == "de":
            text = "Prognosen löschen"
        self.delete_predictions_btn = PyPushButton(
            parent=self.datamanagement_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.delete_predictions_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.delete_predictions_btn.clicked.connect(self.delete_predictions)
        self.delete_predictions_btn_icon = QIcon(Functions.set_svg_icon("delete_white.svg"))
        self.delete_predictions_btn.setIcon(self.delete_predictions_btn_icon)
        self.delete_predictions_btn.setIconSize(QSize(30, 30))

        # add a button to delete the selected trial
        if self.settings["language"] == "eng":
            text = "Delete Trial"
        elif self.settings["language"] == "de":
            text =  "Versuch löschen"
        self.delete_trial_btn = PyPushButton(
            parent=self.datamanagement_page,
            text=text,
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.delete_trial_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))

        self.delete_trial_btn.clicked.connect(self.delete_trial)
        self.delete_trial_btn_icon = QIcon(Functions.set_svg_icon("delete_white.svg"))
        self.delete_trial_btn.setIcon(self.delete_trial_btn_icon)
        self.delete_trial_btn.setIconSize(QSize(30, 30))

        button_layout = QHBoxLayout()   
        button_layout.addWidget(self.export_results_btn)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.rename_trial_btn, 0, Qt.AlignCenter)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.delete_predictions_btn)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.delete_trial_btn)
        self.datamanagement_page_layout.addLayout(button_layout)
        self.datamanagement_page_layout.setAlignment(Qt.AlignCenter)
        self.datamanagement_page_layout.addSpacing(50)

        # add the page to the stack
        self.pages.addWidget(self.datamanagement_page)

    def delete_predictions(self) -> None:
        """
        Delete the predictions of the selected trial.

        This function deletes the predictions of the selected trial from the data management page.
        If "All Trials" is selected, it deletes the predictions for all trials. The function updates
        the status label and uses a separate thread to perform the deletion operation.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial is selected or if there are no trials to delete predictions for.
        """
        # get the selected trial
        selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
        if selected_trial in ["Select Trial", "No Trials", "Keine Versuche", "Versuch auswählen"]:
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.datamanagement_page, "Warning", "Please select a trial to delete the predictions")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.datamanagement_page, "Warnung", "Bitte wählen Sie einen Versuch aus, um die Prognosen zu löschen")
        elif selected_trial in ["All Trials", "Alle Versuche"]:
            # ask the user if they are sure they want to delete all the predictions
            if self.settings["language"] == "eng":
                message = "Are you sure you want to delete all the predictions?"
            elif self.settings["language"] == "de":
                message = "Möchten Sie wirklich alle Prognosen löschen?"
            reply = QtWidgets.QMessageBox.question(self.datamanagement_page, "Message", message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                # delete the predictions of all the trials
                self.delete_predictions_thread = DeleteThread(parent=self.datamanagement_page, delete_function=self.delete_predictions_for_multiple_trials())
                self.delete_predictions_thread.start()
                self.delete_predictions_thread.finished_signal.connect(self.delete_predictions_finished_slot)
                self.delete_predictions_thread.progress_signal.connect(self.data_management_progress_bar.setValue)
                self.delete_predictions_thread.status_signal.connect(self.data_management_status_label.setText)
        else:
            # get the selected trial
            selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
            # set the text of the status label
            if self.settings["language"] == "eng":
                self.data_management_status_label.setText(f"Deleting predictions for trial {selected_trial}")
            elif self.settings["language"] == "de":
                self.data_management_status_label.setText(f"Prognosen für Versuch {selected_trial} werden gelöscht")
            trial_dir = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]
            # delete the predictions
            self.delete_predictions_thread = DeleteThread(parent=self.datamanagement_page, delete_function=self.delete_predictions_function(trial_dir))
            self.delete_predictions_thread.start()
            self.delete_predictions_thread.finished_signal.connect(self.delete_predictions_finished_slot)
            self.delete_predictions_thread.status_signal.connect(self.data_management_status_label.setText)
            
    def delete_predictions_function(self, trial_dir: str) -> Generator[Tuple[int, str], None, None]:
        """
        Delete the predictions of the selected trial.

        This function deletes various prediction-related files and directories within the specified trial directory.
        It removes the "predictions" directory, "coco" directory, "instances_results.xlsx" file, "image_analysis_results.json" file,
        "instances_results.csv" file, and "segmentation_masks" directory if they exist.

        Args:
            trial_dir (str): The directory of the trial for which predictions are to be deleted.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        image_analysis_dir = os.path.join(trial_dir, "01_image_analysis")
        if os.path.exists(image_analysis_dir):
            shutil.rmtree(image_analysis_dir)
            
        image_analysis_json_path = os.path.join(trial_dir, "_aux_files", "image_analysis_results.json")
        if os.path.exists(image_analysis_json_path):
            os.remove(image_analysis_json_path)
        
        yield 100, "Predictions deleted successfully"
        
    def delete_predictions_for_multiple_trials(self) -> Generator[Tuple[int, str], None, None]:
        """
        Delete the predictions for multiple trials.

        This function iterates over the list of trial folders and deletes the predictions for each trial.
        It updates the status label with the progress and status message for each trial.

        Yields:
            Tuple[int, str]: The progress value and status message for each trial.
        """
        for idx, trial in enumerate(self.get_trial_folders()):
            trial_dir = os.path.join(self.imported_files_dir, trial)
            if self.settings["language"] == "eng":
                text = f"Deleting predictions for trial {trial}" + f" ({idx + 1}/{len(self.get_trial_folders())})"
                self.data_management_status_label.setText(text)
            elif self.settings["language"] == "de":
                text = f"Prognosen für Versuch {trial} werden gelöscht" + f" ({idx + 1}/{len(self.get_trial_folders())})"
                self.data_management_status_label.setText(text)
            
            predictions_dir = os.path.join(trial_dir, "predictions")
            if os.path.exists(predictions_dir):
                shutil.rmtree(predictions_dir)
            progress = (idx + 1) / len(self.get_trial_folders()) * 100
            yield progress, f"Predictions deleted successfully for {trial}"
        
    def delete_predictions_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the delete predictions process.

        This function is called when the delete predictions process is finished. It updates the UI to reflect the status of the process,
        including displaying a message box and updating the status label based on whether the process was successful or not.

        Args:
            value (bool): The status of the delete predictions process. True if successful, False otherwise.

        Returns:
            None
        """
        print(f"Delete predictions finished: {value}")
        self.data_management_progress_bar.hide()
        if value == False:
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.warning(self.datamanagement_page, "Warning", "Failed to delete the AI-Predictions")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.warning(self.datamanagement_page, "Warnung", "KI-Vorhersagen konnten nicht gelöscht werden")
        else:
            if self.settings["language"] == "eng":
                QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "AI-Predictions deleted successfully")
            elif self.settings["language"] == "de":
                QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "KI-Vorhersagen erfolgreich gelöscht")

        # wait for a few seconds before hiding the status label
        time.sleep(2)
        self.data_management_status_label.setText(" ")
        
    def export_results(self) -> None:
        """
        Export the results of the selected trial.

        This function exports the results of the selected trial to a user-specified directory.
        If "All Trials" is selected, it exports the results for all trials. The function updates
        the status label and uses a separate thread to perform the export operation.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial is selected or if there are no trials to export.
        """
        # get the selected trial
        selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
        if selected_trial == "Select Trial" or selected_trial == "Versuch auswählen":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "Please select a trial to export"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Bitte wählen Sie einen Versuch zum Exportieren"
            QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)
        elif selected_trial == "No Trials" or selected_trial == "Keine Versuche":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "No trials to export"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Keine Versuche zum Exportieren"
            QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)
        elif selected_trial == "All Trials" or selected_trial == "Alle Versuche":

            # get the destination directory
            if self.settings["language"] == "eng":
                message = "Select a directory to save the results"
            elif self.settings["language"] == "de":
                message = "Wählen Sie ein Verzeichnis, um die Ergebnisse zu speichern"
            destination_dir = QFileDialog.getExistingDirectory(self.datamanagement_page, message, options=QFileDialog.ShowDirsOnly)
            final_destination_dir = os.path.join(destination_dir)
            
            # set label to "Exporting results ..."
            if self.settings["language"] == "eng":
                self.data_management_status_label.setText("Exporting results ...")
            elif self.settings["language"] == "de":
                self.data_management_status_label.setText("Ergebnisse werden exportiert ...")
                
            self.data_management_progress_bar.show()
            
            # create a export thread
            self.export_thread = ExportThread(parent=self.datamanagement_page, export_function=self.export_results_directories(final_destination_dir))
            self.export_thread.start()
            self.export_thread.finished_signal.connect(self.export_finished_slot)
            self.export_thread.progress_signal.connect(self.data_management_progress_bar.setValue)
            self.export_thread.status_signal.connect(self.data_management_status_label.setText)

            # remove the trial from the combo box
            self.refresh_trial_folder_names()
        else:
            # get the selected trial
            selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
            trial_dir = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]

            # get the destination directory
            if self.settings["language"] == "eng":
                message = f"Export results for trial {selected_trial} to the selected directory"
            elif self.settings["language"] == "de":
                message = f"Exportieren Sie die Ergebnisse für den Versuch {selected_trial} in das ausgewählte Verzeichnis"
                
            destination_dir = QFileDialog.getExistingDirectory(self.datamanagement_page, message, os.path.expanduser("~/Downloads"), options=QFileDialog.ShowDirsOnly)
            final_destination_dir = os.path.join(destination_dir, selected_trial)
            
            # set label to "Exporting results ..."
            if self.settings["language"] == "eng":
                self.data_management_status_label.setText(f"Exporting results for trial {selected_trial}")
            elif self.settings["language"] == "de":
                self.data_management_status_label.setText(f"Exportiere Ergebnisse für Versuch {selected_trial}")
                
            self.data_management_progress_bar.show()

            # create a export thread
            self.export_thread = ExportThread(parent=self.datamanagement_page, export_function=self.export_results_directory(trial_dir, final_destination_dir))
            self.export_thread.start()
            self.export_thread.finished_signal.connect(self.export_finished_slot)
            self.export_thread.progress_signal.connect(self.data_management_progress_bar.setValue)
            self.export_thread.status_signal.connect(self.data_management_status_label.setText)

            # remove the trial from the combo box
            self.refresh_trial_folder_names()
        
    def export_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the export process.

        This function is called when the export process is finished. It updates the UI to reflect the status of the process,
        including displaying a message box and updating the status label based on whether the process was successful or not.

        Args:
            value (bool): The status of the export process. True if successful, False otherwise.

        Returns:
            None
        """
        print(f"Export finished: {value}")
        self.data_management_progress_bar.hide()
        if self.settings["language"] == "eng":
            self.data_management_status_label.setText("Export completed")
            QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "Export completed")
        elif self.settings["language"] == "de":
            self.data_management_status_label.setText("Export abgeschlossen")
            QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "Export abgeschlossen")

        # wait for a few seconds before hiding the status label
        time.sleep(2)
        self.data_management_status_label.setText(" ")



    def export_results_directory(self, trial_dir: str, destination_dir: str) -> Generator[Tuple[int, str], None, None]:
        if self.settings["language"] == "eng":
            status = f"Exporting results for trial {os.path.basename(trial_dir)}"
        elif self.settings["language"] == "de":
            status = f"Exportiere Ergebnisse für Versuch {os.path.basename(trial_dir)}"
        
        if destination_dir:
            # Create a directory for the trial in the destination directory
            destination_dir = os.path.join(destination_dir, os.path.basename(trial_dir))
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            # Count total files for progress tracking
            total_files = sum(len(files) for _, _, files in os.walk(trial_dir))
            copied_files = 0

            # Copy files and directories
            for root, dirs, files in os.walk(trial_dir):
                rel_path = os.path.relpath(root, trial_dir)
                dest_path = os.path.join(destination_dir, rel_path)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_path, file)
                    shutil.copy2(src_file, dest_file)
                    copied_files += 1
                    progress = int((copied_files / total_files) * 100)
                    yield progress, status
        else:
            if self.settings["language"] == "eng":
                status = "Export failed: No destination directory selected"
            elif self.settings["language"] == "de":
                status = "Export fehlgeschlagen: Kein Zielverzeichnis ausgewählt"
            yield 0, status

        if self.settings["language"] == "eng":
            final_status = "Export completed"
        elif self.settings["language"] == "de":
            final_status = "Export abgeschlossen"

        yield 100, final_status
        
    def export_results_directories(self, destination_dir: str) -> Generator[Tuple[int, str], None, None]:
        """
        Export the results of all trials to the specified destination directory.

        This function iterates over the list of trial directories, creates corresponding directories
        in the destination directory, and copies the results of each trial to the destination directory.
        It updates the progress and status messages during the export process.

        Args:
            destination_dir (str): The directory where the trial results will be exported.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        trial_dirs = [os.path.join(self.imported_files_dir, trial) for trial in self.get_trial_folders() if trial not in self.folder_translation.values()]

        for trial_dir in trial_dirs:
            # Create a directory for the trial in the destination directory
            process = int((trial_dirs.index(trial_dir) + 1) / len(trial_dirs) * 100)
            if self.settings["language"] == "eng":
                status = f"Exporting results for trial {os.path.basename(trial_dir)} ({int(trial_dirs.index(trial_dir) + 1)}/{len(trial_dirs)}) ..."
            elif self.settings["language"] == "de":
                status = f"Exportiere Ergebnisse für Versuch {os.path.basename(trial_dir)} ({int(trial_dirs.index(trial_dir) + 1)}/{len(trial_dirs)}) ..."
            yield process, status

            trial_name = os.path.basename(trial_dir)
            final_destination_dir = os.path.join(destination_dir, trial_name)
            if not os.path.exists(final_destination_dir):
                os.makedirs(final_destination_dir)
            # Copy the results to the selected directory
            shutil.copytree(trial_dir, final_destination_dir, dirs_exist_ok=True)

        # Set the final status
        if self.settings["language"] == "eng":
            final_status = "Export completed"
        elif self.settings["language"] == "de":
            final_status = "Export abgeschlossen"
        else:
            final_status = "Export completed"
        yield 100, final_status
        
    def rename_trial(self) -> None:
        """
        Rename the selected trial.

        This function allows the user to rename the selected trial. It prompts the user to enter a new name for the trial,
        renames the trial directory, and refreshes the trial folder names in the combo box. If the renaming process fails,
        it displays an appropriate warning message.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial is selected or if the renaming process fails.
            QtWidgets.QMessageBox.information: If the trial is renamed successfully.
        """
        # get the selected trial
        selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
        if selected_trial == "Select Trial" or selected_trial == "No Trials" or selected_trial == "Keine Versuche":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "Please select a trial to rename"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Bitte wählen Sie einen Versuch zum Umbenennen"
            QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)
        else:
            # get the new name of the trial
            if self.settings["language"] == "eng":
                new_trial_name, ok = QtWidgets.QInputDialog.getText(self.datamanagement_page, "Rename Trial", "Enter the new name of the trial")
            elif self.settings["language"] == "de":
                new_trial_name, ok = QtWidgets.QInputDialog.getText(self.datamanagement_page, "Versuch umbenennen", "Geben Sie den neuen Namen des Versuchs ein")
            if ok:
                # get the directory of the selected trial
                trial_dir = os.path.join(self.imported_files_dir, selected_trial)
                if isinstance(trial_dir, tuple):
                    trial_dir = trial_dir[0]
                # get the new directory
                new_trial_dir = os.path.join(self.imported_files_dir, new_trial_name)
                # rename the trial
                try:
                    os.rename(trial_dir, new_trial_dir)
                    # refresh the trial folder names
                    self.refresh_trial_folder_names()
                    # show a message box
                    if self.settings["language"] == "eng":
                        warning_name = "Info"
                        warning_message = "Trial renamed successfully"
                    elif self.settings["language"] == "de":
                        warning_name = "Info"
                        warning_message = "Versuch erfolgreich umbenannt"
                    QtWidgets.QMessageBox.information(self.datamanagement_page, warning_name, warning_message)	
                except Exception as e:
                    if self.settings["language"] == "eng":
                        warning_name = "Warning"
                        warning_message = f"Failed to rename the trial: {e}"
                    elif self.settings["language"] == "de":
                        warning_name = "Warnung"
                        warning_message = f"Versuch konnte nicht umbenannt werden: {e}"
                    QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)

    def delete_trial(self) -> None:
        """
        Delete the selected trial.

        This function deletes the selected trial from the data management page. If "All Trials" is selected,
        it deletes all trials. The function updates the status label and uses a separate thread to perform
        the deletion operation.

        Raises:
            QtWidgets.QMessageBox.warning: If no trial is selected or if there are no trials to delete.
            QtWidgets.QMessageBox.question: If the user is prompted to confirm the deletion of all trials or a specific trial.
        """
        # remove the trial from the combo box
        #self.refresh_trial_folder_names()
        # get the selected trial
        selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
        print(f"Selected trial current text: {selected_trial}")
        if selected_trial == "Select Trial":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "Please select a trial to delete"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Bitte wählen Sie einen Versuch zum Löschen aus"
            QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)
        elif selected_trial == "All Trials":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "Are you sure you want to delete ALL trials?"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Sind Sie sicher, dass Sie ALLE Versuche löschen möchten?"
            if QtWidgets.QMessageBox.question(self.datamanagement_page, warning_name, warning_message) == QtWidgets.QMessageBox.Yes:
                # create a delete thread
                self.delete_thread = DeleteThread(parent=self.datamanagement_page, delete_function=self.delete_trial_directories())
                self.delete_thread.start()
                self.delete_thread.finished_signal.connect(self.delete_finished_slot)
                self.delete_thread.status_signal.connect(self.data_management_status_label.setText)       
            else:
                return
        elif selected_trial == "No Trials" or selected_trial == "Keine Versuche":
            if self.settings["language"] == "eng":
                warning_name = "Warning"
                warning_message = "No trials to delete"
            elif self.settings["language"] == "de":
                warning_name = "Warnung"
                warning_message = "Keine Versuche zum Löschen"
            QtWidgets.QMessageBox.warning(self.datamanagement_page, warning_name, warning_message)
        else:
            # get the selected trial
            selected_trial = self.trial_selection_combo_datamanagement_page.currentText()
            trial_dir = os.path.join(self.imported_files_dir, selected_trial)
            if isinstance(trial_dir, tuple):
                trial_dir = trial_dir[0]

            if self.settings["language"] == "eng":
                trial_dir_message = f"Are you sure you want to delete the trial '{os.path.basename(trial_dir)}'?"
                trial_dir_message_name = "Delete Trial"
            elif self.settings["language"] == "de":
                trial_dir_message = f"Sind Sie sicher, dass Sie den Versuch '{os.path.basename(trial_dir)}' löschen möchten?"
                trial_dir_message_name = "Versuch löschen"

            if QtWidgets.QMessageBox.question(self.datamanagement_page, trial_dir_message_name, trial_dir_message) == QtWidgets.QMessageBox.Yes:
                
                if self.settings["language"] == "eng":
                    self.data_management_status_label.setText(f"Deleting trial {selected_trial}")
                elif self.settings["language"] == "de":
                    self.data_management_status_label.setText(f"Versuch {selected_trial} wird gelöscht")
                
                # create a delete thread
                self.data_management_progress_bar.show()
                self.delete_thread = DeleteThread(parent=self.datamanagement_page, delete_function=self.delete_trial_directory(trial_dir))
                self.delete_thread.start()
                self.delete_thread.finished_signal.connect(self.delete_finished_slot)
                self.delete_thread.status_signal.connect(self.data_management_status_label.setText)

            else:
                return
            
    def delete_finished_slot(self, value: bool) -> None:
        """
        Handle the completion of the delete process.

        This function is called when the delete process is finished. It updates the UI to reflect the status of the process,
        including displaying a message box and updating the status label based on whether the process was successful or not.

        Args:
            value (bool): The status of the delete process. True if successful, False otherwise.

        Returns:
            None
        """
        self.data_management_progress_bar.hide()
        print(f"Delete finished: {value}")
        #self.data_management_progress_bar.hide()
        if self.settings["language"] == "eng":
            self.data_management_status_label.setText("Delete completed")
            QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "Delete completed")
        elif self.settings["language"] == "de":
            self.data_management_status_label.setText("Löschen abgeschlossen")
            QtWidgets.QMessageBox.information(self.datamanagement_page, "Info", "Löschen abgeschlossen")
        # wait for a few seconds before hiding the status label
        time.sleep(2)
        self.data_management_status_label.setText(" ")
        self.refresh_trial_folder_names()

    def delete_trial_directory(self, trial_dir: str) -> Generator[Tuple[int, str], None, None]:
        """
        Delete the selected trial directory.

        This function deletes the specified trial directory and updates the status of the delete process.

        Args:
            trial_dir (str): The directory of the trial to be deleted.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        # delete the trial directory
        
        # Count total files for progress tracking
        total_files = sum(len(files) for _, _, files in os.walk(trial_dir))
        deleted_files = 0
        
        status = f"Deleting trial {os.path.basename(trial_dir)}"

        # Copy files and directories
        for root, dirs, files in os.walk(trial_dir):
            for file in files:
                src_file = os.path.join(root, file)
                os.remove(src_file)
                deleted_files += 1
                progress = int((deleted_files / total_files) * 100)
                yield progress, status
        
        shutil.rmtree(trial_dir)
        if self.settings["language"] == "eng":
            yield 100, "Delete completed"
        elif self.settings["language"] == "de":
            yield 100, "Löschen abgeschlossen"
            
    def delete_trial_directories(self) -> Generator[Tuple[int, str], None, None]:
        """
        Delete the selected trial directories.

        This function iterates over the list of trial directories and deletes each one.
        It updates the progress and status messages during the deletion process.

        Yields:
            Tuple[int, str]: The progress value and status message.
        """
        # Get the destination directory
        trial_dirs = [os.path.join(self.imported_files_dir, trial) for trial in self.get_trial_folders() if trial not in self.folder_translation.values()]
        for trial_dir in trial_dirs:
            process = int((trial_dirs.index(trial_dir) + 1) / len(trial_dirs) * 100)
            if self.settings["language"] == "eng":
                status = f"Deleting trial {trial_dir}" + f" ({trial_dirs.index(trial_dir) + 1}/{len(trial_dirs)})"
            elif self.settings["language"] == "de":
                status = f"Lösche Versuch {trial_dir}" + f" ({trial_dirs.index(trial_dir) + 1}/{len(trial_dirs)})"
            yield process, status
            try:
                shutil.rmtree(trial_dir)
            except Exception as e:
                print(f"Error deleting trial: {e}")

                if self.settings["language"] == "eng":
                    warning_name = "Warning"
                    warning_message = f"Failed to delete trial {os.path.basename(trial_dir)}"
                elif self.settings["language"] == "de":
                    warning_name = "Warnung"
                    warning_message = f"Versuch {os.path.basename(trial_dir)} konnte nicht gelöscht werden"
                yield 100, warning_message

            if self.settings["language"] == "eng":
                status = f"Deleting trial {trial_dirs.index(trial_dir) + 1} of {len(trial_dirs)}"
            elif self.settings["language"] == "de":
                status = f"Lösche Versuch {trial_dirs.index(trial_dir) + 1} von {len(trial_dirs)}"
            yield process, status

        if self.settings["language"] == "eng":
            yield 100, "Delete completed"
        elif self.settings["language"] == "de":
            yield 100, "Löschen abgeschlossen"


    # ----- SETUP TRAINER PAGE ----- #
    '''    
    def refresh_train_data_dirs(self) -> None:
        """
        #Refresh the directories of the trials to train the AI model on.

        #This function clears the current items in the trial selection combo box for the trainer page
        #and repopulates it with the updated list of training data directories.
        """
        self.trial_selection_combo_trainer_page.clear()
        self.trial_selection_combo_trainer_page.addItems(self.get_train_data_dirs())
    
    def get_train_data_dirs(self) -> list:
        """
        #List all the directories in the train data directory.

        #This function retrieves the list of directories within the training data directory.
        #If no directories are found, it returns a list with a message indicating no trials are available.
        #Otherwise, it returns a list of directories prefixed with a selection prompt.

        #Returns:
        #    list: A list of directory names or a message indicating no trials are available.
        """
        train_data_dirs = [dir for dir in os.listdir(self.ai_train_dir) if os.path.isdir(os.path.join(self.ai_train_dir, dir))]
        if len(train_data_dirs) == 0:
            if self.settings["language"] == "eng":
                train_data_dirs = ["No Trials"]
            elif self.settings["language"] == "de":
                train_data_dirs = ["Keine Versuche"]
        else:
            if self.settings["language"] == "eng":
                train_data_dirs = ["Select Trial"] + train_data_dirs
            elif self.settings["language"] == "de":
                train_data_dirs = ["Versuch auswählen"] + train_data_dirs
        return train_data_dirs

    def setup_trainer_page(self):
        self.ai_train_dir = "ai_train_datasets"
        if not os.path.exists(self.ai_train_dir):
            os.makedirs(self.ai_train_dir)

        self.trainer_page = QWidget()
        self.trainer_page.setObjectName(u"trainer_page")
        self.trainer_page.setStyleSheet(u"font-size: 14pt")
        self.trainer_page_layout = QVBoxLayout(self.trainer_page)
        self.trainer_page_layout.setObjectName(u"trainer_page_layout")
        self.trainer_page_label = QLabel(self.trainer_page)
        self.trainer_page_label.setObjectName(u"trainer_page_label")
        self.trainer_page_label.setAlignment(Qt.AlignCenter)
        self.trainer_page_layout.addWidget(self.trainer_page_label)
        self.pages.addWidget(self.trainer_page)

        # add a combo box to select the trial
        self.trainer_page_label.setText("Trainer Page")
        self.trainer_page_layout.addWidget(self.trainer_page_label)
        self.trainer_page_layout.setAlignment(Qt.AlignCenter)
        self.trainer_page_layout.addSpacing(20)

        # add pixmap
        self.trainer_page_pixmap = QLabel(self.trainer_page)
        self.trainer_page_pixmap.setObjectName(u"trainer_page_pixmap")
        self.trainer_page_pixmap.setAlignment(Qt.AlignCenter)
        self.trainer_page_pixmap.setPixmap(QPixmap(Functions.set_image(Functions.set_svg_icon("AI_icon_grey.svg"))))
        self.trainer_page_layout.addWidget(self.trainer_page_pixmap)



        self.trial_selection_label_trainer = QLabel(self.trainer_page)
        self.trial_selection_label_trainer.setObjectName(u"trial_selection_label_trainer")
        self.trial_selection_label_trainer.setAlignment(Qt.AlignCenter)
        self.trial_selection_label_trainer.setText("Select a trial to train the AI model on")
        self.trainer_page_layout.addWidget(self.trial_selection_label_trainer)

        # create a combo box to select the trial
        self.trial_selection_combo_trainer_page = PyComboBox(self.trainer_page)
        self.trial_selection_combo_trainer_page.setGeometry(QRect(0, 0, 200, 500))
        self.trial_selection_combo_trainer_page.setObjectName(u"trial_selection_combo_trainer_page")
        #self.trial_selection_combo_trainer_page.addItem("Select Trial")
        self.trial_selection_combo_trainer_page.addItems(self.get_train_data_dirs())
        self.trainer_page_layout.addWidget(self.trial_selection_combo_trainer_page, 0, Qt.AlignHCenter)

        # add a label indicating the ETA of the training
        #self.training_ETA_label = QLabel(self.trainer_page)
        #self.training_ETA_label.setObjectName(u"training_ETA_label")
        #self.training_ETA_label.setAlignment(Qt.AlignCenter)
        #self.training_ETA_label.setText(" ")
        #self.trainer_page_layout.addWidget(self.training_ETA_label)
        #self.trainer_page_layout.addSpacing(5)

        # add a progress bar to show the progress of the training; hide it initially but show it when the training starts
        self.training_progress_bar = PYProgressBar(parent=self.trainer_page,
                                        color = "black",
                                        bg_color = self.themes["app_color"]["context_color"],
                                        border_color=self.themes["app_color"]["white"],
                                        border_radius="5px")
        self.training_progress_bar.setMaximumSize(QSize(600, 100))
        self.training_progress_bar.setObjectName(u"training_progress_bar")
        self.training_progress_bar.setOrientation(Qt.Horizontal)
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.hide()
        self.training_progress_bar.setAlignment(Qt.AlignHCenter)
        self.trainer_page_layout.addWidget(self.training_progress_bar, 0, Qt.AlignHCenter)
        self.trainer_page_layout.addSpacing(5)
        
        # add a label to show the status of the training
        self.training_status_label = QLabel(self.trainer_page)
        self.training_status_label.setObjectName(u"training_status_label")
        self.training_status_label.setAlignment(Qt.AlignCenter)
        self.training_status_label.setText(" ")

        self.trainer_page_layout.addWidget(self.training_status_label)
        self.trainer_page_layout.addSpacing(5)

        # add a button to open the nuclei pre-annotation tool
        self.open_nuclei_annotation_tool_btn = PyPushButton(
            parent=self.trainer_page,
            text="Open Nuclei Annotation Tool",
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.open_nuclei_annotation_tool_btn_icon = QIcon(Functions.set_svg_icon("nucleus_white.svg"))
        self.open_nuclei_annotation_tool_btn.setIcon(self.open_nuclei_annotation_tool_btn_icon)
        self.open_nuclei_annotation_tool_btn.setIconSize(QSize(30, 30))

        self.open_nuclei_annotation_tool_btn.setGeometry(QtCore.QRect(25, 25, 200, 100))
        self.trainer_page_layout.addWidget(self.open_nuclei_annotation_tool_btn, 0, Qt.AlignCenter)
        self.open_nuclei_annotation_tool_btn.clicked.connect(self.open_nuclei_annotation_toolbox)



        # add a button to train the AI model
        self.train_model_btn = PyPushButton(
            parent=self.trainer_page,
            text="Train Model",
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.train_model_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.trainer_page_layout.addWidget(self.train_model_btn, 0, Qt.AlignCenter)
        self.trainer_page_layout.setSpacing(50)
        self.train_model_btn.clicked.connect(self.train_model)

        self.train_model_btn_icon = QIcon(Functions.set_svg_icon("trainer_white.svg"))
        self.train_model_btn.setIcon(self.train_model_btn_icon)
        self.train_model_btn.setIconSize(QSize(30, 30))



        # add a buttun to stop the training
        self.stop_training_btn = PyPushButton(
            parent=self.trainer_page,
            text="Stop Training",
            radius=8,
            color=self.themes["app_color"]["text_foreground"],
            bg_color=self.themes["app_color"]["dark_one"],
            bg_color_hover=self.themes["app_color"]["dark_three"],
            bg_color_pressed=self.themes["app_color"]["dark_four"])
        self.stop_training_btn.setGeometry(QtCore.QRect(25, 25, 400, 200))
        self.trainer_page_layout.addWidget(self.stop_training_btn, 0, Qt.AlignCenter)
        self.trainer_page_layout.setSpacing(50)
        self.stop_training_btn.clicked.connect(self.stop_training)

        self.stop_training_btn_icon = QIcon(Functions.set_svg_icon("stop_icon.svg"))
        self.stop_training_btn.setIcon(self.stop_training_btn_icon)
        self.stop_training_btn.setIconSize(QSize(30, 30))


        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_nuclei_annotation_tool_btn)
        button_layout.addSpacing(200)
        button_layout.addWidget(self.train_model_btn)
        button_layout.addSpacing(200)
        button_layout.addWidget(self.stop_training_btn)
        self.trainer_page_layout.addLayout(button_layout)
        self.trainer_page_layout.setAlignment(Qt.AlignCenter)
        self.trainer_page_layout.addSpacing(50)




        # add the page to the stack
        self.pages.addWidget(self.trainer_page)

    def open_nuclei_annotation_toolbox(self):
        """
        #Open the nuclei annotation tool
        """
        # open the nuclei annotation tool
        #self.nuclei_annotation_tool = NucleiAnnotationToolbox(parent=self.trainer_page)
        self.open_nuclei_annotation_tool = NucleiAnnotationToolbox()
        #self.nuclei_annotation_tool.show()
        self.open_nuclei_annotation_tool.show()

    def stop_training(self):
        """
        #Stop the training process
        """
        self.stop_training_process = True

        if hasattr(self, "train_thread"):
            self.train_thread.terminate()
            self.train_thread = None
    
        #QtWidgets.QMessageBox.information(self.trainer_page, "Info", "Training stopped")

        self.training_progress_bar.hide()
        if self.settings["language"] == "eng":
            self.training_status_label.setText("Training stopped")
        elif self.settings["language"] == "de":
            self.training_status_label.setText("Training gestoppt")


    def train_model(self):
        """
        #Train the AI model on the selected trial
        """
        # show the progress bar
        self.training_progress_bar.show()
        self.training_progress_bar.setValue(0)
        if self.settings["language"] == "eng":
            self.training_status_label.setText("Training model ...")
        elif self.settings["language"] == "de":
            self.training_status_label.setText("Modelltraining ...")  

        # get the selected trial
        selected_trial = self.trial_selection_combo_trainer_page.currentText()
        if selected_trial == "Select Trial":
            QtWidgets.QMessageBox.warning(self.trainer_page, "Warning", "Please select a trial to train the model on")
        else:
            # get the selected trial
            selected_train_data_dir = self.trial_selection_combo_trainer_page.currentText()
            train_data_dir = os.path.join(self.ai_train_dir, selected_train_data_dir)
            if isinstance(train_data_dir, tuple):
                train_data_dir = train_data_dir[0]

            # create a train thread
            self.train_thread = TrainThread(parent=self.trainer_page, train_function=self.train_model_on_train_data_dir(train_data_dir))
            self.train_thread.start()
            self.train_thread.finished_signal.connect(self.train_finished_slot)
            self.train_thread.time_left_signal.connect(self.training_ETA_label.setText)
            self.train_thread.status_signal.connect(self.training_status_label.setText)

    def train_finished_slot(self, value):
        print(f"Training finished: {value}")
        self.training_progress_bar.hide()
        if self.settings["language"] == "eng":
            self.training_status_label.setText("Training completed")
            QtWidgets.QMessageBox.information(self.trainer_page, "Info", "Training completed")
        elif self.settings["language"] == "de":
            self.training_status_label.setText("Training abgeschlossen")
            QtWidgets.QMessageBox.information(self.trainer_page, "Info", "Training abgeschlossen")
        

    def setup_train_config(self, train_data_dir):
        """
        #Set up the configuration dictionary
        """
        config_dict = {
            "use_cuda": True,
            "num_classes": 2,
            "lr": 0.001,
            "epochs": 10,
            "batch_size": 1,
            "backbone": "resnet50",
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "data_dir": train_data_dir,
            "checkpoints_dir": "training_checkpoints",
            "do_evaluation_during_training": True,
            "perform_slicing": True,
            "print_freq": 400,
            "eval_freq": 1,
            "do_model_evaluation": False,
            "pretrained": True,
            "trainable_layers": 3,
        }
        return config_dict

    def train_model_on_train_data_dir(self, train_data_dir):
        """
        #Train the AI model on the selected trial
        """
        print(f"Training model on trial: {train_data_dir}")
        # train the model on the selected trial

        # get the configuration dictionary  
        config_dict = self.setup_train_config(train_data_dir)

        # create a model interface with the current trial as output directory
        trainer_object = Trainer(config_dict=config_dict)
        for time_left in trainer_object.run():
            yield time_left, "Training model on trial"
        #trainer_object.run()
    '''    

    def retranslateUi(self, MainPages: QWidget) -> None:
        """
        Retranslate the UI elements based on the selected language.

        This function updates the text of various UI elements to match the selected language.
        It retrieves the language settings from the Settings object and updates the text of labels,
        buttons, and other UI components accordingly.

        Args:
            MainPages (QWidget): The main pages widget containing the UI elements to be retranslated.

        Returns:
            None
        """
        MainPages.setWindowTitle(QCoreApplication.translate("MainPages", u"Form", None))
        self.label.setText(QCoreApplication.translate("MainPages", u"Welcome to AI Cell Detection", None))
        
        self.settings = Settings().items
        self.language = self.settings["language"]
        self.folder_translation = {"eng": ["Select Trial", "All Trials", "No Trials"], "de": ["Versuch auswählen", "Alle Versuche", "Keine Versuche"]}

        if self.language == "eng":
            self.label.setText(QCoreApplication.translate("MainPages", u"Welcome to AI Cell Detection", None))
            #self.import_status_label.setText(QCoreApplication.translate("MainPages", u"Import Status", None))

            self.trial_selection_label_results.setText(QCoreApplication.translate("MainPages", u"Select a trial to view the results", None))
            self.export_results_btn.setText(QCoreApplication.translate("MainPages", u"Export Results", None))

            self.datamanagement_page_label.setText(QCoreApplication.translate("MainPages", u"Data Management", None))
            self.trial_selection_label_datamanagement.setText(QCoreApplication.translate("MainPages", u"Select a trial to manage", None))
            self.delete_trial_btn.setText(QCoreApplication.translate("MainPages", u"Delete Trial", None))

            #self.trainer_page_label.setText(QCoreApplication.translate("MainPages", u"Trainer Page", None))
            #self.trial_selection_label_trainer.setText(QCoreApplication.translate("MainPages", u"Select a trial to train the AI model on", None))
            #self.open_nuclei_annotation_tool_btn.setText(QCoreApplication.translate("MainPages", u"Open Nuclei Annotation Tool", None))
            #self.train_model_btn.setText(QCoreApplication.translate("MainPages", u"Train Model", None))
            #self.stop_training_btn.setText(QCoreApplication.translate("MainPages", u"Stop Training", None))

        elif self.language == "de":
            self.label.setText(QCoreApplication.translate("MainPages", u"Willkommen bei AI Cell Detection", None))
            #self.import_status_label.setText(QCoreApplication.translate("MainPages", u"Importstatus", None))

            self.trial_selection_label_results.setText(QCoreApplication.translate("MainPages", u"W\u00e4hlen Sie einen Versuch aus, um die Ergebnisse anzuzeigen", None))
            self.export_results_btn.setText(QCoreApplication.translate("MainPages", u"Ergebnisse exportieren", None))

            self.datamanagement_page_label.setText(QCoreApplication.translate("MainPages", u"Datenverwaltung", None))
            self.trial_selection_label_datamanagement.setText(QCoreApplication.translate("MainPages", u"W\u00e4hlen Sie einen Versuch aus, um ihn zu verwalten", None))
            self.delete_trial_btn.setText(QCoreApplication.translate("MainPages", u"Versuch l\u00f6schen", None))

            #self.trainer_page_label.setText(QCoreApplication.translate("MainPages", u"Trainerseite", None))
            #self.trial_selection_label_trainer.setText(QCoreApplication.translate("MainPages", u"W\u00e4hlen Sie einen Versuch aus, um das KI-Modell darauf zu trainieren", None))
            #self.open_nuclei_annotation_tool_btn.setText(QCoreApplication.translate("MainPages", u"Nuclei-Annotationstool \u00f6ffnen", None))
            #self.train_model_btn.setText(QCoreApplication.translate("MainPages", u"Modell trainieren", None))
            #self.stop_training_btn.setText(QCoreApplication.translate("MainPages", u"Training stoppen", None))
