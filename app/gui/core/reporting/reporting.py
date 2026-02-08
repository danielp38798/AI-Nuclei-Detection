"""
Script for data evaluation generated from DL nuclei detection
Author: Daniel Pointer and Michael Kranz
Version: V2.0
Date: 02.02.2024
"""
#====================Imports==============================
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint as pp
from fpdf import FPDF
import numpy as np
import textwrap
#import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import shutil as shutil
import matplotlib
matplotlib.use('agg')
#====================Class definition======================

import sys
import os
from typing import Generator, Tuple


def get_base_path() -> str:
    """
    Determines the base path of the application.

    If the application is run as a bundled executable, the PyInstaller bootloader
    sets a sys._MEIPASS attribute to the path of the temp folder it extracts its
    bundled files to. Otherwise, it uses the directory of the script being run.

    Returns:
        str: The base path of the application.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundled executable, the PyInstaller
        # bootloader sets a sys._MEIPASS attribute to the path of the temp folder it
        # extracts its bundled files to.
        return sys._MEIPASS
    else:
        # Otherwise, just use the directory of the script being run
        return os.path.dirname(os.path.abspath(__file__))
    
class Report(FPDF):
    #================General settings of report========================
    def set_language(self, language: str = "eng") -> None:
        """
        Sets the language for the report.

        Args:
            language (str): The language to set for the report. Defaults to "eng".
                            Accepts "eng" for English and "de" for German.
        """
        if language == "de":
            self.language = "de"
        elif language == "eng" or language == "en":
            self.language = "eng"
        else:
            self.language = "eng"
            print("Invalid language. Adapted to 'English'.")

    def set_header(self, title: str = "Report") -> None:
        """
        Sets the header for the report, including the logo and title.

        Args:
            title (str): The title to set for the report. Defaults to "Report".
        """
        self.base_path = get_base_path()
        logo_path = os.path.join(self.base_path, "application_resources", "OTH_Logo_BFM.png")
        self.image(logo_path, 10, 8, 25)
        self.set_font('helvetica', 'B', 20)
        self.cell(0, 10, title, border=False, ln=True, align='C')
        self.ln(25)

    def set_footer(self) -> None:
        """
        Sets the footer for the report, including the page number and copyright information.
        """
        self.set_y(-35)
        self.set_font('helvetica', 'I', 10)
        if self.language == "de":
            self.cell(0, 10, f'(c) Daniel Pointer und Michael Kranz \t Seite {self.page_no()}/{{nb}}', align='R')
        else:
            self.cell(0, 10, f'(c) Daniel Pointer and Michael Kranz \t Page {self.page_no()}/{{nb}}', align='R')

    def set_chapter_title(self, num_chapter: int = 0, title_chapter: str = "chapter", link: int = None) -> None:
        """
        Sets the title for a chapter in the report.

        Args:
            num_chapter (int): The chapter number. Defaults to 0.
            title_chapter (str): The title of the chapter. Defaults to "chapter".
            link (int): The link to the chapter. Defaults to None.
        """
        self.set_link(link)
        self.set_font('helvetica', '', 20)
        chapter_title = f'{num_chapter}. {title_chapter}'
        self.cell(0, 5, chapter_title, ln=True)
        self.ln()

    #====================Make content for report=======================
    def make_section(self, num_chapter: int = 0, num_section: int = 0, title_section: str = "section", link: int = None) -> None:
        """
        Creates a section in the report.

        Args:
            num_chapter (int): The chapter number. Defaults to 0.
            num_section (int): The section number. Defaults to 0.
            title_section (str): The title of the section. Defaults to "section".
            link (int): The link to the section. Defaults to None.
        """
        self.set_link(link)
        self.set_font('helvetica', '', 16)
        section_title = f'{num_chapter}.{num_section} {title_section}'
        self.cell(0, 5, section_title, ln=True)
        self.ln()

    def make_chapter(self, num_chapter: int = 0, title_chapter: str = "chapter", link: int = None) -> None:
        """
        Creates a chapter in the report.

        Args:
            num_chapter (int): The chapter number. Defaults to 0.
            title_chapter (str): The title of the chapter. Defaults to "chapter".
            link (int): The link to the chapter. Defaults to None.
        """
        self.set_chapter_title(num_chapter=num_chapter, title_chapter=title_chapter, link=link)

    def make_settings_footer(self, script_version: float = 0, date: str = None) -> None:
        """
        Creates the footer with settings information for the report.

        Args:
            script_version (float): The version of the script. Defaults to 0.
            date (str): The date of the report. Defaults to None.
        """
        self.set_font('helvetica', '', 12)
        self.ln(10)
        if self.language == "de":
            self.cell(0, 10, f'erstellt mit Version:\t {script_version}', ln=True, align='L')
            self.cell(0, 10, f'Datum:\t {date}', ln=True, align='L')
        else:
            self.cell(0, 10, f'Created with version:\t {script_version}', ln=True, align='L')
            self.cell(0, 10, f'Date:\t {date}', ln=True, align='L')
            
            
class Reporting():
    """
    Class Reporting reads 'image_analysis_results.json' and extracts its attributes.
    When creating an object, additional information can be given, e.g., unit='mm' or 'µm'/'um' and
    verbose=True/False; verbose=True: display of all print commands.
    """
    #===================Initialize class=====================================
    def __init__(self, unit: str = 'mm', trial_dir: str = "imports", hp_params_path: str = "hyperparameter.json", 
                 language: str = "eng", verbose: bool = False) -> None:
        """
        Initializing function for class Reporting. Retrieves data from results JSON file and casts them into lists.
        Information from JSON file is stored in self.results.

        Args:
            unit (str): Unit of measurement, default is "mm".
            trial_dir (str): Directory containing the trial data, default is "imports".
            hp_params_path (str): Path to the hyperparameters JSON file, default is "hyperparameter.json".
            language (str): Language for the report, default is "eng" for English or "de" for German. Other entries will be adapted to English.
            verbose (bool): If True, displays all print commands, default is False.
        """
        self.script_dir = os.getcwd()
        self.temp_dir = os.path.join(self.script_dir, "temp")
        self.results_dir = os.path.join(self.script_dir, trial_dir, "_aux_files")
        self.hp_params_path = hp_params_path

        if language == "de":
            self.language = "de"
        elif language == "eng" or language == "en":
            self.language = "eng"
        else:
            self.language = "eng"
            print("Invalid language. Adapted to 'English'.")

        self.results = {}

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        results_json = os.path.join(self.results_dir, "image_analysis_results.json")

        if not os.path.exists(results_json):
            print("No results file found!")
            return
        else:
            with open(os.path.join(self.results_dir, "image_analysis_results.json")) as instances_results:
                self.results = json.load(instances_results)
                self.num_images = len(self.results["image_data"])
                self.class_names = self.results["class_names"]
                if unit == "mm":
                    self.unit = 0  # 1 for µm (um) and 0 for mm
                    self.str_unit = "mm"
                elif unit == "um" or unit == 'µm':
                    self.unit = 1
                    self.str_unit = "um"
                else:  # in case of invalid unit entry -> set to mm
                    self.unit = 0
                    self.str_unit = "mm"
                    print("Invalid units: Changed to 'mm'")
                self.verbose = verbose
            instances_results.close()

        if not os.path.exists(self.hp_params_path):
            print("No hyperparameters file found!")
            return
        else:
            with open(self.hp_params_path) as hp:
                self.hyperparams = json.load(hp)
                
    def get_filenames(self) -> list:
        """
        Returns a list of filenames (image files) extracted from the results JSON file.

        Returns:
            list: A list of filenames.
        """
        self.list_filenames = [filename for filename in self.results["image_data"].keys() if self.results["image_data"][filename] != {}]
        if self.verbose:
            pp(self.list_filenames)
        return self.list_filenames
    
    def get_processing_date(self) -> str:
        """
        Returns the date of processing in the format dd.mm.YYYY.

        Returns:
            str: The processing date.
        """
        self.processing_date = self.results["date"]
        if self.verbose:
            pp(self.processing_date)
        return self.processing_date
    
    def check_results_file_integrity(self) -> bool:
        """
        Checks if the results file is complete and has all necessary information.

        Returns:
            bool: True if the results file is complete, False otherwise.
        """
        if self.results == {}:
            print("No results file found!")
            return False
        if self.results["image_data"] == {}:
            print("No image data found in results file!")
            return False
        else:
            return True

    #===================Get hyperparameter from json file==============================
    def get_model_name(self) -> str:
        """
        Retrieves the model name from the hyperparameters.

        Returns:
            str: The model name.
        """
        self.model_name = self.hyperparams["model_description"]
        return self.model_name

    def get_weight_decay(self) -> float:
        """
        Retrieves the weight decay from the hyperparameters.

        Returns:
            float: The weight decay.
        """
        self.weight_decay = self.hyperparams["weight_decay"]
        return self.weight_decay

    def get_backbone(self) -> str:
        """
        Retrieves the backbone from the hyperparameters.

        Returns:
            str: The backbone.
        """
        self.backbone = self.hyperparams["backbone"]
        return self.backbone

    def get_batch_size(self) -> int:
        """
        Retrieves the batch size from the hyperparameters.

        Returns:
            int: The batch size.
        """
        self.batch_size = self.hyperparams["batch_size"]
        return self.batch_size

    def get_lr(self) -> float:
        """
        Retrieves the learning rate from the hyperparameters.

        Returns:
            float: The learning rate.
        """
        self.lr = self.hyperparams["lr"]
        return self.lr

    def get_epochs(self) -> int:
        """
        Retrieves the number of epochs from the hyperparameters.

        Returns:
            int: The number of epochs.
        """
        self.epochs = self.hyperparams["epochs"]
        return self.epochs

    def get_optimizer(self) -> str:
        """
        Retrieves the optimizer from the hyperparameters.

        Returns:
            str: The optimizer.
        """
        self.optimizer = self.hyperparams["optimizer"]
        return self.optimizer

    def get_momentum(self) -> float:
        """
        Retrieves the momentum from the hyperparameters.

        Returns:
            float: The momentum.
        """
        self.momentum = self.hyperparams["momentum"]
        return self.momentum

    def get_pretrained_weights_url(self) -> str:
        """
        Retrieves the URL of the pretrained weights from the hyperparameters.

        Returns:
            str: The URL of the pretrained weights.
        """
        self.pretrained_weights_url = self.hyperparams["pretrained_weights_url"]
        return self.pretrained_weights_url

    def get_pretrained_weights_name(self) -> str:
        """
        Retrieves the name of the pretrained weights from the hyperparameters.

        Returns:
            str: The name of the pretrained weights.
        """
        self.pretrained_weights_name = self.hyperparams["pretrained_weights_name"]
        return self.pretrained_weights_name

    def get_used_model_for_analysis(self) -> str:
        """
        Retrieves the model name used for analysis from the hyperparameters.

        Returns:
            str: The model name used for analysis.
        """
        self.used_model = self.hyperparams["model_name"]
        return self.used_model


    #===================Get result data from json file==============================
    def get_image_area(self, unit: str = None) -> float | None:
        """
        Returns the total image area from the results JSON for the first image.

        Args:
            unit (str, optional): 'mm2' or 'um2'. Defaults to self.unit.

        Returns:
            float | None: Image area in the specified unit.
        """
        try:
            first_image = next(iter(self.results["image_data"].values()))
            first_class = next(iter(first_image.values()))
            area_key = f"image_area_{self.str_unit}2"
            return first_class.get(area_key)
        except Exception as e:
            print(f"Failed to get image area: {e}")
            return None


    def get_count_for_class(self, image_file_name: str, class_id: int) -> int | None:
        """
        Returns the object count for a specific class in a given image.
        """
        class_info = self.results["image_data"].get(image_file_name, {}).get(str(class_id))
        if not class_info:
            print(f"No data for class {class_id} in image {image_file_name}")
            return None
        return class_info.get("count")


    def get_area_for_class(self, image_file_name: str, class_id: int, unit: str = None) -> float | None:
        """
        Returns the area of a specific class in the given image.
        """
        class_info = self.results["image_data"].get(image_file_name, {}).get(str(class_id))
        print(class_info)
        if not class_info:
            print(f"No data for class {class_id} in image {image_file_name}")
            return None
        area = class_info.get(f"area_{self.str_unit}2")
        return area


    def get_relative_area_for_class(self, image_file_name: str, class_id: int) -> float | None:
        """
        Returns the relative area of a specific class in the given image.
        """
        class_info = self.results["image_data"].get(image_file_name, {}).get(str(class_id))
        if not class_info:
            print(f"No data for class {class_id} in image {image_file_name}")
            return None
        return class_info.get("area_relative")


    def get_density_for_class(self, image_file_name: str, class_id: int, unit: str = None) -> float | None:
        """
        Returns the density of a specific class in the given image.
        """
        #unit = unit or self.unit
        class_info = self.results["image_data"].get(image_file_name, {}).get(str(class_id))
        if not class_info:
            print(f"No data for class {class_id} in image {image_file_name}")
            return None
        return class_info.get(f"density_per_{self.str_unit}2")

    def get_used_unit(self) -> str:
        """
        Returns the unit of measurement used for postprocessing.

        Returns:
            str: The unit of measurement ("mm" or "µm").
        """
        return self.str_unit

    def get_used_language(self) -> str:
        """
        Returns the language used for postprocessing.

        Returns:
            str: The language ("eng" or "de").
        """
        return self.language
    
    #====================Make graphs for report==============================
    def make_graph(self, data_list_1: list = None, data_list_2: list = None, x_label: str = "x", y_label: str = "Images",
                   name_graph: str = "fig", legend_title: str = "Classes", title: str = None, dpi: int = 200, grid: bool = False) -> bool:
        """
        Function for plotting a stacked horizontal bar graph. A bar for each label.

        Args:
            data_list_1 (list): Data for the first set of bars. Default is None.
            data_list_2 (list): Data for the second set of bars. Default is None.
            x_label (str): Label for the x-axis. Default is "x".
            y_label (str): Label for the y-axis. Default is "Images".
            name_graph (str): Name of the graph file to be saved. Default is "fig".
            legend_title (str): Title for the legend. Default is "Classes".
            title (str): Title of the graph. Default is None.
            dpi (int): Dots per inch for the saved graph image. Default is 200.
            grid (bool): Whether to display a grid on the graph. Default is False.

        Returns:
            bool: True after successful execution.
        """
        if data_list_2 is not None:
            if self.language == "de":
                legend_title = "Klassen"
                y_label = "Bilder"
            else:
                legend_title = "Classes"
                y_label = "Images"
            cm = 1/2.54  # centimeters in inches
            fig = plt.figure() #figsize=(10*cm, 5*cm))
            plt.rcParams["figure.autolayout"] = True
            wrapped_filenames = [textwrap.fill(filename, 20) for filename in self.list_filenames]
            g1 = plt.barh(wrapped_filenames, data_list_2, color="blue")
            g2 = plt.barh(wrapped_filenames, data_list_1, left=data_list_2, color="orange")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            # set text size
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
    
            if grid:
                plt.grid()
            plt.legend([g1, g2], self.class_names, title=legend_title)
            plt.title(title)
            temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
            plt.savefig(temp_dir, dpi=dpi)
            plt.close()
            return True
        else:
            if self.language == "de":
                legend_title = "Klassen"
                y_label = "Bilder"
            else:
                legend_title = "classes"
                y_label = "Images"
            cm = 1/2.54
            fig = plt.figure()
            plt.rcParams["figure.autolayout"] = True
            wrapped_filenames = [textwrap.fill(filename, 20) for filename in self.list_filenames]
            g1 = plt.barh(wrapped_filenames, data_list_1, color="orange")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            if grid:
                plt.grid()
            plt.legend([g1], self.class_names, title=legend_title)
            plt.title(title)
            temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
            plt.savefig(temp_dir, dpi=dpi)
            plt.close()
            return True
        
    def make_graph_for_multiple_classes(self, data_list: list = None, x_label: str = "X", y_label: str = "Y", 
                                        name_graph: str = "fig", legend_title: str = "classes", title: str = None, 
                                        dpi: int = 200, grid: bool = False) -> bool:
        """
        Function for plotting a stacked horizontal bar graph for multiple classes.

        Args:
            data_list (list): List of data for each class. Each sublist represents data for one class.
            x_label (str): Label for the x-axis. Default is "X".
            y_label (str): Label for the y-axis. Default is "Y".
            name_graph (str): Name of the graph file to be saved. Default is "fig".
            legend_title (str): Title for the legend. Default is "classes".
            title (str): Title of the graph. Default is None.
            dpi (int): Dots per inch for the saved graph image. Default is 200.
            grid (bool): Whether to display a grid on the graph. Default is False.

        Returns:
            bool: True after successful execution.
        """
        cm = 1/2.54
        fig = plt.figure()
        plt.rcParams["figure.autolayout"] = True
        wrapped_filenames = [textwrap.fill(filename, 20) for filename in self.list_filenames]

        # every row in data_list is the value for one class
        for i in range(len(data_list)):
            plt.barh(wrapped_filenames, data_list[i], left=np.sum(data_list[:i], axis=0))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        if grid:
            plt.grid()
        plt.legend(self.class_names, title=legend_title)
        plt.title(title)
        temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
        plt.savefig(temp_dir, dpi=dpi)
        plt.close()
        return True
    
    def make_graph_plotly(self, data_list_1: list = None, data_list_2: list = None, x_label: str = "x", y_label: str = "Images", 
                            name_graph: str = "fig", legend_title: str = "Cell classes", title: str = None, dpi: int = 200, 
                            grid: bool = False) -> bool:
        """
        Function for plotting a stacked horizontal bar graph using Plotly.

        Args:
            data_list_1 (list): Data for the first set of bars. Default is None.
            data_list_2 (list): Data for the second set of bars. Default is None.
            x_label (str): Label for the x-axis. Default is "x".
            y_label (str): Label for the y-axis. Default is "Images".
            name_graph (str): Name of the graph file to be saved. Default is "fig".
            legend_title (str): Title for the legend. Default is "Cell classes".
            title (str): Title of the graph. Default is None.
            dpi (int): Dots per inch for the saved graph image. Default is 200.
            grid (bool): Whether to display a grid on the graph. Default is False.

        Returns:
            bool: True after successful execution.
        """

        if self.language == "de":
            legend_title = "Zellklassen"
            y_label = "Bilder"

        wrapped_filenames = [textwrap.fill(filename, 20) for filename in self.list_filenames]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=wrapped_filenames, x=data_list_2, name="Decondensed Nuclei", orientation='h'))
        fig.add_trace(go.Bar(y=wrapped_filenames, x=data_list_1, name="Nuclei", orientation='h', 
                                marker=dict(color="orange"), offset=data_list_2))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=legend_title,
            barmode='stack',
            showlegend=True,
            #grid=dict(visible=grid),
            autosize=True
        )

        temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
        fig.write_image(temp_dir, width=dpi, height=dpi)
        return True
    
    def make_boxplot(self, data_list_1: list = None, data_list_2: list = None, x_label: str = "X", y_label: str = "Y",
                     name_graph: str = "fig", legend_title: str = "classes", label_plot_2: str = "Plot 1", 
                     label_plot_1: str = "Plot 2", title: str = None, dpi: int = 200, grid: bool = False) -> bool:
        """
        Function for plotting a boxplot. If two data lists are provided, it plots both; otherwise, it plots only the first data list.

        Args:
            data_list_1 (list): Data for the first plot. Default is None.
            data_list_2 (list): Data for the second plot. Default is None.
            x_label (str): Label for the x-axis. Default is "X".
            y_label (str): Label for the y-axis. Default is "Y".
            name_graph (str): Name of the graph file to be saved. Default is "fig".
            legend_title (str): Title for the legend. Default is "classes".
            label_plot_2 (str): Label for the second plot. Default is "Plot 1".
            label_plot_1 (str): Label for the first plot. Default is "Plot 2".
            title (str): Title of the graph. Default is None.
            dpi (int): Dots per inch for the saved graph image. Default is 200.
            grid (bool): Whether to display a grid on the graph. Default is False.

        Returns:
            bool: True after successful execution.
        """
        if self.language == "de":
            legend_title = "Klassen"
            y_label = "Bilder"
        else:
            legend_title = "Classes"
            y_label = "Images"
        plt.figure()
        if data_list_2 is not None and sum(data_list_2) != 0:
            data_list_1 = np.array(data_list_1).transpose()
            data_list_2 = np.array(data_list_2).transpose()
            data = [data_list_2, data_list_1]
            plt.boxplot(data)
            if grid:
                plt.grid()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks([1, 2], [label_plot_2, label_plot_1])
        else:
            data_list_1 = np.array(data_list_1).transpose()
            plt.boxplot(data_list_1)
            if grid:
                plt.grid()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks([1], [label_plot_1])
      
        temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
        plt.savefig(temp_dir, dpi=dpi)
        plt.close()
        return True

    def make_boxplot_for_mutliple_classes(self, data_list: list = None, x_label: str = "X", y_label: str = "Y",
                                          name_graph: str = "fig", legend_title: str = "classes", title: str = None, 
                                          dpi: int = 200, grid: bool = False) -> bool:
        """
        Function for plotting a boxplot for multiple classes.

        Args:
            data_list (list): List of data for each class. Each sublist represents data for one class.
            x_label (str): Label for the x-axis. Default is "X".
            y_label (str): Label for the y-axis. Default is "Y".
            name_graph (str): Name of the graph file to be saved. Default is "fig".
            legend_title (str): Title for the legend. Default is "classes".
            title (str): Title of the graph. Default is None.
            dpi (int): Dots per inch for the saved graph image. Default is 200.
            grid (bool): Whether to display a grid on the graph. Default is False.

        Returns:
            bool: True after successful execution.
        """
        plt.figure()
        data = np.array(data_list).transpose()
        plt.boxplot(data)
        plt.xticks([i+1 for i in range(len(data_list))], self.class_names)
        if grid:
            plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        temp_dir = os.path.join(self.temp_dir, f"{name_graph}.png")
        plt.savefig(temp_dir, dpi=dpi)
        plt.close()
        return True
    
    #====================Misc================================================
    def delete_temp_files(self) -> None:
        """
        Deletes all temporary files in the temp directory.

        This function iterates through all files in the temp directory and attempts to delete them.
        If a file is open or cannot be deleted, it catches the exception and prints an error message.
        Finally, it forcefully deletes the entire temp directory.

        Args:
            self: The instance of the Reporting class.
        """
        # force delete of temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=False)

    def get_class_names(self) -> list:
        """
        Returns the class names.

        This function retrieves the class names from the results JSON file.

        Args:
            self: The instance of the Reporting class.

        Returns:
            list: A list of class names.
        """
        return self.class_names

    def get_class1_to_class2_ratio(self) -> float:
        """
        Returns the ratio of class 1 to class 2.

        This function calculates the ratio of the sum of counts for class 1 to the sum of counts for class 2.

        Args:
            self: The instance of the Reporting class.

        Returns:
            float: The ratio of class 1 to class 2.
        """
        class1 = sum(self.get_decond_nuclei_counts())
        class2 = sum(self.get_nuclei_counts())
        ratio = class1 / class2
        return ratio
    
     ##====================Report generation=====================================
    def generate_report_for_trial(self, unit: str, language: str, trial_dir: str, hp_params_path: str):
        """
        Generates a report for a single trial.

        This function initializes the Reporting class, checks the integrity of the results file,
        extracts necessary data, creates plots and a PDF report, and saves the report in the trial directory.

        Args:
            unit (str): Unit of measurement, either "mm" or "µm".
            language (str): Language for the report, either "eng" for English or "de" for German.
            trial_dir (str): Directory containing the trial data.
            hp_params_path (str): Path to the hyperparameters JSON file.

        Yields:
            Tuple[int, str, str, bool]: A tuple containing the progress percentage, status message, trial name, and success value.
        """
        reporting = Reporting(unit=unit, 
                            language=language,
                            trial_dir=trial_dir, 
                            hp_params_path=hp_params_path, 
                            verbose=False)
        trial_name = os.path.basename(trial_dir)
        if reporting.check_results_file_integrity():
            list_file_names = reporting.get_filenames()
            date = reporting.get_processing_date()
            image_size = reporting.get_image_area()
            unit = reporting.get_used_unit()
            language = reporting.get_used_language()

            # -----------------Create report---------------------
            class_names = reporting.get_class_names()
            classes_info = {int(class_id): {"name": [],
                                            "count": [],
                                            "area": [],
                                            "area_relative": [],
                                            "density": []} for class_id in range(1, len(class_names)+1)}

            # create plots for each class
            for image_file in list_file_names:
                for class_id in range(1, len(class_names)+1):
                    classes_info[class_id]["name"].append(class_names[class_id-1])
                    classes_info[class_id]["count"].append(reporting.get_count_for_class(image_file, class_id))
                    classes_info[class_id]["area"].append(reporting.get_area_for_class(image_file, class_id))
                    classes_info[class_id]["area_relative"].append(reporting.get_relative_area_for_class(image_file, class_id))
                    classes_info[class_id]["density"].append(reporting.get_density_for_class(image_file, class_id))

            data_list_counts = [classes_info[class_id]["count"] for class_id in range(1, len(class_names)+1)]
            data_list_area = [classes_info[class_id]["area"] for class_id in range(1, len(class_names)+1)]
            data_list_area_relative = [classes_info[class_id]["area_relative"] for class_id in range(1, len(class_names)+1)]
            data_list_density = [classes_info[class_id]["density"] for class_id in range(1, len(class_names)+1)]

            # create boxplot for all classes
            if reporting.language == "eng":
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_counts, x_label="Classes", y_label="Counts",
                                    name_graph=f"001_Counts_boxplot", title=f"Counts of all classes in all n = {len(list_file_names)} images")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_area, x_label="Classes", y_label="Area",
                                        name_graph=f"002_Area_covered_boxplot", title=f"Area of all classes in all n = {len(list_file_names)} images")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_area_relative, x_label="Classes", y_label="Area relative",
                                        name_graph=f"003_Area_covered_relative_boxplot", title=f"Relative area of all classes in all n = {len(list_file_names)} images")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_density, x_label="Classes", y_label="Density",
                                        name_graph=f"004_Density_boxplot", title=f"Density of all classes in all n = {len(list_file_names)} images")
                
                reporting.make_graph_for_multiple_classes(data_list=data_list_counts, x_label="Counts", y_label="Images",
                                        name_graph=f"005_Class_counts", title=f"Counts in all n = {len(list_file_names)} images")
                reporting.make_graph_for_multiple_classes(data_list=data_list_area, x_label="Area", y_label="Images",
                                        name_graph=f"006_Class_area", title=f"Area in all n = {len(list_file_names)} images")
                reporting.make_graph_for_multiple_classes(data_list=data_list_area_relative, x_label="Area relative", y_label="Images",
                                        name_graph=f"007_Class_area_relative", title=f"Relative area in all n = {len(list_file_names)} images")
                reporting.make_graph_for_multiple_classes(data_list=data_list_density, x_label="Density", y_label="Images",
                                        name_graph=f"008_Class_density", title=f"Density in all n = {len(list_file_names)} images")
            elif reporting.language == "de":
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_counts, x_label="Klassen", y_label="Zählungen",
                                    name_graph=f"001_Counts_boxplot", title=f"Zählungen aller Klassen in allen n = {len(list_file_names)} Bildern")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_area, x_label="Klassen", y_label="Fläche",
                                        name_graph=f"002_Area_covered_boxplot", title=f"Fläche aller Klassen in allen n = {len(list_file_names)} Bildern")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_area_relative, x_label="Klassen", y_label="Fläche relativ",
                                        name_graph=f"003_Area_covered_relative_boxplot", title=f"Relative Fläche aller Klassen in allen n = {len(list_file_names)} Bildern")
                reporting.make_boxplot_for_mutliple_classes(data_list=data_list_density, x_label="Klassen", y_label="Dichte",
                                        name_graph=f"004_Density_boxplot", title=f"Dichte aller Klassen in allen n = {len(list_file_names)} Bildern")
                
                reporting.make_graph_for_multiple_classes(data_list=data_list_counts, x_label="Zählungen", y_label="Bilder",
                                        name_graph=f"005_Class_counts", title=f"Zählungen in allen n = {len(list_file_names)} Bildern")
                reporting.make_graph_for_multiple_classes(data_list=data_list_area, x_label="Fläche", y_label="Bilder",
                                        name_graph=f"006_Class_area", title=f"Fläche in allen n = {len(list_file_names)} Bildern")
                reporting.make_graph_for_multiple_classes(data_list=data_list_area_relative, x_label="Fläche relativ", y_label="Bilder",
                                        name_graph=f"007_Class_area_relative", title=f"Relative Fläche in allen n = {len(list_file_names)} Bildern")
                reporting.make_graph_for_multiple_classes(data_list=data_list_density, x_label="Dichte", y_label="Bilder",
                                        name_graph=f"008_Class_density", title=f"Dichte in allen n = {len(list_file_names)} Bildern")
            
            temp_dir = reporting.temp_dir
            trail_name = os.path.basename(os.path.basename(reporting.results_dir))
            report_file = os.path.join(reporting.results_dir, f"Report_{trail_name}.pdf")
            report = Report('P', 'mm', 'A4')
            report.set_language(language=language)
            report.add_page()
           
            report.set_header(title=f"Report: {trial_name}")
            # creation of document links
            chapter_1_link = report.add_link()
            chapter_2_link = report.add_link()
            # table of contents
            report.ln(10)
            report.set_font('helvetica', '', 18)
            if language == "de":
                report.cell(0, 10, 'Inhaltsverzeichnis', ln=True)
            else:
                report.cell(0, 10, 'Table of contents', ln=True)
            report.ln(10)
            report.set_font('helvetica', '', 12)
            if hasattr(reporting, 'hyperparams'):
                report.cell(0, 10, '1. Hyperparameter', ln=True, link=chapter_1_link)
                if language == "de":
                    report.cell(0, 10, '2. Allgemeine Informationen', ln=True, link=chapter_2_link)
                else:
                    report.cell(0, 10, '2. General Information', ln=True, link=chapter_2_link)
            else:
                if language == "de":
                    report.cell(0, 10, '1. Allgemeine Informationen', ln=True, link=chapter_2_link)
                else:
                    report.cell(0, 10, '1. General Information', ln=True, link=chapter_2_link)
            report.ln(120)
            report.make_settings_footer(date=date, script_version=1.0)
            report.set_footer()
            report.set_title("Report")
            report.set_author("Daniel Pointer and Michael Kranz 2024")
            report.alias_nb_pages() # required for page numbering
            report.ln(20)
            
            if hasattr(reporting, 'hyperparams'):
                # ----------------- Page 2 ----------------
                report.set_header(title=f"Report: {trial_name}")
                report.ln(20)
                if language == "de":
                    report.make_chapter(num_chapter=1, title_chapter="Für das Training verwendete Hyperparameter", link=chapter_1_link)
                else:
                    report.make_chapter(num_chapter=1, title_chapter="Hyperparameter used for model training", link=chapter_1_link)
                
                # table 1: hyperparameter
                report.set_font('helvetica', '', 12)
                if language == "de":
                    report.cell(0, 10, 'Tabelle 1: Für Prozessierung verwendete Hyperparameter.', ln=True, align='L')
                else:
                    report.cell(0, 10, 'Table 1: Used hyperparameters for processing.', ln=True, align='L')
                report.cell(70, 10, 'Model' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_model_name()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Learning rate' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_lr()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Epochs' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_epochs()), border=1, align='L', ln=True)

                report.cell(70, 10, 'Batch size' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_batch_size()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Optimizer' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_optimizer()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Momentum' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_momentum()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Weight decay' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_weight_decay()), border=1, align='L', ln=True)

                report.cell(70, 10, 'Backbone' , border=1, align='L', ln=False)
                report.cell(100, 10, str(reporting.get_backbone()), border=1, align='L', ln=True)
                
                report.cell(70, 10, 'Pretrained Weights' , border=1, align='L', ln=False)
                report.cell(100,10, str(reporting.get_pretrained_weights_name()), border=1, align='L', ln=True, link=str(reporting.get_pretrained_weights_url()))

                report.set_footer()
                report.ln(10)

            # ----------------- Page 3 ----------------
            report.set_header(title=f"Report: {trial_name}")
            report.ln(10)
            if language == "de":
                report.make_chapter(num_chapter=2, title_chapter="Allgemeine Informationen", link=chapter_2_link)
            else:
                report.make_chapter(num_chapter=2, title_chapter="General Information", link=chapter_2_link)
            
            # table 1: hyperparameter
            report.set_font('helvetica', '', 12)
            # include all generated plots in report
            for idx, image_file in enumerate(os.listdir(temp_dir)):
                if ".png" in image_file: # skip Report.pdf
                    if idx != 0:
                        report.set_header(title=f"Report: {trial_name}")
                    # read image and calculate width
                    img = Image.open(os.path.join(temp_dir, image_file))
                    image_width, image_height = img.size
                    # calculate image width in mm
                    image_width = image_width * 0.264583

                    # Calculate the x position for the image to be centered
                    # Get the width of the page
                    page_width = report.w
                    
                    # avoid page overflow
                    if image_width > page_width:
                        x = (page_width - 150) / 2
                    else:
                        x = (page_width - image_width) / 2
                    report.image(name=os.path.join(temp_dir, image_file), x=x, w=150)
                    report.ln(20)
                    report.set_footer()
                    report.ln(10)

                    img.close()
            
            # save file and open it
            try:
                report.output(report_file)
                os.startfile(report_file)
                print(f"Report for trial {trial_name} created successfully!")
                status = f"Report generation successful for trial: '{trial_name}'"
                success_value = True
                return success_value, status, trial_name
            except Exception as e:
                print(f"Error {e}: Could not create report")
                status = f"Error {e}: Could not create report for trial: '{trial_name}'"
                success_value = False
                return success_value, status, trial_name
        
        else:
            print("Results file is incomplete. Report generation failed for trial: " + trial_name)
            if language == "de":
                status = f"Ergebnisdatei ist unvollständig. Reportgenerierung fehlgeschlagen für Versuch: '{trial_name}'"
            else:
                status = f"Results file is incomplete. Report generation failed for trial: '{trial_name}'"
            success_value = False
            
            return success_value, status, trial_name
    

    
        


if __name__ == "__main__":
    report = Report('P', 'mm', 'A4')
    report.set_language(language="de")