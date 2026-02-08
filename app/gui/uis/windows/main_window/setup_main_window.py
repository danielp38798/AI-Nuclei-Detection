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
from gui.widgets.py_table_widget.py_table_widget import PyTableWidget
from . functions_main_window import *
import sys
import os

# IMPORT QT CORE
# ///////////////////////////////////////////////////////////////
from qt_core import *

# IMPORT SETTINGS
# ///////////////////////////////////////////////////////////////
from gui.core.json_settings import Settings

# IMPORT THEME COLORS
# ///////////////////////////////////////////////////////////////
from gui.core.json_themes import Themes

# IMPORT PY ONE DARK WIDGETS
# ///////////////////////////////////////////////////////////////
from gui.widgets import *

# LOAD UI MAIN
# ///////////////////////////////////////////////////////////////
from . ui_main import *

# MAIN FUNCTIONS 
# ///////////////////////////////////////////////////////////////
from . functions_main_window import *
#from functions_main_window import MainFunctions

from gui.core.inference import ModelInterface
from gui.core.functions import Functions

# PY WINDOW
# ///////////////////////////////////////////////////////////////
class SetupMainWindow:
    def __init__(self):
        super().__init__()
        # SETUP MAIN WINDOw
        # Load widgets from "gui\uis\main_window\ui_main.py"
        # ///////////////////////////////////////////////////////////////
        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)

    # ADD LEFT MENUS
    # ///////////////////////////////////////////////////////////////

    add_left_menus_en = [
        {
            "btn_icon" : "icon_home.svg",
            "btn_id" : "btn_home",
            "btn_text" : "Home",
            "btn_tooltip" : "Home page",
            "show_top" : True,
            "is_active" : True
        },
        {
            "btn_icon" : "icon_folder_open.svg",
            "btn_id" : "btn_import_images",
            "btn_text" : "Import Images",
            "btn_tooltip" : "Import Images",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_microscope.svg",
            "btn_id" : "btn_image_analysis",
            "btn_text" : "Image Analysis",
            "btn_tooltip" : "Image Analysis",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_show_results.svg",
            "btn_id" : "btn_results",
            "btn_text" : "View/Save Results",
            "btn_tooltip" : "View/Save Results",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "data_management_2_icon_left.svg",
            "btn_id" : "btn_data_management",
            "btn_text" : "Data Management",
            "btn_tooltip" : "Open data management",
            "show_top" : True,
            "is_active" : False

        },
        #{
            #"btn_icon" : "trainer_icon_left.svg",
            #"btn_id" : "btn_ai_trainer",
            #"btn_text" : "AI Trainer",
            #"btn_tooltip" : "Open AI Trainer",
            #"show_top" : True,
            #"is_active" : False

        #},
        {
            "btn_icon" : "icon_info.svg",
            "btn_id" : "btn_info",
            "btn_text" : "Information",
            "btn_tooltip" : "Open informations",
            "show_top" : False,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_settings.svg",
            "btn_id" : "btn_settings",
            "btn_text" : "Settings",
            "btn_tooltip" : "Open settings",
            "show_top" : False,
            "is_active" : False
        }
    ]
    add_left_menus_de = [
        {
            "btn_icon" : "icon_home.svg",
            "btn_id" : "btn_home",
            "btn_text" : "Startseite",
            "btn_tooltip" : "Startseite",
            "show_top" : True,
            "is_active" : True
        },
        {
            "btn_icon" : "icon_folder_open.svg",
            "btn_id" : "btn_import_images",
            "btn_text" : "Bilder importieren",
            "btn_tooltip" : "Bilder importieren",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_microscope.svg",
            "btn_id" : "btn_image_analysis",
            "btn_text" : "Bildanalyse",
            "btn_tooltip" : "Bildanalyse",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_show_results.svg",
            "btn_id" : "btn_results",
            "btn_text" : "Ergebnisse ansehen/speichern",
            "btn_tooltip" : "Ergebnisse ansehen/speichern",
            "show_top" : True,
            "is_active" : False
        },
        {
            "btn_icon" : "data_management_2_icon_left.svg",
            "btn_id" : "btn_data_management",
            "btn_text" : "Datenverwaltung",
            "btn_tooltip" : "Datenverwaltung öffnen",
            "show_top" : True,
            "is_active" : False

        },
        #{
            #"btn_icon" : "trainer_icon_left.svg",
            #"btn_id" : "btn_ai_trainer",
            #"btn_text" : "KI Trainer",
            #"btn_tooltip" : "KI Trainer öffnen",
            #"show_top" : True,
            #"is_active" : False

        #},
        {
            "btn_icon" : "icon_info.svg",
            "btn_id" : "btn_info",
            "btn_text" : "Informationen",
            "btn_tooltip" : "Informationen öffnen",
            "show_top" : False,
            "is_active" : False
        },
        {
            "btn_icon" : "icon_settings.svg",
            "btn_id" : "btn_settings",
            "btn_text" : "Einstellungen",
            "btn_tooltip" : "Einstellungen öffnen",
            "show_top" : False,
            "is_active" : False
        }
    ]
     # ADD TITLE BAR MENUS
    # ///////////////////////////////////////////////////////////////

    add_title_bar_menus_en = [
          {
            "btn_icon" : "icon_settings.svg",
            "btn_id" : "btn_top_settings",
            "btn_tooltip" : "Top settings",
            "is_active" : False
        }
    ]
    add_title_bar_menus_de = [
            {
                "btn_icon" : "icon_settings.svg",
                "btn_id" : "btn_top_settings",
                "btn_tooltip" : "Obere Einstellungen",
                "is_active" : False
            }
    ]

    # SETUP CUSTOM BTNs OF CUSTOM WIDGETS
    # Get sender() function when btn is clicked
    # ///////////////////////////////////////////////////////////////
    def setup_btns(self):
        if self.ui.title_bar.sender() != None:
            return self.ui.title_bar.sender()
        elif self.ui.left_menu.sender() != None:
            return self.ui.left_menu.sender()
        elif self.ui.left_column.sender() != None:
            return self.ui.left_column.sender()

    # SETUP MAIN WINDOW WITH CUSTOM PARAMETERS
    # ///////////////////////////////////////////////////////////////
    def setup_gui(self):
        # APP TITLE
        # ///////////////////////////////////////////////////////////////
        self.setWindowTitle(self.settings["app_name"])
        
        # REMOVE TITLE BAR
        # ///////////////////////////////////////////////////////////////
        if self.settings["custom_title_bar"]:
            self.setWindowFlag(Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground)

        # ADD GRIPS
        # ///////////////////////////////////////////////////////////////
        if self.settings["custom_title_bar"]:
            self.left_grip = PyGrips(self, "left", self.hide_grips)
            self.right_grip = PyGrips(self, "right", self.hide_grips)
            self.top_grip = PyGrips(self, "top", self.hide_grips)
            self.bottom_grip = PyGrips(self, "bottom", self.hide_grips)
            self.top_left_grip = PyGrips(self, "top_left", self.hide_grips)
            self.top_right_grip = PyGrips(self, "top_right", self.hide_grips)
            self.bottom_left_grip = PyGrips(self, "bottom_left", self.hide_grips)
            self.bottom_right_grip = PyGrips(self, "bottom_right", self.hide_grips)

        # LEFT MENUS / GET SIGNALS WHEN LEFT MENU BTN IS CLICKED / RELEASED
        # ///////////////////////////////////////////////////////////////
        # ADD MENUS
        #self.ui.left_menu.add_menus(SetupMainWindow.add_left_menus)

        if self.settings["language"] == "de":
            self.ui.left_menu.add_menus(SetupMainWindow.add_left_menus_de)
        else:
            self.ui.left_menu.add_menus(SetupMainWindow.add_left_menus_en)

        # SET SIGNALS
        self.ui.left_menu.clicked.connect(self.btn_clicked)
        self.ui.left_menu.released.connect(self.btn_released)


        
        # TITLE BAR / ADD EXTRA BUTTONS
        # ///////////////////////////////////////////////////////////////
        # ADD MENUS
        #self.ui.title_bar.add_menus(SetupMainWindow.add_title_bar_menus)

        # SET SIGNALS
        #self.ui.title_bar.clicked.connect(self.btn_clicked)
        #self.ui.title_bar.released.connect(self.btn_released)

        # ADD Title
        if self.settings["custom_title_bar"]:
            self.ui.title_bar.set_title(self.settings["app_name"])
        else:
            #self.ui.title_bar.set_title("Welcome to AI Cell Detection")
            if self.settings["language"] == "eng":
                self.ui.title_bar.set_title("Welcome to AI Nuclei Detection")
            elif self.settings["language"] == "de":
                self.ui.title_bar.set_title("Willkommen bei der KI-Zellkernerkennung")

        # LEFT COLUMN SET SIGNALS
        # ///////////////////////////////////////////////////////////////
        self.ui.left_column.clicked.connect(self.btn_clicked)
        self.ui.left_column.released.connect(self.btn_released)

        # SET INITIAL PAGE / SET LEFT AND RIGHT COLUMN MENUS
        # ///////////////////////////////////////////////////////////////
        MainFunctions.set_page(self, self.ui.load_pages.welcome_page)
        MainFunctions.set_left_column_menu(
            self,
            menu = self.ui.left_column.menus.settings_menu_left,
            title = "Settings Left Column",
            icon_path = Functions.set_svg_icon("icon_settings.svg")
        )
        MainFunctions.set_right_column_menu(self, self.ui.right_column.menu_1)

    # RESIZE GRIPS AND CHANGE POSITION
    # Resize or change position when window is resized
    # ///////////////////////////////////////////////////////////////
    def resize_grips(self):
        if self.settings["custom_title_bar"]:
            self.left_grip.setGeometry(5, 10, 10, self.height())
            self.right_grip.setGeometry(self.width() - 15, 10, 10, self.height())
            self.top_grip.setGeometry(5, 5, self.width() - 10, 10)
            self.bottom_grip.setGeometry(5, self.height() - 15, self.width() - 10, 10)
            self.top_right_grip.setGeometry(self.width() - 20, 5, 15, 15)
            self.bottom_left_grip.setGeometry(5, self.height() - 20, 15, 15)
            self.bottom_right_grip.setGeometry(self.width() - 20, self.height() - 20, 15, 15)