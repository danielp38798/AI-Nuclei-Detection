"""
Main Application File

Windows Desktop application for automated nuclei detection and analysis in immunofluorescence images,
GUI based on "PyOneDark_Qt_Widgets_Modern_GUI" 
https://github.com/Wanderson-Magalhaes/PyOneDark_Qt_Widgets_Modern_GUI
"""

# IMPORT PACKAGES AND MODULES
# ///////////////////////////////////////////////////////////////
from gui.uis.windows.main_window.functions_main_window import *
import sys
import os
import json

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# IMPORT QT CORE
# ///////////////////////////////////////////////////////////////
from qt_core import *

from gui.core.functions import Functions

# IMPORT SETTINGS
# ///////////////////////////////////////////////////////////////
from gui.core.json_settings import Settings

# IMPORT PY ONE DARK WINDOWS
# ///////////////////////////////////////////////////////////////
# MAIN WINDOW
from gui.uis.windows.main_window import *

# IMPORT PY ONE DARK WIDGETS
# ///////////////////////////////////////////////////////////////
from gui.widgets import *


# IMPORT QSPLASHSCREEN FOR LOADING SCREEN
# ///////////////////////////////////////////////////////////////
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtGui import QPixmap


# ADJUST QT FONT DPI FOR HIGHT SCALE AN 4K MONITOR
# ///////////////////////////////////////////////////////////////
#os.environ["QT_FONT_DPI"] = "96"
# IF IS 4K MONITOR ENABLE 'os.environ["QT_SCALE_FACTOR"] = "2"'
import sys

from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QMainWindow
#from gui.splashscreen.ui_main import Ui_MainWindow
from gui.splashscreen.ui_splash_screen import Ui_SplashScreen
from gui.splashscreen.widgets import CircularProgress


import sys
import os

# Counter
counter = 0

def get_base_path() -> str:
    """
    Get the base path of the application.

    If the application is run as a bundled executable, the PyInstaller bootloader
    sets a sys._MEIPASS attribute to the path of the temp folder it extracts its
    bundled files to. Otherwise, it returns the directory of the script being run.

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

class SplashScreen(QMainWindow):
    def __init__(self) -> None:
        """Initialize the splash screen."""
        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        # Remove Title bar
        self.setWindowFlags(Qt.FramelessWindowHint)  # type: ignore
        self.setAttribute(Qt.WA_TranslucentBackground)  # type: ignore

        self.progress = CircularProgress()
        self.progress.width = 270  # type: ignore
        self.progress.height = 270  # type: ignore
        self.progress.value = 50
        self.progress.setFixedSize(self.progress.width,  # type: ignore
                                   self.progress.height)
        self.progress.move(15, 15)
        self.progress.font_size = 20
        self.progress.addShadow(True)
        self.progress.setParent(self.ui.centralwidget)
        self.progress.show()
    
        self.show()

        self.main = None

        # Add SHADOW
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(self.shadow)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def update(self) -> None:
        """Update the progress bar and handle the transition to the main window."""
        global counter

        # Set value to progress bar
        self.progress.setValue(counter)

        # Stop counter
        if counter >= 100:
            self.timer.stop()

            # Open a new Window
            self.main = MainWindow()
            self.main.show()

            # Close Splash Screen
            self.close()

        # Increases counter
        counter += 1



# MAIN WINDOW
# ///////////////////////////////////////////////////////////////
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()

        self.base_path = get_base_path()

        # SET LANGUAGE
        # ///////////////////////////////////////////////////////////////
        #self.set_language()

        # SETUP MAIN WINDOW
        # Load widgets from "gui\uis\main_window\ui_main.py"
        # ///////////////////////////////////////////////////////////////
        self.ui = UI_MainWindow(base_path=get_base_path())
        self.ui.setup_ui(self)

        # LOAD SETTINGS
        # ///////////////////////////////////////////////////////////////
        settings = Settings()
        self.settings = settings.items

        # SETUP MAIN WINDOW
        # ///////////////////////////////////////////////////////////////
        self.hide_grips = True # Show/Hide resize grips
        SetupMainWindow.setup_gui(self)

        # STORE ALL FUNCTIONS
        self.functions = Functions()

        self.ui.load_pages.refresh_trial_folder_names()
        
        # SHOW MAIN WINDOW
        # ///////////////////////////////////////////////////////////////
        self.show()
    
    def set_language(self) -> None:
        """Set the language of the application."""
        with open("language.json") as language_file:
            self.language = json.load(language_file)
        

    # LEFT MENU BTN IS CLICKED
    # Run function when btn is clicked
    # Check function by object name / btn_id
    # ///////////////////////////////////////////////////////////////
    def btn_clicked(self) -> None:
        """Handle the event when a button in the left menu is clicked."""
        # GET BT CLICKED
        btn = SetupMainWindow.setup_btns(self)

        # Remove Selection If Clicked By "btn_close_left_column"
        #if btn.objectName() != "btn_settings":
            #self.ui.left_menu.deselect_all_tab()

        # Get Title Bar Btn And Reset Active         
        #top_settings = MainFunctions.get_title_bar_btn(self, "btn_top_settings")
        #top_settings.set_active(False)

        # LEFT MENU
        # ///////////////////////////////////////////////////////////////
        
        # HOME BTN
        if btn.objectName() == "btn_home":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # Load Page 1
            MainFunctions.set_page(self, self.ui.load_pages.welcome_page)

            # set the text of the title bar to the current page
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("Home")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("Startseite")


         # IMPORT BTN
        if btn.objectName() == "btn_import_images":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            MainFunctions.set_page(self, self.ui.load_pages.import_page)

            # set the text of the title bar to the current page
            #self.ui.title_bar.title_label.setText("Import Images")
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("Import Images")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("Bilder importieren")   


        # IMAGE ANALYSIS BTN
        if btn.objectName() == "btn_image_analysis":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # update the trial names every time the page is loaded
            self.ui.load_pages.refresh_trial_folder_names()

            # Load Page 5
            MainFunctions.set_page(self, self.ui.load_pages.image_analysis_page)

            # set the text of the title bar to the current page
            #self.ui.title_bar.title_label.setText("Image Analysis")
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("Image Analysis")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("Bildanalyse")

        # LOAD RESULTS PAGE
        if btn.objectName() == "btn_results":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            self.ui.load_pages.refresh_trial_folder_names()
            # if 
            #if self.ui.load_pages.check_prediction_data() == False:
            MainFunctions.set_page(self, self.ui.load_pages.results_page)
            #else:
                #MainFunctions.set_page(self, self.ui.load_pages.results_page_initial)

            # set the text of the title bar to the current page
            #self.ui.title_bar.title_label.setText("Results")
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("Results")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("Ergebnisse")

        # DATA MANAGEMENT BTN
        if btn.objectName() == "btn_data_management":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            
            self.ui.load_pages.refresh_trial_folder_names()
            # Load Page 6
            MainFunctions.set_page(self, self.ui.load_pages.datamanagement_page)

            # set the text of the title bar to the current page
            #self.ui.title_bar.title_label.setText("Data Management")
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("Data Management")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("Datenverwaltung")


        # AI TRAINER BTN
        if btn.objectName() == "btn_ai_trainer":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            self.ui.load_pages.refresh_train_data_dirs()
            # Load Page 7
            MainFunctions.set_page(self, self.ui.load_pages.trainer_page)

            # set the text of the title bar to the current page
            #self.ui.title_bar.title_label.setText("AI Trainer")
            if self.settings["language"] == "eng":
                self.ui.title_bar.title_label.setText("AI Trainer")
            elif self.settings["language"] == "de":
                self.ui.title_bar.title_label.setText("KI Trainer")

        # INFO BTN
        if btn.objectName() == "btn_info":
            # CHECK IF LEFT COLUMN IS VISIBLE
            if not MainFunctions.left_column_is_visible(self):
                self.ui.left_menu.select_only_one_tab(btn.objectName())

                # Show / Hide
                MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            else:
                if btn.objectName() == "btn_close_left_column":
                    self.ui.left_menu.deselect_all_tab()
                    # Show / Hide
                    MainFunctions.toggle_left_column(self)
                
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # Change Left Column Menu
            if btn.objectName() != "btn_close_left_column":
                if self.settings["language"] == "eng":
                    title = "Information"
                elif self.settings["language"] == "de":
                    title = "Informationen"
                MainFunctions.set_left_column_menu(
                    self, 
                    menu = self.ui.left_column.menus.info_menu,
                    title = title,
                    icon_path = Functions.set_svg_icon("icon_info.svg")
                )

        # SETTINGS LEFT
        if btn.objectName() == "btn_settings" or btn.objectName() == "btn_close_left_column":
            # CHECK IF LEFT COLUMN IS VISIBLE
            if not MainFunctions.left_column_is_visible(self):
                # Show / Hide
                MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            else:
                if btn.objectName() == "btn_close_left_column":
                    self.ui.left_menu.deselect_all_tab()
                    # Show / Hide
                    MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # Change Left Column Menu
            if btn.objectName() != "btn_close_left_column":
                if self.settings["language"] == "eng":
                    title = "Preferences"
                elif self.settings["language"] == "de":
                    title = "Einstellungen"
                MainFunctions.set_left_column_menu(
                    self, 
                    menu = self.ui.left_column.menus.settings_menu_left,
                    title = title,
                    icon_path = Functions.set_svg_icon("icon_settings.svg")
                )
        
        # TITLE BAR MENU
        # ///////////////////////////////////////////////////////////////
        
        # SETTINGS TITLE BAR
        """
        if btn.objectName() == "btn_top_settings":
            # Toogle Active
            if not MainFunctions.right_column_is_visible(self):
                btn.set_active(True)

                # Show / Hide
                MainFunctions.toggle_right_column(self)
            else:
                btn.set_active(False)

                # Show / Hide
                MainFunctions.toggle_right_column(self)

            # Get Left Menu Btn            
            top_settings = MainFunctions.get_left_menu_btn(self, "btn_settings")
            top_settings.set_active_tab(False)    
        """
        

        # DEBUG
        #print(f"Button {btn.objectName()}, clicked!")

    # LEFT MENU BTN IS RELEASED
    # Run function when btn is released
    # Check function by object name / btn_id
    # ///////////////////////////////////////////////////////////////
    def btn_released(self):
        # GET BT CLICKED
        btn = SetupMainWindow.setup_btns(self)

        # DEBUG
        #print(f"Button {btn.objectName()}, released!")

    # RESIZE EVENT
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handle the resize event for the main window.

        This function is called whenever the main window is resized. It ensures that
        the resize grips are adjusted accordingly.

        Args:
            event (QResizeEvent): The resize event object containing details about the resize.
        """
        SetupMainWindow.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

def update_settings(window: SplashScreen) -> None:
    """
    Update the settings of the main window.

    This function retrieves the latest settings and updates the settings
    attribute of the main window if it exists.

    Args:
        window (SplashScreen): The splash screen window containing the main window.
    """
    settings = Settings()
    if hasattr(window, "main") and hasattr(window.main, "settings"):
        window.main.settings = settings.items
        
        
if __name__ == "__main__":
    """
    Main entry point for the application.

    This function initializes the QApplication, sets up the splash screen,
    and starts a timer to update settings every second. It also handles
    exceptions by logging them to a file and exiting the program gracefully.

    Args:
        None

    Returns:
        None
    """
    try:
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon("icon.ico"))
        window = SplashScreen()
        app.processEvents() 

        # Timer to update settings every second
        settings_timer = QTimer()
        settings_timer.timeout.connect(lambda: update_settings(window))
        settings_timer.start(500)  # 1000 milliseconds = 1 second

        sys.exit(app.exec())
    except Exception as e:
        # Store the exception in a file
        error_log = os.path.join(get_base_path(), "error_log.txt")
        if os.path.exists(error_log):
            os.remove(error_log)
        else:
            with open("error_log.txt", "w") as f:
                f.write(str(e))
            # Print the exception
            print(e)
        # Exit the program
        sys.exit(1)

#window = MainWindow()

# EXEC APP
# ///////////////////////////////////////////////////////////////
#sys.exit(app.exec()) # this indention is not working after thread creation of trainer