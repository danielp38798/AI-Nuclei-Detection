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

# STYLE
# ///////////////////////////////////////////////////////////////
style = '''
QProgressBar {{
    border: 2px solid {_border_color};
    border-radius: {_border_radius};
    text-align: center;
    color: {_color};
}}

QProgressBar::chunk {{
    background-color: {_bg_color};
    width: 20px;
}}
'''


# PY PUSH BUTTON
# ///////////////////////////////////////////////////////////////
class PYProgressBar(QProgressBar):
    def __init__(self, 
                parent = None,
                color = "white",
                border_color = "rgb(52, 59, 72)",
                border_radius = "5px",
                bg_color = "rgb(52, 59, 72)"):
        super().__init__()

        if parent != None:
                self.setParent(parent)
        # APPLY STYLE
        # ///////////////////////////////////////////////////////////////
                # SET STYLESHEET
        custom_style = style.format(
            _border_color = bg_color,
            _border_radius = border_radius,
            _color = color,
            _bg_color = bg_color,

        )
        self.setStyleSheet(custom_style)

        