from qt_core import *


style = '''
QLabel {{
	border-radius: {_radius}px;
	border: {_border_size}px solid transparent;
    color: {_color};
}}
QLabel:hover {{
    color: {_color_hover};
}}
'''
# PY LABEL
# ///////////////////////////////////////////////////////////////
class PyLabel(QLabel):
    def __init__(
        self, 
        radius = 8,
        border_size = 2,
        color = "#FFF",
        color_hover = "#FFF"
    ):
        super().__init__()

        # PARAMETERS
        

        # SET STYLESHEET
        self.set_stylesheet(
            radius,
            border_size,
            color,
            color_hover
        )

    # SET STYLESHEET
    def set_stylesheet(
        self,
        radius,
        border_size,
        color,
        color_hover
    ):
        self.setStyleSheet(style.format(
            _radius = radius,
            _border_size = border_size,
            _color = color,
            _color_hover = color_hover
        ))