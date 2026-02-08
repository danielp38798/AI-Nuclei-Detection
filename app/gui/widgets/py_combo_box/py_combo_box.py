from qt_core import *


# Custom ComboBox
style = """ 
QComboBox {

    border: 2px solid rgb(52, 59, 72);
    border-radius: 5px;
    padding: 5px;
    background-color: rgb(27, 29, 35);
    color: rgb(255, 255, 255);
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 3px;
    border-left-color: rgba(39, 44, 54, 150);
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}


QComboBox QAbstractItemView {
    border: 2px solid rgb(52, 59, 72);
    border-radius: 5px;
    background-color: rgb(27, 29, 35);
    color: rgb(255, 255, 255);
    selection-background-color: rgb(39, 44, 54);
}
"""


class PyComboBox(QComboBox):
    def __init__(self, *args):
        super().__init__()

        # Set the current index
        self.setCurrentIndex(0)

        # Set the font
        self.setFont(QFont("Arial", 12))

        # Set the style
        self.setStyleSheet(style)

        # Set the size
        self.setMinimumHeight(35)
        self.setMinimumWidth(100)
        self.setMaximumWidth(150)

