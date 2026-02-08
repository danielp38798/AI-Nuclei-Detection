__version__ = "0.11.16"

"""
from . import annotation
from . import auto_model
from . import models 
from . import prediction


"""

from gui.core.sahi.annotation import BoundingBox, Category, Mask
from gui.core.sahi.auto_model import AutoDetectionModel
from gui.core.sahi.models.base import DetectionModel
from gui.core.sahi.prediction import ObjectPrediction

__all__ = ['BoundingBox', 'Category', 'Mask', 'AutoDetectionModel','DetectionModel', 'ObjectPrediction']
