"""
Image cropper for DL nuclei detection
Version: V 1.0
Author: Daniel Pointner and Michael Kranz
Date: 13.02.2024
Time: 21:15
"""


import numpy as np
import cv2
import os
from pprint import pprint as pp

class ImageCropper():
    def __init__(self, image: np.ndarray = None, image_path: str = None, saving_path: str = None,  
                 crop_left: float = 0, crop_right: float = 0, crop_top: float = 0, crop_bottom: float = 0,
                 set_min: float = 0.0, set_max: float = 0.25, filename: str = None, show_cropped_image: bool = False) -> None:
        """
        Function initializes ImageCropper class
        param: numpy.ndarray image: Image array
        param: str image_path: Path to the image file (image OR image_path is required)
        param: str saving_path: Path to save the cropped image (not required when image instance is needed)
        param: float crop_left: Percentage to crop from the left (value 0.0-0.25)
        param: float crop_right: Percentage to crop from the right (value 0.0-0.25)
        param: float crop_top: Percentage to crop from the top (value 0.0-0.25)
        param: float crop_bottom: Percentage to crop from the bottom (value 0.0-0.25)
        param: float set_min: Defines lower cropping boundary (e.g. 0% of image height/width)
        param: float set_max: Defines upper cropping boundary (e.g. 25% of image height/width)
        param: str filename: Filename used for naming cropped image
        param: bool show_cropped_image: If True, display the cropped image
        return: None
        """
        self.set_min = set_min
        self.set_max = set_max
        self.script_dir = os.getcwd()
        self.crop_left = self.check_boundary(crop_left) # passed via GUI
        self.crop_right = self.check_boundary(crop_right)
        self.crop_top = self.check_boundary(crop_top)
        self.crop_bottom = self.check_boundary(crop_bottom)
        self.show_cropped_image = show_cropped_image
        self.filename = filename
        self.saving_path = saving_path
        # load image if it was not given in the constructor
        if not isinstance(image, np.ndarray):
            self.image_path = image_path
            image = cv2.imread(self.image_path)
        
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image = image
        self.crop_image()

        if isinstance(self.saving_path, str):
            self.cropped_image_file = os.path.join(self.saving_path)
            print(self.cropped_image_file)
            cv2.imwrite(self.cropped_image_file, self.cropped_image)
        
    def height(self) -> int:
        """
        Function to return height of image in number of pixels
        param: self
        returns: int self.image_height
        """
        return self.image_height
    
    def width(self) -> int:
        """
        Function to return width of image in number of pixels
        param: self
        returns: int self.image_width
        """
        return self.image_width
    
    def check_boundary(self, boundary: float = None) -> float:
        """
        Function to check if entry for cropping is below or above given limits.
        Values are adapted to upper boundary if too high and adapted to lower boundary if value below 0
        param: float boundary: Boundary value to check (set_min <= boundary <= set_max)
        returns: float boundary after adapting
        """
        if boundary > self.set_max:
            boundary = self.set_max
        if boundary < self.set_min:
            boundary = self.set_min
        return boundary

    def crop_image(self) -> tuple:
        """
        Function crops image from every side and returns cropped image as well 
        as the coordinates of new corner points.
        Percentage up to set_max*100 % can be cropped from left, right, top, and bottom
        param: self
        returns: tuple: (numpy.ndarray self.cropped_image, tuple self.ulc, tuple self.urc, tuple self.llc, tuple self.lrc)
        """
        crop_left_pix = int(self.crop_left * self.width())
        crop_right_pix = int(self.crop_right * self.width())
        crop_top_pix = int(self.crop_top * self.height())
        crop_bottom_pix = int(self.crop_bottom * self.height())

        """
        Definition of roi

        0,0
        +--------------+
        |              | 
        |   UL-----UR  | 
        |   |      |   | 
        |   LL-----LR  |
        |              | 
        +--------------+ height, width
        UL: upper left corner
        UR: upper right corner
        LL: lower left corner
        LR: lower right corner
        """
        # Definition of corners of cropped image
        self.ulc = (crop_left_pix, crop_top_pix)   
        self.urc = (self.width() - crop_right_pix, crop_top_pix)
        self.llc = (crop_left_pix, self.height() - crop_bottom_pix)
        self.lrc = (self.width() - crop_right_pix, self.height() - crop_bottom_pix)

        # cropped_image = image[StartY:EndY, StartX:EndX]
        self.cropped_image = self.image[self.ulc[1]: self.llc[1], self.ulc[0]: self.urc[0]]
        if self.show_cropped_image:
            cv2.imshow("Cropped Image", self.cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return self.cropped_image, self.ulc, self.urc, self.llc, self.lrc
