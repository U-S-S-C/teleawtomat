# screen_capture.py

import numpy as np
import cv2
from mss import mss
from PIL import Image

class ScreenCapture:
    def __init__(self, top=100, left=0, width=400, height=300):
        self.bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
        self.sct = mss()

    def set_area(self, top, left, width, height):
        self.bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}

    def capture_area(self):
        sct_img = self.sct.grab(self.bounding_box)
        return np.array(sct_img)
