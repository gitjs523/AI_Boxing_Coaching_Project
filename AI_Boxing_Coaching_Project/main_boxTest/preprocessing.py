"""CLAHE on LAB lightness only. Applied before MoveNet/HSV; toggle with 'c'."""
import cv2

class FramePreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.enabled = False
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def apply(self, bgr):
        if not self.enabled:
            return bgr
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((self.clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
