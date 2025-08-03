import cv2
import numpy as np


def extract_bit_features(image):
    return cv2.resize(image, (32, 32)).flatten() / 255.0
