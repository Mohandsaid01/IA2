import numpy as np
from .glcm_descriptor import extract_glcm_features
from .haralick_descriptor import extract_haralick_features
from .bit_descriptor import extract_bit_features

def extract_combined_features(image):
    glcm = extract_glcm_features(image)
    haralick = extract_haralick_features(image)
    bit = extract_bit_features(image)
    return np.concatenate([glcm, haralick, bit])
