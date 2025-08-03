import mahotas
import cv2

def extract_haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = mahotas.features.haralick(gray).mean(axis=0)
    return features