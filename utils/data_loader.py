import os
import cv2

def load_images_from_folder(folder):
    images = []
    filenames = []

    for root, _, files in os.walk(folder): 
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    filenames.append(os.path.relpath(filepath, 'data/dataset'))
    return images, filenames



