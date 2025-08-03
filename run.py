import os
import numpy as np
import cv2
import importlib.util


def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_dir = os.path.dirname(os.path.abspath(__file__))
desc_dir = os.path.join(base_dir, 'descriptors')
utils_dir = os.path.join(base_dir, 'utils')


glcm = import_from_path("glcm", os.path.join(desc_dir, "glcm_descriptor.py"))
haralick = import_from_path("haralick", os.path.join(desc_dir, "haralick_descriptor.py"))
bit = import_from_path("bit", os.path.join(desc_dir, "bit_descriptor.py"))
loader = import_from_path("loader", os.path.join(utils_dir, "data_loader.py"))


def extract_combined_features(image):
    return np.concatenate([
        glcm.extract_glcm_features(image),
        haralick.extract_haralick_features(image),
        bit.extract_bit_features(image)
    ])


dataset_path = os.path.join("data", "dataset")
images, _ = loader.load_images_from_folder(dataset_path)


features_path = os.path.join("data", "features")
os.makedirs(features_path, exist_ok=True)


print("Extracting GLCM features...")
glcm_features = np.array([glcm.extract_glcm_features(img) for img in images])
np.save(os.path.join(features_path, "glcm_features.npy"), glcm_features)

print("Extracting Haralick features...")
haralick_features = np.array([haralick.extract_haralick_features(img) for img in images])
np.save(os.path.join(features_path, "haralick_features.npy"), haralick_features)

print("Extracting BiT features...")
bit_features = np.array([bit.extract_bit_features(img) for img in images])
np.save(os.path.join(features_path, "bit_features.npy"), bit_features)

print("Extracting Combined features...")
combined_features = np.array([extract_combined_features(img) for img in images])
np.save(os.path.join(features_path, "combined_features.npy"), combined_features)

print(" Done! All features saved to 'data/features/'")
