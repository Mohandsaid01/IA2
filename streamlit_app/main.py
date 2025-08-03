import os
import sys
import streamlit as st
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.auth import authenticate
from utils.data_loader import load_images_from_folder
from utils.similarity import calculate_distance
from descriptors import glcm_descriptor, haralick_descriptor, bit_descriptor, combined_descriptor

st.set_page_config(page_title="CBIR - Recherche d'Images")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Connexion")
    user = st.text_input("Nom d'utilisateur")
    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if authenticate(user, pwd):
            st.session_state.authenticated = True
            st.rerun() 
        else:
            st.error("Échec de l'authentification")
else:
    st.title("🔍 CBIR - Recherche par Contenu")
    uploaded_file = st.file_uploader("Uploader une image", type=['jpg', 'png', 'jpeg'])
    descriptor_choice = st.selectbox("Choisir le descripteur", ["GLCM", "Haralick", "BiT", "Combiné"])
    metric_choice = st.selectbox("Choisir la mesure de distance", ["euclidean", "manhattan", "chebyshev", "canberra"])
    num_results = st.slider("Nombre d'images à afficher", 1, 10, 5)

    if uploaded_file:
        query_img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        name_map = {"Combiné": "combined", "GLCM": "glcm", "Haralick": "haralick", "BiT": "bit"}
        features_path = f"data/features/{name_map[descriptor_choice]}_features.npy"
        dataset_path = "data/dataset/"
        try:
            features = np.load(features_path)
        except FileNotFoundError:
            st.error("Fichier de caractéristiques introuvable. Veuillez exécuter `run.py` d'abord.")
            st.stop()

        images, filenames = load_images_from_folder(dataset_path)

        descriptor_func = {
            "GLCM": glcm_descriptor.extract_glcm_features,
            "Haralick": haralick_descriptor.extract_haralick_features,
            "BiT": bit_descriptor.extract_bit_features,
            "Combiné": combined_descriptor.extract_combined_features
        }[descriptor_choice]

        query_feature = descriptor_func(query_img)

        distances = [calculate_distance(query_feature, f, metric_choice) for f in features]
        top_indices = np.argsort(distances)[:num_results]

        st.subheader(" Résultats")
        for i in top_indices:
            distance = distances[i]
            similarity = 1 / (1 + distance) 
            similarity_percentage = similarity * 100
            caption = f"{filenames[i]} | 🔍 Similarité: {similarity_percentage:.2f}%"
            st.image(images[i], caption=caption, width=200)
