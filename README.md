#  CBIR - Recherche d'Images Basée sur le Contenu

Application Web de recherche d’images basée sur leur contenu visuel (CBIR) à l’aide de descripteurs d’images (GLCM, Haralick, BiT et combiné). L’interface est développée avec **Streamlit**.

---

## 🧠 Fonctionnalités

- 🔐 Authentification utilisateur
- 📤 Téléversement d'une image requête
- 📊 Choix du descripteur (GLCM, Haralick, BiT, Combiné)
- 📏 Sélection de la mesure de distance (Euclidean, Manhattan, Chebyshev, Canberra)
- 🖼️ Affichage des images les plus similaires
- 📈 Affichage de la distance (ou pourcentage de similarité)

---

## 📁 Structure du Projet

cbir_project/
├── data/
│ ├── dataset/ # Dossiers d’images classées
│ └── features/ # Fichiers .npy contenant les descripteurs extraits
├── descriptors/ # Fichiers des descripteurs
├── streamlit_app/
│ └── main.py # Interface principale de l’application
├── utils/ # Authentification, chargement, similarité
├── run.py # Script pour extraire les caractéristiques
├── requirements.txt
└── README.md


---

## 🚀 Exécution

### 1. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

1- pip install -r requirements.txt




2- python run.py
# Lors de l'exécution de ce script, les quatre fichiers .npy seront recréés automatiquement,
# y compris les deux fichiers volumineux (combined_features.npy et bit_features.npy)
# qui n'ont pas été inclus dans le projet en raison de leur taille dépassant la limite autorisée par GitHub.



3- streamlit run streamlit_app/main.py


🧪 Technologies utilisées
Python 3.10+

OpenCV

NumPy

Streamlit

Mahotas

Scikit-learn

📌 Auteur

Mohand Said Halfaoui#

