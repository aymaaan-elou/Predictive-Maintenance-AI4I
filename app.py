import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Prédiction de Panne - Stellantis", page_icon="🚗")
st.title("🚗 Maintenance Prédictive des Véhicules Connectés")
st.markdown("Simulez les données de capteurs pour prédire le risque de panne (Projet IA - Inspiré de l'Industrie 4.0).")

# Charger le modèle (mis en cache pour ne pas ralentir l'app)
@st.cache_resource
def load_model():
    return joblib.load('model_maintenance.pkl')

model = load_model()

# Création des curseurs pour l'utilisateur
st.sidebar.header("Paramètres des capteurs")

temp_air = st.sidebar.slider("Température de l'air (K)", 295.0, 305.0, 300.0)
temp_process = st.sidebar.slider("Température du processus (K)", 305.0, 315.0, 310.0)
vitesse = st.sidebar.slider("Vitesse de rotation (rpm)", 1000, 3000, 1500)
couple = st.sidebar.slider("Couple (Nm)", 0.0, 80.0, 40.0)
usure = st.sidebar.slider("Usure de l'outil (min)", 0, 250, 100)

# Bouton de prédiction
if st.button("Prédire le risque de panne"):
    # Préparer les données d'entrée
    input_data = pd.DataFrame([{
        'Air temperature [K]': temp_air,
        'Process temperature [K]': temp_process,
        'Rotational speed [rpm]': vitesse,
        'Torque [Nm]': couple,
        'Tool wear [min]': usure
    }])
    
    # Prédiction
    prediction = model.predict(input_data)[0]
    probabilite = model.predict_proba(input_data)[0][1] * 100
    
    # Affichage du résultat
    st.subheader("Résultat de l'analyse :")
    if prediction == 1:
        st.error(f"⚠️ **Risque de panne élevé ({probabilite:.1f}%)**")
        st.write("Une intervention de maintenance est recommandée.")
    else:
        st.success(f"✅ **Fonctionnement normal (Risque : {probabilite:.1f}%)**")
        st.write("La machine fonctionne dans les paramètres attendus.")