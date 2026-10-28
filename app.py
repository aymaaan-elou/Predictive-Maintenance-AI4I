import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détection de Fraude Bancaire - XAI", page_icon="🛡️", layout="wide")
st.title("🛡️ Détection de Fraude Bancaire")
st.markdown("Simulez une transaction pour obtenir une prédiction de fraude avec explication (XGBoost + SMOTE).")

# Charger le modèle (mis en cache pour ne pas ralentir l'app)
@st.cache_resource
def load_model():
    return joblib.load('model_fraud.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Récupérer l'ordre exact des colonnes attendues par le modèle
try:
    xgb_model = model.named_steps['model']
    expected_columns = xgb_model.get_booster().feature_names
except Exception as e:
    st.error(f"Erreur lors de l'extraction des noms de colonnes du modèle : {e}")
    st.stop()

# Le seuil optimal trouvé lors de l'entraînement (à ajuster selon votre résultat Colab)
# Si vous n'avez pas sauvegardé le seuil, mettez la valeur par défaut 0.5
BEST_THRESHOLD = 0.85 # Exemple : à remplacer par votre valeur exacte trouvée dans Colab

# ==========================================
# BARRE LATÉRALE : Paramètres de la transaction
# ==========================================
st.sidebar.header("Paramètres de la transaction")

st.sidebar.subheader("Montant et temps")
amount = st.sidebar.number_input("Montant (Amount)", min_value=0.0, max_value=50000.0, value=100.0, step=10.0)
time_val = st.sidebar.number_input("Temps (Time)", min_value=0.0, max_value=200000.0, value=50000.0, step=100.0)

st.sidebar.subheader("Features PCA (V1-V10)")
st.sidebar.caption("Les valeurs V1-V28 sont anonymisées. Ajustez ces curseurs pour simuler une transaction suspecte.")

# Générer les sliders pour les 10 premières features PCA (les plus importantes)
v_values = {}
for i in range(1, 11):
    # On donne une plage réaliste basée sur les données normales (-5 à 5)
    v_values[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, step=0.1)

# ==========================================
# PRÉDICTION
# ==========================================
if st.button("Analyser la transaction", type="primary"):
    # 1. Construire le dictionnaire avec TOUTES les 30 colonnes
    input_dict = {'Time': time_val, 'Amount': amount}
    
    # Remplir V1 à V28 (V1-V10 avec les sliders, V11-V28 à 0 par défaut)
    for i in range(1, 29):
        input_dict[f'V{i}'] = v_values.get(f'V{i}', 0.0)
    
    # 2. Créer le DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 3. CORRECTION : Réaligner les colonnes selon l'ordre exact du modèle
    input_df = input_df[expected_columns]
    
    # 4. Prédiction
    try:
        # Probabilité brute
        probability = model.predict_proba(input_df)[0][1] * 100
        
        # Application du seuil personnalisé (car la fraude est rare, le seuil 0.5 n'est pas idéal)
        prediction = 1 if (probability / 100) >= BEST_THRESHOLD else 0
        
        # 5. Affichage
        st.subheader("Résultat de l'analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ **ALERTE : Transaction suspecte**")
            else:
                st.success(f"✅ **Transaction normale**")
                
        with col2:
            st.metric(label="Probabilité de fraude", value=f"{probability:.1f}%")
            
        if prediction == 1:
            st.warning(f"Le modèle a détecté un risque dépassant le seuil critique de {BEST_THRESHOLD*100}%. Une vérification manuelle est recommandée.")
        else:
            st.info(f"Le risque est en dessous du seuil de {BEST_THRESHOLD*100}%. La transaction peut être validée.")
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.write("Vérifiez que l'ordre et le nombre de colonnes correspondent exactement à ceux de l'entraînement.")

st.markdown("---")
st.caption("Projet réalisé dans le cadre de mon exploration du Master Big Data & IA. Modèle : XGBoost + SMOTE.")
