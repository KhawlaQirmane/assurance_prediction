from pydantic import BaseModel, Field  # Utilisé pour la validation des données
import numpy as np
import pandas as pd  # Utilisé pour la manipulation de données
import joblib  # Utilisé pour charger le modèle sauvegardé
from flask import Flask, request, jsonify, render_template

# Charger le modèle
model = joblib.load('ELN_model.pkl')

class DonneesEntree(BaseModel):
    age: float  
    sex: str  
    bmi: float  
    children: int  
    smoker: str
    region: str = Field(..., description="Must be one of: southeast, southwest, northwest, northeast")
    
    

# Créer l'application Flask
app = Flask(__name__)



# Endpoint pour vérifier l'état de l'API

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})



# Endpoint pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400
    
    
    try:
        # Extraction et validation des données d'entrée en utilisant Pydantic
        donnees = DonneesEntree(**request.json)
        donnees_df = pd.DataFrame([donnees.dict()])  # Conversion en DataFrame

        # Utilisation du modèle pour prédire et obtenir les probabilités
        predictions = model.predict(donnees_df)

        # Compilation des résultats dans un dictionnaire
        resultats = donnees.dict()
        resultats['prediction'] = int(predictions[0])
        
        # Renvoie les résultats sous forme de JSON
        return jsonify({"resultats": resultats})
    except Exception as e:
        # Gestion des erreurs et renvoi d'une réponse d'erreur
        return jsonify({"erreur": str(e)}), 400
    
    
# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
