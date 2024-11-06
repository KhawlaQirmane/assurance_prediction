from pydantic import BaseModel, Field  # Used for data validation
import numpy as np
import pandas as pd  # Used for data manipulation
import joblib  # Used for loading the saved model
from flask import Flask, request, jsonify, render_template

# Load the trained model
model = joblib.load('ELN_model.pkl')

# List of columns used in the model training
model_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                 'region_northwest', 'region_southeast', 'region_southwest'] 

class DonneesEntree(BaseModel):
    age: float  
    sex: str  
    bmi: float  
    children: int  
    smoker: str
    region: str = Field(..., description="Must be one of: southeast, southwest, northwest, northeast")

# Create the Flask app
app = Flask(__name__)

# Endpoint for checking the health of the API
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Endpoint for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Render the HTML form when the user accesses the page
        return render_template('index.html')
    
    if request.method == 'POST':
        if not request.json:
            return jsonify({"erreur": "Aucun JSON fourni"}), 400
        
        try:
            # Extraction et validation des données d'entrée
            donnees = DonneesEntree(**request.json)
            donnees_df = pd.DataFrame([donnees.dict()])  # Convertit en DataFrame

            # One-hot encode les colonnes catégorielles (sex, smoker, region)
            donnees_df = pd.get_dummies(donnees_df, columns=['sex', 'smoker', 'region'], drop_first=True)

            # Ajoute les colonnes manquantes avec des valeurs de zéro
            for col in model_columns:
                if col not in donnees_df.columns:
                    donnees_df[col] = 0

            # Réordonne les colonnes pour correspondre à l'ordre d'entraînement
            donnees_df = donnees_df[model_columns]

            # Utilisation du modèle pour prédire les charges
            predictions = model.predict(donnees_df)

            # Compilation des résultats dans un dictionnaire
            resultats = donnees.dict()
            resultats['prediction'] = float(predictions[0])  # Cast to float for consistency

            # Renvoie les résultats sous forme de JSON
            return jsonify({"resultats": resultats})

        except Exception as e:
            # Gestion des erreurs et renvoi d'une réponse d'erreur
            return jsonify({"erreur": str(e)}), 400

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
