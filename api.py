from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('ELN_model.pkl')

# Créer l'application Flask
app = Flask(__name__)

# Endpoint pour vérifier l'état de l'API
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Endpoint pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données d'entrée
    data = request.get_json(force=True)
    try:
        # Extraire les valeurs des variables explicatives
        features = [data['age'], data['sex'], data['bmi'], data['children'], data['smoker'], data['region']]
        
        # Assurez-vous que les features sont dans le bon ordre attendu par le modèle
        features_array = np.array(features).reshape(1, -1)
        
        # Faire la prédiction
        prediction = model.predict(features_array)
        
        # Retourner la prédiction sous forme de JSON
        return jsonify({'charges': prediction[0]})
    
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400

# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
