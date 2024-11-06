from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('ELN_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create the Flask app
app = Flask(__name__,template_folder='templates')

# Endpoint for checking the health of the API
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract input data from the form
        age = float(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error message if any exception occurs

    # Prepare the input data for prediction (same structure as during training)
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # One-hot encode categorical columns
    input_data_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Ensure the input data has the same columns as the model expects
    expected_columns = ['smoker_yes', 'sex_male', 'region_northwest', 'region_southeast', 'region_southwest', 'age', 'bmi', 'children']
    
    # Reindex the DataFrame to match the model's expected columns, filling missing columns with 0
    input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

    # Separate binary and non-binary data (same as done in your training model)
    binary_data = input_data_encoded[["smoker_yes", "sex_male", "region_northwest", "region_southeast", "region_southwest"]]
    non_binary_data = input_data_encoded.drop(columns=["smoker_yes", "sex_male", "region_northwest", "region_southeast", "region_southwest"])

    # Scale non-binary data using the trained scaler
    non_binary_scaled = scaler.transform(non_binary_data)
    non_binary_scaled_df = pd.DataFrame(non_binary_scaled, columns=non_binary_data.columns)

    # Concatenate scaled non-binary data with binary data
    final_input_data = pd.concat([non_binary_scaled_df, binary_data], axis=1)

    # Make prediction
    prediction = model.predict(final_input_data)
    prediction_result = prediction[0]  # Get the prediction (first value)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_result})

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
