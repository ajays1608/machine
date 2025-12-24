
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('logistic_regression_model.joblib')

# Get the feature names from the training data (assuming X_train was a DataFrame)
# This is crucial to ensure the input data to the model has the correct column order.
# You might need to adjust this if X_train was not a DataFrame directly.
# For this example, let's assume we can get them from the original X DataFrame
# or manually define them if X_train columns are lost.

# Based on your notebook state, X is available. Let's use its columns.
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Get data posted as json
        
        # Convert dictionary to DataFrame, ensuring correct column order
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        output = {
            'prediction': int(prediction[0]),
            'probability_no_diabetes': float(prediction_proba[0][0]),
            'probability_diabetes': float(prediction_proba[0][1])
        }
        
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In a Colab environment, we usually run Flask via ngrok for external access.
    # For local testing, you can run app.run(debug=True, port=5000)
    # The actual running via ngrok will be in the next steps.
    print("Flask app 'app.py' created successfully. You'll need to run it in a separate cell.")
