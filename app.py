%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load FULL PIPELINE (not only model)
model = joblib.load("logistic_regression_model.joblib")

feature_columns = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Check missing fields
        missing = [col for col in feature_columns if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([data])[feature_columns]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability_no_diabetes": float(proba[0]),
            "probability_diabetes": float(proba[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load FULL PIPELINE (not only model)
model = joblib.load("logistic_regression_model.joblib")

feature_columns = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Check missing fields
        missing = [col for col in feature_columns if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([data])[feature_columns]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "probability_no_diabetes": float(proba[0]),
            "probability_diabetes": float(proba[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
