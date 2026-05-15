from flask import Flask, render_template, request

import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)


# Load ML Models
rf_model = joblib.load("models/random_forest_fraud.pkl")

smote_model = joblib.load("models/fraud_smote_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # User Inputs
    amount = float(request.form["Amount"])

    risk_level = request.form["risk_level"]

    selected_model = request.form["model_name"]


    # Generate hidden simulated PCA features
    random_features = np.random.normal(0, 1, 28)


    # Simulate suspicious transaction behavior
    if risk_level == "High":
        random_features *= 3

    elif risk_level == "Medium":
        random_features *= 1.5


    # Generate random transaction timeline value
    time = np.random.randint(0, 172000)


    # Final feature structure
    features = [time]

    features.extend(random_features)

    features.append(amount)


    # Convert into DataFrame
    input_df = pd.DataFrame([features])


    # Model Selection
    if selected_model == "Random Forest":
        model = rf_model

    else:
        model = smote_model


    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]


    # Result Message
    if prediction == 1:
        result = "Fraudulent Transaction Detected"

    else:
        result = "Legitimate Transaction"


    # Return Output
    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability * 100, 2),
        selected_model=selected_model
    )


if __name__ == "__main__":
    app.run(debug=True)