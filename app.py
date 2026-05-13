from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/random_forest_fraud.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(30):
        value = float(request.form[f"V{i}"])
        features.append(value)

    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result = "⚠ Fraudulent Transaction Detected"
    else:
        result = "✅ Legitimate Transaction"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
        



   



