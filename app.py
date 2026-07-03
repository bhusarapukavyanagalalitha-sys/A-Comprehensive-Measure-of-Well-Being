import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
with open("HDI.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================================
# Replace this list with the EXACT feature names used to train
# your model.
# ============================================================
FEATURE_NAMES = [
    "Country",
    "Life expectancy",
    "Mean years of schooling",
    "Gross national income (GNI) per capita"
    # Add remaining columns here
]

# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Home", methods=["GET", "POST"])
def my_home():
    return render_template("home.html")


@app.route("/Prediction", methods=["GET", "POST"])
def prediction():
    return render_template("indexnew.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        # Get form values
        input_features = [float(x) for x in request.form.values()]

        # Check feature count
        if len(input_features) != len(FEATURE_NAMES):
            return render_template(
                "resultnew.html",
                prediction_text=f"Expected {len(FEATURE_NAMES)} inputs but received {len(input_features)}."
            )

        # Create dataframe
        df = pd.DataFrame([input_features], columns=FEATURE_NAMES)

        # Predict
        prediction = model.predict(df)

        # Convert output to float
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 2:
                y_pred = float(prediction[0][0])
            else:
                y_pred = float(prediction[0])
        else:
            y_pred = float(prediction)

        y_pred = round(y_pred, 2)

        # HDI Categories
        if 0.30 <= y_pred < 0.40:
            category = "Low HDI"

        elif 0.40 <= y_pred < 0.70:
            category = "Medium HDI"

        elif 0.70 <= y_pred < 0.80:
            category = "High HDI"

        elif 0.80 <= y_pred <= 1.00:
            category = "Very High HDI"

        else:
            category = "Prediction out of expected HDI range"

        return render_template(
            "resultnew.html",
            prediction_text=f"{category} : {y_pred}"
        )

    except Exception as e:
        return render_template(
            "resultnew.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)