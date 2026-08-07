import os
import json
import joblib
import numpy as np
import pandas as pd


BASE = os.path.dirname(
    os.path.abspath(__file__)
)


ARTIFACT_DIR = os.path.join(
    BASE,
    "artifacts"
)


FEATURE_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_names.json"
)


SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.pkl"
)



# Load feature names

with open(
    FEATURE_PATH,
    "r"
) as f:

    feature_names = json.load(f)



# Load scaler

scaler = joblib.load(
    SCALER_PATH
)



def load_model(model_name):


    model_path = os.path.join(
        ARTIFACT_DIR,
        model_name + ".pkl"
    )


    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "Model not found: " + model_name
        )


    return joblib.load(
        model_path
    )





def predict_liver(patient_data):


    selected_model = patient_data.get(
        "model",
        "random_forest"
    )



    model = load_model(
        selected_model
    )



    input_values = []



    for feature in feature_names:


        input_values.append(
            patient_data.get(
                feature,
                0
            )
        )



    input_array = pd.DataFrame(
    [input_values],
    columns=feature_names
)



    # Apply same scaling used during training

    input_array = scaler.transform(
        input_array
    )



    prediction = model.predict(
        input_array
    )[0]



    probability = model.predict_proba(
        input_array
    )[0]



    confidence = float(
    round(
        max(probability) * 100,
        2
    )
)


    if prediction == 1:

        result = "Liver Disease Detected"

    else:

        result = "No Liver Disease"



    return {

    "result": str(result),

    "confidence": round(float(confidence),2),

    "model_used": str(selected_model)

}