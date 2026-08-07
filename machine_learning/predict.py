import os
import json
import joblib
import numpy as np


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



with open(
    FEATURE_PATH,
    "r"
) as f:

    features = json.load(f)



scaler = joblib.load(
    SCALER_PATH
)



def load_model(model_name):


    model_path = os.path.join(
        ARTIFACT_DIR,
        f"{model_name}.pkl"
    )


    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"{model_name} model not found"
        )


    return joblib.load(
        model_path
    )





def predict_liver(data):


    selected_model = data.get(
        "model",
        "random_forest"
    )


    model = load_model(
        selected_model
    )



    values = []



    for feature in features:

        values.append(
            data.get(
                feature,
                0
            )
        )



    input_data = np.array(
        values
    ).reshape(
        1,-1
    )



    # Apply scaling

    input_data = scaler.transform(
        input_data
    )



    prediction = model.predict(
        input_data
    )[0]



    probability = model.predict_proba(
        input_data
    )[0]



    confidence = round(
        max(probability)*100,
        2
    )



    if prediction == 1:

        result = "Liver Disease Detected"

    else:

        result = "No Liver Disease"



    return {


        "prediction": result,


        "confidence": confidence,


        "model_used": selected_model


    }





if __name__ == "__main__":


    sample = {


        "model":"random_forest",


        "Age":65,


        "Gender":1,


        "Total_Bilirubin":1.2,


        "Direct_Bilirubin":0.4,


        "Alkaline_Phosphotase":200,


        "Alamine_Aminotransferase":30,


        "Aspartate_Aminotransferase":40,


        "Total_Protiens":7,


        "Albumin":3.5,


        "Albumin_and_Globulin_Ratio":1.0


    }



    print(
        predict_liver(sample)
    )